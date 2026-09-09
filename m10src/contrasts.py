"""The contrast step: read the screen's registered contrasts off the arms' per-query COV files.

This is the step `run_arm.f_verdict` refuses to do for itself -- *"this runner never guesses the
student"* -- and the one the registry's `_runner_contract` warns about: **`cov_macro.contrast`
takes `quantile` as a parameter and the caller MUST pass each contrast's own value.** F1 is
two-sided at alpha/24 per tail; the other eleven are one-sided at alpha/12. A single stale default
here decides eleven contrasts, one of which (`A4-A3`) controls whether the 834,463 generated queries
enter the build, so the quantile is read from the registry per contrast and cross-checked against
`cov_macro`'s constants rather than assumed.

What it does NOT do: pick a recipe. It emits each contrast's point estimate, one-sided bound,
resolution label and the arms it was read off. The family rules (`multi_arm_winner`, `E_cost`,
`D_tie`, `three_outcome`, `generated_half_in_build`) are applied to those labels by
`m10src/decision_rules.py` and the registry's `outcome_to_action`, and the recipe lock records the
result.

Read points. A contrast with `at_examples` reads the cov file whose label carries `read<N>`
(family F's 20M read); every other contrast reads its arms' FINAL cycle end. Sign stability is the
registered clause "the sign is stable across the last two cycle-end checkpoints", so the point
estimate is recomputed at the previous cycle end -- no bootstrap, the clause is about the sign.
Family A's two contrasts are exempt (`rules.sign_stability_exemption`) and use `lower > MDE`
instead of the generic bar.

Arms that have not run (`E-bs128` is CLOUD_ONLY) and arms that are CUT (`C-M9init`) produce no
contrast: they are reported as `not_computed` with the reason, never as unresolved, because
`rules.arm_failure` and `rules.C_skipped` are different dispositions and the Bonferroni
denominator is 12 either way.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cov_macro
from run_arm import REGISTRY, RESULTS, WORK, sha256_file, slug

REPO = Path(__file__).resolve().parents[1]
F_VERDICT = RESULTS / "m10_F_verdict.json"


def cfg():
    return json.loads(REGISTRY.read_text())


# ------------------------------------------------------------------------- reading COV files ----

def cov_files(arm):
    """{label: path} for the arm's cycle-end reads, in cycle order. Mid-cycle reads are excluded:
    the registered clause names CYCLE ENDS, and `cov_mid*` files exist for the plateau rule."""
    d = WORK / slug(arm)

    def cycle_no(p):
        return int(p.stem[len("cov_cycle"):].split("_")[0])   # lexicographic breaks at cycle10

    return {p.stem[len("cov_"):]: p for p in sorted(d.glob("cov_cycle*.json"), key=cycle_no)}


def read_point(arm, at_examples=None):
    """-> (label, {unit: {qid: score}}). `at_examples` selects the `read<N>`-tagged cycle end."""
    files = cov_files(arm)
    if not files:
        raise FileNotFoundError(f"{arm}: no cov_cycle*.json under {WORK / slug(arm)}")
    if at_examples:
        want = f"read{int(at_examples)}"
        hit = [l for l in files if l.endswith(want)]
        if len(hit) != 1:
            raise KeyError(f"{arm}: {len(hit)} cycle-end reads tagged {want!r}, expected 1 "
                           f"(have {list(files)})")
        label = hit[0]
    else:
        label = list(files)[-1]
    return label, json.loads(files[label].read_text())["per_unit_query"]


def previous_cycle_end(arm, label):
    """The cycle end before `label`, for the sign-stability clause. None if there is none."""
    order = list(cov_files(arm))
    i = order.index(label)
    if i == 0:
        return None, None
    prev = order[i - 1]
    return prev, json.loads(cov_files(arm)[prev].read_text())["per_unit_query"]


def check_against_arm_record(arm, label):
    """The COV file being read must be the exact file the arm's COMMITTED record hashes.

    `cov_files` reads whatever is in the arm directory. A re-run, a stale directory or a smoke that
    overwrote a real artifact would be read silently -- and "smokes overwrite real artifacts" is a
    hazard this project has been bitten by twice (`calib.run_arm` wrote the real `P0.json`;
    `harvest.draw` wrote the real `harvest_draw.json`). The arm record stores
    `per_query_scores_sha256` per checkpoint, so this is an exact identity check, not a heuristic.
    """
    p = RESULTS / f"m10_arm_{slug(arm)}.json"
    if not p.exists():
        raise FileNotFoundError(f"{arm}: no committed arm record at {p}")
    rec = json.loads(p.read_text())
    if rec.get("status") != "complete":
        raise ValueError(f"{arm}: arm record status {rec.get('status')!r}, not 'complete'")
    want = {c["label"]: c.get("per_query_scores_sha256")
            for c in rec.get("cov", {}).get("per_checkpoint", []) if isinstance(c, dict)}
    if label not in want:
        raise ValueError(f"{arm}: the record names checkpoints {sorted(want)}, not {label!r}")
    got = sha256_file(cov_files(arm)[label])
    if want[label] and got != want[label]:
        raise ValueError(f"{arm}/{label}: the file on disk hashes to {got[:12]}… and the committed "
                         f"arm record says {want[label][:12]}…; the directory is stale or was "
                         f"overwritten")


# ------------------------------------------------------------------------------- the contrast ---

def quantile_for(cid, c, boot):
    """Each contrast's OWN quantile, cross-checked against `cov_macro`'s constants.

    The registry default is alpha/n_contrasts and a two-sided contrast carries its own half of it.
    Both are asserted against `cov_macro.ONE_SIDED`/`F_PER_TAIL` so a registry edit and the
    implementation cannot drift apart silently -- the exact drift that left 11 contrasts pointed at
    the stale alpha/13.
    """
    q = c.get("quantile", boot["quantile"])
    tails = c.get("tails", 1)
    want = cov_macro.F_PER_TAIL if tails == 2 else cov_macro.ONE_SIDED
    if abs(q - want) > 1e-15:
        raise ValueError(f"{cid}: registry quantile {q} != cov_macro's {want} for tails={tails}")
    return q, tails


def point_estimate(a_scores, b_scores, unit_family):
    aligned = cov_macro.align(a_scores, b_scores, unit_family)
    w = cov_macro.weights(unit_family)
    return float(sum(w[u] * (x - y).mean() for u, (_q, x, y) in aligned.items())), aligned


def compute(cid, reg=None, B=None, verbose=True):
    """-> the contrast record, written to `results/m10_contrast_<cid>.json`."""
    reg = reg or cfg()
    c = reg["contrasts"][cid]
    st, boot = reg["statistics"], reg["statistics"]["bootstrap"]
    alias = reg["anchor_aliases"]
    arms = reg["arms"]
    q, tails = quantile_for(cid, c, boot)
    a_name, b_name = alias.get(c["a"], c["a"]), alias.get(c["b"], c["b"])

    def not_computed(reason):
        """A disposition is an ARTIFACT, not an absence. Without a file on disk, `selection` cannot
        tell "this contrast was never read" from "this contrast has no artifact yet", and the two
        take different actions."""
        rec = {"contrast": cid, "registered_spec": c, "not_computed": reason,
               "registry_sha256": sha256_file(REGISTRY)}
        (RESULTS / f"m10_contrast_{cid}.json").write_text(json.dumps(rec, indent=1) + "\n")
        return rec

    for nm, side in ((a_name, c["a"]), (b_name, c["b"])):
        e = arms.get(nm, {})
        if e.get("cut"):
            return not_computed(f"{side} -> {nm} is CUT under W8 band 1 (`rules.C_skipped`); the "
                                f"Bonferroni denominator is unchanged at {boot['n_contrasts']}")
        if e.get("pending"):
            return not_computed(f"{side} -> {nm} is registered `pending: {e['pending']}` and has "
                                f"not run; the disposition is REGISTERED, not inferred from a "
                                f"missing file (`arms.{nm}._pending`)")
        if not cov_files(nm):
            return not_computed(f"{side} -> {nm} has no cycle-end COV read: the arm has not run on "
                                f"this machine (`E-bs128` is CLOUD_ONLY, `outcome_to_action.E`)")

    at = c.get("at_examples")
    la, sa = read_point(a_name, at)
    lb, sb = read_point(b_name, at)
    for nm, lab in ((a_name, la), (b_name, lb)):
        check_against_arm_record(nm, lab)
    uf = dict(cov_macro.SURFACE)
    point, aligned = point_estimate(sa, sb, uf)

    # -- orientation. F1 is read highest-minus-runner-up (`rules.F_orientation`), which is why it
    # is two-sided; every other contrast keeps the registry's own a - b.
    flipped = False
    if c.get("family_rule") == "F_tournament" and point < 0:
        a_name, b_name, la, lb, sa, sb = b_name, a_name, lb, la, sb, sa
        point, aligned = point_estimate(sa, sb, uf)
        flipped = True

    r = cov_macro.contrast(aligned, uf, B=B or boot["B"], seed=boot["seed"], quantile=q,
                           method=boot["quantile_method"], chunk=boot["chunk"])

    # -- sign stability across the last two cycle ends
    pa, psa = previous_cycle_end(a_name, la)
    pb, psb = previous_cycle_end(b_name, lb)
    prev_point = None
    if psa is not None and psb is not None:
        prev_point, _ = point_estimate(psa, psb, uf)
    # ONE source of truth for the exemption: the registered rule name, never a hardcoded id.
    # `rules.sign_stability_exemption` names family A's two contrasts, and those two are exactly
    # the ones carrying an A rule -- so the rule name IS the exemption.
    A_RULES = ("three_outcome", "generated_half_in_build")
    exempt = c.get("rule") in A_RULES
    sign_stable = None if prev_point is None else (prev_point > 0) == (point > 0)

    MDE = st["MDE"]
    lower = r["lower_bound_raw"]
    rule = c.get("rule")
    if rule in A_RULES:
        # family A: the corrected bar is `lower > MDE`, and sign stability does not apply
        resolved = lower > MDE
        label = ("RESOLVED" if resolved else
                 "POSITIVE, NOT RESOLVED" if point >= MDE and lower > 0 else "NOT POSITIVE")
    else:
        # `is True`, not `is not False`: an arm with a single cycle end has sign_stable None, and
        # a clause that requires the last TWO cycle ends is not satisfied by having one. No live
        # effect today (all ten computed contrasts have both), and that is why it needed a test.
        resolved = bool(point >= MDE and lower > 0 and sign_stable is True)
        label = "RESOLVED" if resolved else "NOT RESOLVED"

    out = {
        "contrast": cid,
        "_what": c.get("_w9") or c.get("note") or "",
        "registered_spec": c,
        "registry_sha256": sha256_file(REGISTRY),
        "arms": {"a": a_name, "b": b_name, "orientation": "highest-minus-runner-up (F_orientation)"
                 if c.get("family_rule") == "F_tournament" else "registry a - b",
                 "flipped_from_registry_order": flipped},
        "read": {"a_label": la, "b_label": lb, "at_examples": at,
                 "prev_cycle_end": {"a": pa, "b": pb}},
        "rule": rule, "family_rule": c.get("family_rule"),
        "resolve": {"MDE": MDE, "point_ge_MDE": bool(point >= MDE),
                    "lower_bound_gt_0": bool(lower > 0),
                    "lower_bound_gt_MDE": bool(lower > MDE),
                    "sign_stable_last_two_cycle_ends": sign_stable,
                    "sign_stability_exempt": exempt,
                    "point_at_prev_cycle_end": prev_point,
                    "resolved": resolved, "label": label},
        "macro": {a_name: cov_macro.macro(sa, uf)[0], b_name: cov_macro.macro(sb, uf)[0]},
        "by_family": {a_name: cov_macro.macro(sa, uf)[1], b_name: cov_macro.macro(sb, uf)[1]},
        "power_note": f"distance_raw {r['distance_raw']:.6f} against the registered resolution "
                      f"distance {st['measured_resolution_distance']} "
                      f"(`results/m10_cov_resolution.json`), which was measured between UNRELATED "
                      f"models and was expected to over-estimate a real contrast's width. "
                      f"Disclosure only: it sizes nothing and alpha does not move (amendment A4).",
    }
    out.update(r)
    p = RESULTS / f"m10_contrast_{cid}.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    if verbose:
        print(f"{cid:8} {a_name:14} - {b_name:14} point {point:+.6f} lower {lower:+.6f} "
              f"dist {r['distance_raw']:.6f}  {label}")
    return out


# ------------------------------------------------------------------------------- the F verdict --

def f_verdict_from(f1, reg=None, verbose=True):
    """Re-issue `results/m10_F_verdict.json` from an F1 record, under the CURRENT registry hash.

    `run_arm.f_verdict` pins `registry_sha256`, so every registry amendment invalidates the verdict
    and it is re-issued from the same contrast -- the amendment of 2026-09-09 changed no statistic,
    so the numbers below are F1's own, unchanged.
    """
    from run_arm import f_arms
    reg = reg or cfg()
    win = f1["arms"]["a"] if f1["delta_raw"] > 0 else f1["arms"]["b"]
    student = reg["arms"][win]["student"]
    recs = sorted(sha256_file(RESULTS / f"m10_arm_{slug(n)}.json") for n in f_arms(reg))
    v = {
        "winner": student,
        "_winner_note": "the registry `student` string of the arm with the higher 20M COV macro "
                        "whose margin RESOLVED under F_tournament; serve_cost_order was not "
                        "invoked, so this is evidence and not a product preference",
        "contrast": {
            "rule": "F_tournament / F1, two-sided at alpha/24 per tail per F_orientation, read at "
                    f"{f1['registered_spec']['at_examples']:,} examples, oriented "
                    "highest-minus-runner-up",
            "point": f1["delta_raw"], "lower": f1["lower_bound_raw"],
            "resolved": f1["resolve"]["resolved"], "distance": f1["distance_raw"],
            "MDE": f1["resolve"]["MDE"],
            "sign_stable_last_two_cycle_ends": f1["resolve"]["sign_stable_last_two_cycle_ends"],
            "B": f1["B"], "seed": f1["seed"], "quantile": f1["quantile"],
            "quantile_method": f1["quantile_method"],
            "n_units": len(f1["n_by_unit"]), "n_paired_queries": sum(f1["n_by_unit"].values()),
            "draws_sha256": f1["draws_sha256"], "plan_sample_sha256": f1["plan_sample_sha256"],
            "artifact": "results/m10_contrast_F1.json"},
        "registry_sha256": sha256_file(REGISTRY),
        "sha256_of_F_records": recs,
        "_f_arms": f_arms(reg),
        "_what": "F's verdict, read by run_arm.f_verdict: every arm after family F trains on F's "
                 "WINNER backbone (anchor.init \"F's winner backbone\", Recipe amendment A6). "
                 "Written by the CONTRAST step, never by the runner.",
    }
    F_VERDICT.write_text(json.dumps(v, indent=1) + "\n")
    if verbose:
        print(f"F verdict re-issued: winner {student} under registry {v['registry_sha256'][:12]}…")
    return v


# ------------------------------------------------------------------------------- the selection --

def _read(cid):
    p = RESULTS / f"m10_contrast_{cid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def selection(reg=None, verbose=True):
    """Apply the registry's family rules to the computed contrasts -> the selected recipe.

    Every branch here is a REGISTERED rule read from `m10/screen_registry.json` and, where it has
    an executable form, from `m10src/decision_rules.py`. Nothing is decided in this function: a
    family whose contrasts do not resolve takes its registered default, and a family whose arms
    have not run is PENDING -- which is not the same disposition as unresolved and must not be
    silently collapsed into one (`E-bs128` is CLOUD_ONLY and E1 has not been read).
    """
    import decision_rules as DR
    reg = reg or cfg()
    got = {cid: _read(cid) for cid in reg["contrasts"]}
    computed = {cid: v for cid, v in got.items() if v and not v.get("not_computed")}

    def res(cid):
        v = computed.get(cid)
        return bool(v and v["resolve"]["resolved"])

    def pt(cid):
        v = computed.get(cid)
        return v["delta_raw"] if v else None

    def best(pairs):
        """`rules.multi_arm_winner`: highest point estimate among the alternatives that RESOLVE,
        and **ties on the point estimate go to the default** -- which means `max()` on tuples is
        wrong, because it breaks a tie on the label string instead. No live effect today (nothing
        in G or B resolved), which is exactly why it needed writing down."""
        if not pairs:
            return None
        top = max(v for v, _ in pairs)
        winners = [n for v, n in pairs if v == top]
        return winners[0] if len(winners) == 1 else None      # a tie falls through to the default

    sel, why = {}, {}
    # F -- the student. Already read; the verdict file is the artifact run_arm enforces.
    if computed.get("F1"):
        f1 = computed["F1"]
        if f1["resolve"]["resolved"]:
            sel["student"] = reg["arms"][f1["arms"]["a"]]["student"]
            why["student"] = "F1 RESOLVED (rules.F_tournament): the higher 20M COV macro, adopted "\
                             "as evidence"
        else:
            # `rules.serve_cost_order`: cheapest to serve, by parameter count then layer count then
            # bge-small. A PRODUCT PREFERENCE, and the report must label it as one.
            sel["student"] = "MiniLM-L6-v2"
            why["student"] = ("F1 did NOT resolve, so `rules.serve_cost_order` applies: the "
                              "cheapest to serve is MiniLM-L6 at 23,893,888 params against "
                              "bge-small's 34,540,672 (`results/m10_student_parity_box.json`). "
                              "**This is a PRODUCT PREFERENCE, not evidence, and the report says "
                              "so.**")
    # A -- the corpus. `generated_half_in_build`: not resolved -> A3's corpus, generated dropped.
    if computed.get("A4-A3"):
        keep = res("A4-A3")
        sel["corpus"] = "A4 (harvest + the 834,463 generated queries)" if keep else "A3 (harvest only)"
        why["corpus"] = ("A4-A3 " + computed["A4-A3"]["resolve"]["label"] +
                         " (rules.generated_half_in_build). NOTE rules.generated_half_in_build and "
                         "`_interpretation.exposure`: family A's arms are form-balanced, so this "
                         "contrast tests RE-ALLOCATING 58% of query presentations to seven "
                         "generated forms, not 'does synthetic data help'.")
    # G -- the head. multi_arm_winner over G2/G3 against the 1152 default; G-384 is not selectable.
    g = [(pt(c), n) for c, n in (("G2", "1536-wide linear"), ("G3", "1152-wide MLP")) if res(c)]
    sel["head"] = best(g) or "1152-wide linear (the anchor default)"
    why["head"] = ("rules.multi_arm_winner over G2/G3; G1 (1152 vs 384) is " +
                   (computed["G1"]["resolve"]["label"] if computed.get("G1") else "not computed") +
                   " and `arms.G-384.selectable: false`, so G1 is evidence about M9's 384-wide "
                   "diagnosis and selects nothing.")
    # B -- the mix.
    b = [(pt(c), n) for c, n in (("B1", "100/0"), ("B2", "50/50")) if res(c)]
    sel["mix"] = best(b) or "75/25 (the anchor default)"
    why["mix"] = ("rules.multi_arm_winner over B1/B2. `_interpretation.B`: B is an augmentation "
                  "experiment at FIXED query exposure, so limit what its result decides.")
    # D -- the objective, through the AMENDED tie rule.
    d = DR.d_selection_corrected(res("D1"), pt("D1") or 0.0, res("D2"), pt("D2") or 0.0)
    sel["objective"] = d
    why["objective"] = ("rules.D_tie as amended 2026-09-09 (`decision_rules.d_selection_corrected`)"
                        ". REQUIRED with D2: `_interpretation.D1_D2_precondition` -- D-COV's "
                        "gradients are ~500x smaller on identical batches "
                        "(`results/m10_dcov_gradient_audit.json`), so D2 is confounded with an "
                        "optimizer-scale difference and is not a clean test of its own hypothesis.")
    # E -- the batch, on the COST rule, and PENDING while its arm has not run.
    if computed.get("E1"):
        sel["batch"] = DR.e_after_confirmation_corrected(res("E1"), pt("E1"))
        why["batch"] = "rules.E_cost; E is exempt from quality confirmation (amended 2026-09-09)"
    else:
        sel["batch"] = "PENDING"
        why["batch"] = ("E1 is NOT COMPUTED, not unresolved: `E-bs128` is CLOUD_ONLY and runs on "
                        "the rented A100 (`outcome_to_action.E`). `rules.E_cost` selects bs128 in "
                        "every case except a RESOLVED bs32 win, and an unread contrast cannot "
                        "resolve -- but reading the rule that way would take E's decision from a "
                        "measurement that was never made. The recipe lock carries E as PENDING.")

    out = {"_what": "the selected recipe, from the registered family rules applied to the computed "
                    "contrasts. `m10src/contrasts.selection`; nothing is decided here.",
           "registry_sha256": sha256_file(REGISTRY),
           "selected": sel, "why": why,
           "contrasts": {cid: (v.get("not_computed") if v.get("not_computed") else
                               {"point": v["delta_raw"], "lower": v["lower_bound_raw"],
                                "label": v["resolve"]["label"], "resolved": v["resolve"]["resolved"]})
                         for cid, v in got.items() if v},
           "not_computed": {cid: v["not_computed"] for cid, v in got.items()
                            if v and v.get("not_computed")},
           "missing": [cid for cid, v in got.items() if v is None]}
    (RESULTS / "m10_screen_verdicts.json").write_text(json.dumps(out, indent=1) + "\n")
    if verbose:
        for k, v in sel.items():
            print(f"  {k:10} {v}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("contrasts", nargs="*", help="contrast ids; default every registered one")
    ap.add_argument("--reissue-f-verdict", action="store_true",
                    help="rewrite results/m10_F_verdict.json from results/m10_contrast_F1.json "
                         "under the current registry hash (no number changes)")
    ap.add_argument("--B", type=int, default=None, help="bootstrap draws; default the registry's")
    ap.add_argument("--select", action="store_true",
                    help="apply the family rules to the computed contrasts; compute nothing")
    a = ap.parse_args(argv)
    reg = cfg()
    if a.select:
        selection(reg)
        return 0
    if a.reissue_f_verdict and not a.contrasts:
        f_verdict_from(json.loads((RESULTS / "m10_contrast_F1.json").read_text()), reg)
        return 0
    for cid in (a.contrasts or list(reg["contrasts"])):
        out = compute(cid, reg, B=a.B)
        if out.get("not_computed"):
            print(f"{cid:8} NOT COMPUTED: {out['not_computed']}")
        elif cid == "F1" and a.reissue_f_verdict:
            f_verdict_from(out, reg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
