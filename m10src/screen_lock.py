"""The M10 screen lock: `m10/screen_registry.json` is the authority, and this validates it.

M9's lesson, copied deliberately (`m9src/final_stats.py`): *"Constants come from the registry.
Prose is not authoritative; the registry is."* A rule that reads a number reads it from here, so
a mandate sentence and a running arm cannot disagree silently.

What this refuses, and why each one has bitten something before:
- a contrast naming an arm that does not exist, or an arm no contrast reads;
- a contrast count that is not thirteen (the Bonferroni denominator is 13; a fourteenth contrast
  silently changes every bound -- a Codex pass corrected exactly that count on 2026-09-04);
- statistics that disagree with `m10src/cov_macro`'s implementation or with the measured
  resolution number's artifact;
- a family with no outcome->action entry (an unresolved contrast with no registered default is a
  decision taken after seeing the number).
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m10src"))
REGISTRY = REPO / "m10" / "screen_registry.json"
RESOLUTION = REPO / "results" / "m10_cov_resolution.json"


def cfg():
    return json.loads(REGISTRY.read_text())


def validate(r=None):
    """-> list of problems; empty means the lock is coherent.

    Every check below exists because a Fable or Codex pass showed the registry passing without
    it (S10, 2026-09-05): a dozen mutations a reviewer would reject validated clean.
    """
    import cov_macro
    r = r or cfg()
    bad = []
    arms, contrasts, alias = r["arms"], r["contrasts"], r["anchor_aliases"]
    NON_FAMILY = {"anchor"}

    # -- arms and the trained count. A boolean, not a note-prefix heuristic: an arm could be
    # moved into the "not trained" bucket by editing prose.
    for a, v in arms.items():
        if not isinstance(v.get("trained"), bool):
            bad.append(f"arm {a} has no boolean `trained` field")
    trained = [a for a, v in arms.items() if v.get("trained") is True]
    if len(trained) != r["trained_arms_expected"]:
        bad.append(f"{len(trained)} trained arms, registry expects {r['trained_arms_expected']}")
    if len(contrasts) != r["statistics"]["bootstrap"]["n_contrasts"]:
        bad.append(f"{len(contrasts)} contrasts vs a Bonferroni denominator of "
                   f"{r['statistics']['bootstrap']['n_contrasts']}")

    # -- aliases must land on a real arm, or they are a name for nothing
    for k, v in alias.items():
        if k.startswith("_") or not isinstance(v, str):
            continue
        if v not in arms and not v.startswith("the winner"):
            bad.append(f"anchor alias {k} -> '{v}' is not an arm")

    def resolve(name):
        return alias.get(name, name)
    for cid, c in contrasts.items():
        for side in ("a", "b"):
            if c[side] not in arms and c[side] not in alias:
                bad.append(f"contrast {cid}: '{c[side]}' is neither an arm nor an anchor alias")
        rule = c.get("rule")
        if rule and rule not in r.get("rules", {}):
            bad.append(f"contrast {cid} names rule '{rule}', which is not in `rules`")
        at = c.get("at_examples")
        if at:
            for side in ("a", "b"):
                arm = arms.get(resolve(c[side]))
                if arm and at > max([arm.get("dose_examples", 0)]
                                    + list(arm.get("read_at", []))
                                    + [arm.get("conditional_extension", {}).get("to_examples", 0)]):
                    bad.append(f"contrast {cid} reads {c[side]} at {at:,} examples, beyond its dose")
    everything = list(contrasts.values()) + list(r["descriptive_contrasts"].values())
    read = {resolve(c[s]) for c in everything for s in ("a", "b") if c.get(s)}
    for a, v in arms.items():
        if a in read or a in alias.values() or v.get("family") == "F" or not v.get("trained"):
            continue
        bad.append(f"arm {a} is trained but no contrast reads it")

    # -- the anchor's own settings must agree with every axis it stands in for
    anc = r["anchor"]
    for k, field, want in (("E-bs32", "batch", 32), ("G-1152", "feature_dim", 1152),
                           ("B-75/25", "mix", "75/25")):
        if k in alias and anc.get(field) != want:
            bad.append(f"alias {k} claims the anchor is {want}, but anchor.{field} = "
                       f"{anc.get(field)!r}")

    fams = {v["family"] for v in arms.values() if "family" in v}
    for f in fams:
        v = r["outcome_to_action"].get(f)
        if not v or not str(v).strip():
            bad.append(f"family {f} has no outcome->action entry")
    if set(r["order"]) != fams - NON_FAMILY:
        bad.append(f"order {r['order']} does not cover the decision families "
                   f"{sorted(fams - NON_FAMILY)}")

    st = r["statistics"]
    if tuple(sorted(st["families"])) != tuple(sorted(cov_macro.FAMILIES)):
        bad.append(f"registry families {st['families']} != cov_macro.FAMILIES "
                   f"{list(cov_macro.FAMILIES)}")
    b = st["bootstrap"]
    if b.get("alpha_family") != 0.025:
        bad.append(f"alpha_family is {b.get('alpha_family')}, not the registered 0.025")
    if abs(b["quantile"] - b["alpha_family"] / b["n_contrasts"]) > 1e-15:
        bad.append("quantile is not alpha_family / n_contrasts")
    if b["quantile_method"] != "inverted_cdf":
        bad.append("quantile_method is not the registered inverted_cdf")

    # -- the resolution artifact is REQUIRED: absent, three checks would silently not run
    if not RESOLUTION.exists():
        bad.append(f"{RESOLUTION.name} is missing; the power disclosure cannot be checked")
    else:
        got = json.loads(RESOLUTION.read_text())
        if abs(got["resolution_distance"] - st["measured_resolution_distance"]) > 5e-7:
            bad.append(f"registry resolution {st['measured_resolution_distance']} != artifact "
                       f"{got['resolution_distance']}")
        if got["MDE_registered"] != st["MDE"]:
            bad.append("MDE disagrees with the resolution artifact")
        if not got["mde_below_resolution"]:
            bad.append("artifact says the MDE is at or above the resolution distance")
        for k in ("B", "seed", "chunk", "quantile_method"):
            if got["bootstrap"][k] != b.get(k):
                bad.append(f"bootstrap.{k}: registry {b.get(k)!r} != artifact "
                           f"{got['bootstrap'][k]!r}")
    return bad


if __name__ == "__main__":
    problems = validate()
    for p in problems:
        print("PROBLEM:", p)
    print("LOCK OK" if not problems else f"{len(problems)} PROBLEMS")
    sys.exit(1 if problems else 0)
