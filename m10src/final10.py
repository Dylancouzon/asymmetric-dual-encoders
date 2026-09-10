"""M10's final-run decision procedure: four conjuncts under FIXED-SEQUENCE GATEKEEPING.

`m9src/final9.py:decide()` is **not reused and must not be.** It is hard-coded to M9's two
conjuncts under Holm-2 at the 0.0125 quantile. M10 has four conjuncts on two partitions under
gatekeeping at the full 0.025, so reusing it would silently apply both the wrong quantile and the
wrong multiplicity correction to an irreversible release decision (`instructions-m10.md` §Final
run change 3; Codex, 2026-09-04).

What gatekeeping means here, and why it is not Holm. The four conjuncts are ordered
**C1b → C1a → C2a → C2b**, fixed on 2026-09-04 before any M10 six-set output existed. Each is
tested at the FULL one-sided 0.025, and the sequence STOPS at the first non-rejection. A fixed
sequence spends the family alpha once, in an order chosen in advance, instead of splitting it — so
the release conjunct on the headline partition is tested at full power, and an aim conjunct is
unreachable unless its release conjunct rejected first. M9's claim-table row "C1 fail, C2 pass →
aim permitted" is deleted by construction.

The three things this module refuses to do, each because it would decide the release wrongly:

1. **Report an untested conjunct as failed.** After a non-rejection every later conjunct is
   `NOT_TESTED`. It has no verdict, because no test was run on it, and calling it a failure would
   invent evidence against ourselves as readily as calling it a pass would invent evidence for us.
2. **Renormalize a partition that is missing a dataset.** A 3-dataset clean-4 macro must be
   impossible, not merely unlikely — M9's `_assert_six` lesson, generalized.
3. **Decide on anything but `lower_q025_raw`.** Never a rounded bound, never a two-sided endpoint,
   never `np.quantile`'s default `linear` method, which interpolates toward the next order
   statistic and returns a weakly MORE PERMISSIVE bound — a defect caught in M9 by review, before
   any six-set number existed.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m7src"))
sys.path.insert(0, str(REPO / "m9src"))

import boot                                                          # noqa: E402
from final_stats import draw_plan                                    # noqa: E402  (partition-agnostic)

REGISTRY = REPO / "m10" / "final_run_registry.json"

REJECTED = "REJECTED"
NOT_REJECTED = "NOT_REJECTED"
NOT_TESTED = "NOT_TESTED"


def cfg():
    return json.loads(REGISTRY.read_text())


def registry_sha256():
    return hashlib.sha256(REGISTRY.read_bytes()).hexdigest()


def canonical(conf=None):
    """-> the registry, refusing any configuration that is not the file on disk.

    `decide(conf=...)` used to validate evidence against the CALLER'S conf. Codex set
    `alpha_per_conjunct` to 1.0 on 2026-09-10 and a p-value of 0.9 was accepted as REJECTED — the
    guard was checking the evidence against a doctored ruler. Every production entry point now
    resolves its configuration through here.

    There is no bypass. An earlier version had an `uncanonical_ok` flag "for tests"; it had zero
    callers, and a production-reachable switch whose only function is defeating this check is worse
    than the inconvenience it saved (Fable, 2026-09-10). Tests that need a mutated registry call
    `sequence(conf)` directly, which does not route through here.
    """
    if conf is None:
        return cfg()
    on_disk = cfg()
    # Compare SERIALIZED bytes, not dicts. `dict.__eq__` compares stored items, so a dict subclass
    # overriding `__getitem__` (or `__ne__`) compares equal and reads differently — and the first
    # version of this function then RETURNED the caller's object, which reopened the very attack it
    # was written to close (Fable, 2026-09-10: alpha=1.0 and p=0.9 accepted as REJECTED again).
    if (json.dumps(conf, sort_keys=True, default=str)
            != json.dumps(on_disk, sort_keys=True, default=str)):
        raise ValueError(
            "refusing a configuration that is not `m10/final_run_registry.json` on disk "
            f"(sha256 {registry_sha256()[:12]}…). The decision constants are not caller-supplied: "
            "validating evidence against a caller's ruler is not validation.")
    # Return the FILE, never the caller's object: equality is not identity, and the object that
    # compared equal is not necessarily the object that will be read.
    return on_disk


def sequence(conf=None):
    """The registered order, read from the registry and cross-checked against each conjunct's own
    `order` field — two statements of the same thing that must not be able to disagree."""
    conf = conf or cfg()
    order = list(conf["sequence"]["order"])
    by_field = sorted(conf["conjuncts"], key=lambda c: conf["conjuncts"][c]["order"])
    if order != by_field:
        raise ValueError(f"registry sequence {order} disagrees with the conjuncts' own `order` "
                         f"fields {by_field}")
    return order


def align_partition(a, b, datasets):
    """{dataset: {qid: score}} x2 -> {dataset: (qids, x, y)} on EXACTLY `datasets`.

    Strict alignment (no silent shrinkage of datasets or queries) plus an explicit assertion that
    the result is exactly the registered partition. `boot._align_ids(strict=True)` only requires
    the two inputs to agree with EACH OTHER, so a dataset missing from both would pass it and
    yield a smaller macro that still reports as a number.
    """
    # Restrict BOTH sides to the partition FIRST, then align strictly. The other order made
    # `boot._align_ids(strict=True)` compare the FULL dicts, so a comparator built per-partition
    # (as `m7src/final_run.py:clean4_block` does) against a candidate carrying all six raised on
    # `dataset sets differ` — inside `evidence_for`, i.e. AFTER the access is spent (Fable,
    # 2026-09-10). A dataset missing from either side still raises, one line below.
    want_set = set(datasets)
    a = {ds: v for ds, v in a.items() if ds in want_set}
    b = {ds: v for ds, v in b.items() if ds in want_set}
    aligned = boot._align_ids(a, b, strict=True)
    got, want = tuple(sorted(aligned)), tuple(sorted(datasets))
    if got != want:
        raise ValueError(f"partition requires exactly {want}; got {got}. A macro over a subset "
                         f"must be impossible, not merely unlikely.")
    for ds, (qids, x, y) in aligned.items():
        if not (len(qids) == len(x) == len(y)) or len(qids) == 0:
            raise ValueError(f"{ds}: ragged or empty aligned arrays")
        # HERE, not in `evidence_for`: this is the one object both halves of the pass rule consume,
        # so guarding it upstream left `bootstrap` and `signflip` reachable with NaN by any other
        # route (Fable, 2026-09-10). A NaN makes the bound NaN — and `nan > 0` is False — while
        # every permuted sign-flip statistic is NaN so the p-value returns its MINIMUM.
    assert_scoreable(aligned)
    return aligned


def assert_scoreable(aligned):
    """Every per-query score finite and in [0, 1] — the nDCG@10 range.

    Called by `align_partition` AND by both primitives. Putting it only in the aligner left
    `bootstrap` and `signflip` reachable with a NaN by any other route, and I nonetheless wrote a
    test named "the primitives themselves refuse..." that only exercised the aligner (Codex,
    2026-09-10). A claim in a test name is not a check.
    """
    for ds, (_qids, x, y) in aligned.items():
        for side, arr in (("a", x), ("b", y)):
            if not np.isfinite(arr).all():
                raise ValueError(f"{ds}: side {side} has {int((~np.isfinite(arr)).sum())} "
                                 f"non-finite per-query score(s). A NaN here silently flips a "
                                 f"rejection to a non-rejection.")
            if arr.min() < 0.0 or arr.max() > 1.0:
                raise ValueError(f"{ds}: side {side} has scores outside [0, 1] "
                                 f"(min {arr.min():.4g}, max {arr.max():.4g}); these are nDCG@10 "
                                 f"values and a mis-scaled system would move the macro arbitrarily")


def bootstrap(aligned, plan, conf):
    """Equal-weight macro of per-query differences over a partition of ANY size, full draw vector
    retained. `m9src.final_stats.bootstrap` raises unless k == 6, which is why this exists.

    **`conf` is MANDATORY and the quantile/method come from it — they are not parameters.** The
    previous signature took them as arguments and made enforcement optional, so the unsafe API
    survived beside the safe one and a caller could still label an arbitrary quantile
    `lower_q025_raw` (Codex, 2026-09-10). A value can now only carry that name if it was computed
    the registered way; the banned method's DIRECTION is demonstrated in the tests with numpy
    directly, which needs no unsafe path here.
    """
    assert_scoreable(aligned)
    b = conf["bootstrap"]
    quantile, method = b["quantile"], b["quantile_method"]
    if not plan:
        raise ValueError("empty draw plan")
    reps = {int(v.shape[0]) for v in plan.values()}
    if reps != {int(b["B"])}:
        raise ValueError(f"draw plan has replicate counts {sorted(reps)}; the registry requires "
                         f"exactly B = {b['B']}")
    k = len(aligned)
    if k == 0:
        raise ValueError("empty partition")
    diffs = {ds: (x - y) for ds, (_, x, y) in aligned.items()}
    if set(plan) != set(diffs):
        raise ValueError("draw plan does not cover exactly the aligned datasets")
    for ds, d in diffs.items():
        if plan[ds].shape[1] != d.size:
            raise ValueError(f"{ds}: plan width {plan[ds].shape[1]} != n {d.size}")
    Bs = {ds: int(v.shape[0]) for ds, v in plan.items()}
    if len(set(Bs.values())) != 1:
        raise ValueError(f"draw plan has inconsistent replicate counts: {Bs}")
    B = next(iter(Bs.values()))
    draws = np.zeros(B, dtype=np.float64)
    for ds, d in diffs.items():
        draws += d[plan[ds]].mean(axis=1)
    draws /= k
    return {
        "delta_raw": sum(float(d.mean()) for d in diffs.values()) / k,
        # THE gate. `inverted_cdf` IS the empirical quantile: at B = 10,000 and q = 0.025 it is the
        # 250th order statistic. `linear` would interpolate toward the 251st and return a weakly
        # HIGHER, more permissive bound.
        "lower_q025_raw": float(np.quantile(draws, quantile, method=method)),
        "quantile": quantile, "quantile_method": method, "B": int(B), "k_datasets": k,
        "bootstrap_seed": int(b["seed"]),
        "draws_sha256": hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest(),
        "ci95_raw_reporting_only": [float(np.quantile(draws, 0.025, method=method)),
                                    float(np.quantile(draws, 0.975, method=method))],
        "per_dataset_delta_raw": {ds: float(d.mean()) for ds, d in diffs.items()},
        "n_by_dataset": {ds: int(d.size) for ds, d in diffs.items()},
        "_gate_note": "lower_q025_raw is the ONLY field that decides; ci95_raw is reporting-only",
    }


def assert_evidence_matches_registry(cid, ev, conf):
    """Every registered constant, checked AT the boundary where it decides.

    M9 had `final_stats._assert_matches_registry`; M10 removed that protection and replaced it with
    nothing, so `decide()` trusted whatever `stat` dict it was handed. Both reviewers demonstrated
    the consequence on 2026-09-09: evidence produced with `quantile=0.0125`, `method="linear"`,
    `B=9`, an arbitrary plan seed and even `signflip_p=-1` was accepted as REJECTED, because the
    field is NAMED `lower_q025_raw` no matter how it was computed.

    A guard one function upstream is not a guard: `align_partition` refuses a short partition, and
    `bootstrap` would happily consume a plan built some other way.
    """
    c, b, sf = conf["conjuncts"][cid], conf["bootstrap"], conf["signflip"]
    part = conf["partitions"][c["partition"]]
    st = ev.get("stat") or {}
    problems = []
    # The DECISION FIELD. Presence was all that was checked, so `True` and `inf` REJECTED and
    # `nan` silently did not (Fable, 2026-09-10). `conjunct_rejects` does `> 0` on this value: the
    # bool hole was closed for the sign-flip p and left open for the number that actually decides.
    for k in (b["decision_field"], "delta_raw"):
        v = st.get(k, "__absent__")
        if v == "__absent__":
            problems.append(f"no {k!r} field")
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            problems.append(f"{k}={v!r} is not a finite number")
        elif abs(v) > 1.0:
            # every quantity here is a difference of nDCG@10 values, so |v| <= 1 by construction.
            # 1e300 was accepted and REJECTED (Fable, 2026-09-10).
            problems.append(f"{k}={v!r} is outside [-1, 1]; it is a difference of nDCG@10 values")
    # NOT a refusal. I had `lower > delta` raise, on the reasoning that no bootstrap can put its
    # lower bound above its point estimate. **That is not an invariant** — a percentile bootstrap
    # on a tiny or degenerate fixture can produce it, and Codex constructed one (2026-09-10). Over
    # 4,150 fixtures at the registered seed and B there were ZERO inversions, so it is
    # astronomically unlikely here; but a categorical refusal on the decision path would CONSUME
    # THE IRREVERSIBLE ACCESS on a legitimate run, which is a worse failure than the forged pair it
    # was guarding against — and `evidence_for` being the only production path already prevents
    # that pair. So it is REPORTED and never raised.
    if ev.get("conjunct") != cid:
        problems.append(f"evidence is labelled conjunct {ev.get('conjunct')!r}, not {cid!r}")
    for k, want in (("quantile", b["quantile"]), ("quantile_method", b["quantile_method"]),
                    ("B", b["B"])):
        if st.get(k) != want:
            problems.append(f"{k}={st.get(k)!r}, registered {want!r}")
    if st.get("k_datasets") != len(part):
        problems.append(f"k_datasets={st.get('k_datasets')!r}, partition "
                        f"{c['partition']} has {len(part)}")
    got = tuple(sorted(st.get("n_by_dataset") or {}))
    if got != tuple(sorted(part)):
        problems.append(f"scored datasets {got} != partition {tuple(sorted(part))}")
    p = ev.get("signflip_p")
    # `bool` is a subclass of `int`, so `isinstance(True, int)` is True and `False` behaved as
    # p = 0 and REJECTED. Codex reproduced that on 2026-09-10.
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not (0.0 <= float(p) <= 1.0):
        problems.append(f"signflip_p={p!r} is not a probability")
    # ABSENCE IS NOT EQUALITY. `ev.get(k, registered)` accepted a missing key as a match, so
    # evidence carrying no sign-flip provenance at all passed.
    for k, want in (("signflip_B", sf["B"]), ("signflip_seed", sf["seed"]),
                    ("signflip_alternative", sf["alternative"]),
                    ("signflip_unit_of", "boot.unit_key")):
        if k not in ev:
            problems.append(f"no {k!r} in the evidence")
        elif ev[k] != want:
            problems.append(f"{k}={ev[k]!r}, registered {want!r}")
    if st.get("bootstrap_seed") != b["seed"]:
        problems.append(f"bootstrap_seed={st.get('bootstrap_seed')!r}, registered {b['seed']!r}")
    # ORIENTATION. A reversed contrast with conforming metadata was otherwise indistinguishable.
    if ev.get("a") != c["a"] or ev.get("b") != c["b"]:
        problems.append(f"oriented {ev.get('a')!r} minus {ev.get('b')!r}, registered "
                        f"{c['a']!r} minus {c['b']!r}")
    if ev.get("partition") != c["partition"]:
        problems.append(f"partition={ev.get('partition')!r}, registered {c['partition']!r}")
    if ev.get("comparator_source_sha256") != conf["comparator_source"]["sha256"]:
        problems.append("comparator_source_sha256 does not match the registry")
    for k in ("draw_plan_sha256", "qid_sha256", "registry_sha256"):
        v = ev.get(k)
        if not isinstance(v, str) or len(v) != 64:
            problems.append(f"{k}={v!r} is not a sha256 digest")
    counts = st.get("n_by_dataset") or {}
    if not counts:
        problems.append("no 'n_by_dataset' in the stat")
    if any(type(v) is not int or v <= 0 for v in counts.values()):
        problems.append(f"n_by_dataset has non-positive or non-integer counts: {counts}")
    # The two halves of the pass rule must have scored the same queries. REQUIRED, not
    # `is not None` — dropping the field silently skipped the check, which is the exact
    # "ABSENCE IS NOT EQUALITY" pattern named twenty lines above and re-created here in the same
    # batch (Fable mutation S1, 2026-09-10).
    if "signflip_per_dataset_n" not in ev:
        problems.append("no 'signflip_per_dataset_n' in the evidence")
    elif ev["signflip_per_dataset_n"] != counts:
        problems.append(f"the bootstrap scored {counts} and the sign-flip scored "
                        f"{ev['signflip_per_dataset_n']}")
    if problems:
        raise ValueError(f"{cid}: evidence does not match the registered procedure — "
                         + "; ".join(problems))
    lo, de = st[b["decision_field"]], st["delta_raw"]
    return {"lower_exceeds_point_estimate": True} if lo > de else {}


def conjunct_rejects(stat, sf_p, alpha):
    """The registered pass rule: BOTH the bootstrap bound and the sign-flip test, not either.

    Both operands come from `evidence_for`, which produces plain floats. Three rounds were spent
    escalating against hand-crafted `float` subclasses overriding `__gt__`, then `__float__` — an
    arms race with no adversary, in a repo where the only caller of `decide()` is our own executor.
    The class is dropped deliberately: `evidence_for` being the sole production path is the control,
    not type forensics at the comparison site (Dylan's over-engineering note, 2026-09-10).
    """
    return bool(stat["lower_q025_raw"] > 0.0 and sf_p <= alpha)


def decide(evidence, conf=None):
    """{conjunct_id: {"stat": <bootstrap dict>, "signflip_p": float}} -> the gatekeeping verdict.

    `evidence` need not contain every conjunct: the sequence stops at the first non-rejection, so
    the executor is entitled to stop SCORING there too. A conjunct that is absent because the
    sequence never reached it is `NOT_TESTED`; a conjunct that is absent although the sequence DID
    reach it is an error, not a silent skip.
    """
    conf = canonical(conf)
    order = sequence(conf)
    alpha = conf["sequence"]["alpha_per_conjunct"]
    out, stopped = {}, False
    for cid in order:
        if stopped:
            e = {"status": NOT_TESTED,
                 "partition": conf["conjuncts"][cid]["partition"],
                 "bar_comparator": conf["conjuncts"][cid]["b"],
                 "gate": conf["conjuncts"][cid]["gate"],
                 "_why": "the sequence stopped at an earlier non-rejection; this conjunct carries "
                         "NO verdict, and is never reported as failed"}
            # The executor scores all four in one transaction — the access is spent either way, so
            # the untested conjuncts' descriptive numbers are free. Reporting them (clearly
            # labelled) is better than a blank row a reader fills in with a failure that was never
            # measured. This is what `implementation.executor_contract` requires, and the code
            # agreeing with it is the fix for a contract that was otherwise fictional (Codex).
            ev = evidence.get(cid)
            if ev is not None:
                e.update(assert_evidence_matches_registry(cid, ev, conf))
                e["descriptive_delta_raw"] = ev["stat"]["delta_raw"]
                e["descriptive_lower_q025_raw"] = ev["stat"]["lower_q025_raw"]
                e["_descriptive_note"] = ("computed but NOT TESTED — no inferential claim, and it "
                                          "may not be quoted as a bound")
            out[cid] = e
            continue
        ev = evidence.get(cid)
        if ev is None:
            raise ValueError(f"{cid} is next in the sequence and still untested, but no evidence "
                             f"was supplied for it; the sequence had not stopped")
        flags = assert_evidence_matches_registry(cid, ev, conf)
        rej = conjunct_rejects(ev["stat"], ev["signflip_p"], alpha)
        out[cid] = {"status": REJECTED if rej else NOT_REJECTED,
                    "lower_q025_raw": ev["stat"]["lower_q025_raw"],
                    "delta_raw": ev["stat"]["delta_raw"],
                    "signflip_p": ev["signflip_p"],
                    "alpha": alpha,
                    "partition": conf["conjuncts"][cid]["partition"],
                    "bar_comparator": conf["conjuncts"][cid]["b"],
                    "gate": conf["conjuncts"][cid]["gate"],
                    **flags}          # e.g. lower_exceeds_point_estimate — REPORTED, never raised
        if not rej:
            stopped = True
    # The two conjuncts of a partition must have been computed from ONE alignment and ONE plan.
    # Checking their digests verifies the shared plan AND qid identity from the evidence alone,
    # and cannot raise on a run that reached `decide()` through `evidence_for` — where both are
    # derived from the same registry partition (Fable/Codex, 2026-09-10).
    by_part = {}
    for cid, e in out.items():
        ev = evidence.get(cid)
        if ev is not None:
            by_part.setdefault(conf["conjuncts"][cid]["partition"], []).append((cid, ev))
    for part, members in by_part.items():
        digests = {(ev.get("draw_plan_sha256"), ev.get("qid_sha256")) for _c, ev in members}
        if len(digests) > 1:
            raise ValueError(
                f"the {part} conjuncts {[c for c, _ in members]} were computed from different "
                f"alignments or plans (draw_plan/qid digests differ). A partition's conjuncts share "
                f"one plan and one qid set, or they are not comparable.")
    any_rejected = any(v["status"] == REJECTED for v in out.values())
    return {
        "order": order,
        "alpha_per_conjunct": alpha,
        "procedure": "fixed-sequence gatekeeping; each conjunct at the full one-sided alpha, "
                     "stopping at the first non-rejection",
        "conjuncts": out,
        "rejected": [c for c in order if out[c]["status"] == REJECTED],
        "not_tested": [c for c in order if out[c]["status"] == NOT_TESTED],
        "reserved_batch_runs": any_rejected,
        "_reserved_trigger": conf["reserved"]["trigger"],
        "_vocabulary": "a final-run conjunct REJECTS or does not. It never 'resolves' and is never "
                       "'confirmed' — those words belong to the screen and are on "
                       "`forbidden_words`. Keeping the vocabularies apart is what stops a screen "
                       "verdict being read as a release claim.",
    }


def signflip(aligned, conf):
    """The sign-flip conjunct, computed from the SAME validated `aligned` object the bootstrap uses.

    Two holes closed here (Codex, 2026-09-10). The partition is no longer a caller argument — a
    caller could pass a valid three-dataset list and succeed — because `aligned` arrives already
    validated by `align_partition` against the conjunct's REGISTERED partition. And it returns the
    primitive's full provenance instead of a bare float, so `R`, the seed, the alternative and the
    strata are on the record rather than discarded.
    """
    assert_scoreable(aligned)
    sf = conf["signflip"]
    sub_a = {ds: dict(zip(v[0], v[1])) for ds, v in aligned.items()}
    sub_b = {ds: dict(zip(v[0], v[2])) for ds, v in aligned.items()}
    res = boot.signflip_dep(sub_a, sub_b, R=sf["B"], seed=sf["seed"],
                            alternative=sf["alternative"], strict=True,
                            unit_of=boot.unit_key)           # explicit, as m9src/final_stats does
    return {"signflip_p": float(res["p"]), "signflip_B": int(res["R"]),
            "signflip_seed": int(res["seed"]), "signflip_alternative": res["alternative"],
            "signflip_unit_of": "boot.unit_key", "signflip_strata": res["strata"],
            "signflip_shared_units": int(res["shared_units"]),
            "signflip_per_dataset_n": {k: int(v) for k, v in res["per_dataset_n"].items()}}


def evidence_for(cid, scores, conf=None):
    """**The one production path.** `{system: {dataset: {qid: score}}}` -> the evidence for `cid`.

    Codex's blocking item 1, 2026-09-10: validating caller-supplied evidence is structurally weaker
    than not letting the caller supply it. Everything that could be forged is now DERIVED from the
    registry given `cid` — which systems, in which orientation, over which partition, at which
    quantile, method, B and seeds. The caller supplies raw per-query scores and nothing else.

    It emits complete provenance: the plan digest, the qid digest, the orientation, the partition,
    and both statistics' full configuration.
    """
    conf = canonical(conf)
    c = conf["conjuncts"][cid]
    datasets = conf["partitions"][c["partition"]]
    for sysname in (c["a"], c["b"]):
        if sysname not in scores:
            raise ValueError(f"{cid}: no scores supplied for {sysname!r}")
    aligned = align_partition(scores[c["a"]], scores[c["b"]], datasets)
    plan, plan_digest = draw_plan(aligned, B=conf["bootstrap"]["B"], seed=conf["bootstrap"]["seed"])
    stat = bootstrap(aligned, plan, conf)
    qid_digest = hashlib.sha256()
    for ds in sorted(aligned):
        qid_digest.update(ds.encode())
        qid_digest.update(repr(list(aligned[ds][0])).encode())
    ev = {"stat": stat, "conjunct": cid, "partition": c["partition"],
          "orientation": f"{c['a']} minus {c['b']}", "a": c["a"], "b": c["b"],
          "draw_plan_sha256": plan_digest, "qid_sha256": qid_digest.hexdigest(),
          "comparator_source_sha256": conf["comparator_source"]["sha256"],
          "registry_sha256": registry_sha256(),
          "_comparator_hash_is_a_stamp": "this field is COPIED from the registry by `evidence_for`, "
                                         "so the guard's check of it is a tautology, not a "
                                         "measurement of the comparator file (Fable, 2026-09-10). "
                                         "The real check is `test_final_run_registry`'s hash of "
                                         "`results/perquery.json`."}
    ev.update(signflip(aligned, conf))
    return ev


def headline(verdict, conf=None):
    """-> the registered sentences the outcome permits, in sequence order, plus what is forbidden.

    A production function, because "the report writer will pick the right sentence" is not a
    control. A sentence is emitted ONLY for a conjunct whose status is REJECTED — never for one
    that was NOT_TESTED, which is the failure mode that would claim what was never measured.
    """
    conf = canonical(conf)
    hv = conf["headline_verbatim"]
    out = []
    for cid in sequence(conf):
        if verdict["conjuncts"][cid]["status"] != REJECTED:
            continue
        key = next((k for k in hv if k.startswith(cid + "_")), None)
        if key is None:
            raise ValueError(f"{cid} REJECTED but no registered sentence exists for it")
        out.append({"conjunct": cid, "sentence": hv[key]})
    return {"emit": out,
            "emit_nothing_because": None if out else "no conjunct rejected",
            "must_not": hv["_must_not"],
            "must_accompany": hv["_stella_disclosure"],
            "_rule": "one sentence per REJECTED conjunct, in sequence order. A conjunct that was "
                     "NOT_TESTED emits nothing — it has no verdict."}


__all__ = ["cfg", "canonical", "registry_sha256", "sequence", "align_partition", "assert_scoreable",
           "bootstrap",
           "signflip", "evidence_for", "conjunct_rejects", "assert_evidence_matches_registry",
           "decide", "headline", "draw_plan", "REJECTED", "NOT_REJECTED", "NOT_TESTED"]
