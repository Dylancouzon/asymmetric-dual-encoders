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
    aligned = boot._align_ids(a, b, strict=True)
    aligned = {ds: v for ds, v in aligned.items() if ds in set(datasets)}
    got, want = tuple(sorted(aligned)), tuple(sorted(datasets))
    if got != want:
        raise ValueError(f"partition requires exactly {want}; got {got}. A macro over a subset "
                         f"must be impossible, not merely unlikely.")
    for ds, (qids, x, y) in aligned.items():
        if not (len(qids) == len(x) == len(y)) or len(qids) == 0:
            raise ValueError(f"{ds}: ragged or empty aligned arrays")
    return aligned


def bootstrap(aligned, plan, quantile, method="inverted_cdf", conf=None):
    """Equal-weight macro of per-query differences over a partition of ANY size, full draw vector
    retained. `m9src.final_stats.bootstrap` raises unless k == 6, which is why this exists.

    `conf=None` is permitted ONLY so a test can demonstrate the banned method's direction. Pass the
    registry in every real call and a non-registered quantile or method is refused here rather than
    being labelled `lower_q025_raw` regardless of how it was computed.
    """
    if conf is not None:
        b = conf["bootstrap"]
        if quantile != b["quantile"] or method != b["quantile_method"]:
            raise ValueError(f"refusing to compute the decision field at quantile={quantile!r} "
                             f"method={method!r}; registered {b['quantile']!r}/"
                             f"{b['quantile_method']!r}. `linear` interpolates toward the next "
                             f"order statistic and is weakly MORE PERMISSIVE.")
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
    if b["decision_field"] not in st:
        problems.append(f"no {b['decision_field']!r} field")
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
    if not isinstance(p, (int, float)) or not (0.0 <= float(p) <= 1.0):
        problems.append(f"signflip_p={p!r} is not a probability")
    if ev.get("signflip_B", sf["B"]) != sf["B"] or ev.get("signflip_seed", sf["seed"]) != sf["seed"]:
        problems.append("sign-flip B/seed do not match the registry")
    if problems:
        raise ValueError(f"{cid}: evidence does not match the registered procedure — "
                         + "; ".join(problems))


def conjunct_rejects(stat, sf_p, alpha):
    """The registered pass rule: BOTH the bootstrap bound and the sign-flip test, not either."""
    return bool(stat["lower_q025_raw"] > 0 and sf_p <= alpha)


def decide(evidence, conf=None):
    """{conjunct_id: {"stat": <bootstrap dict>, "signflip_p": float}} -> the gatekeeping verdict.

    `evidence` need not contain every conjunct: the sequence stops at the first non-rejection, so
    the executor is entitled to stop SCORING there too. A conjunct that is absent because the
    sequence never reached it is `NOT_TESTED`; a conjunct that is absent although the sequence DID
    reach it is an error, not a silent skip.
    """
    conf = conf or cfg()
    order = sequence(conf)
    alpha = conf["sequence"]["alpha_per_conjunct"]
    out, stopped = {}, False
    for cid in order:
        if stopped:
            out[cid] = {"status": NOT_TESTED,
                        "partition": conf["conjuncts"][cid]["partition"],
                        "bar_comparator": conf["conjuncts"][cid]["b"],
                        "gate": conf["conjuncts"][cid]["gate"],
                        "_why": "the sequence stopped at an earlier non-rejection; this conjunct "
                                "carries NO verdict, and is never reported as failed"}
            continue
        ev = evidence.get(cid)
        if ev is None:
            raise ValueError(f"{cid} is next in the sequence and still untested, but no evidence "
                             f"was supplied for it; the sequence had not stopped")
        assert_evidence_matches_registry(cid, ev, conf)
        rej = conjunct_rejects(ev["stat"], ev["signflip_p"], alpha)
        out[cid] = {"status": REJECTED if rej else NOT_REJECTED,
                    "lower_q025_raw": ev["stat"]["lower_q025_raw"],
                    "delta_raw": ev["stat"]["delta_raw"],
                    "signflip_p": ev["signflip_p"],
                    "alpha": alpha,
                    "partition": conf["conjuncts"][cid]["partition"],
                    "bar_comparator": conf["conjuncts"][cid]["b"],
                    "gate": conf["conjuncts"][cid]["gate"]}
        if not rej:
            stopped = True
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


def signflip_p(a, b, datasets, conf=None):
    """The sign-flip conjunct on one partition, at the registered B/seed/alternative.

    It runs `align_partition` FIRST and scores only what that validated — the reviewers found this
    function filtering to a subset and calling the primitive directly, so a dataset missing from
    both inputs silently produced a p-value over three datasets while `align_partition` (the other
    half of the same pass rule) would have refused. One invariant enforced in one place is not
    enforced; both halves of a conjunctive rule must refuse.
    """
    conf = conf or cfg()
    sf = conf["signflip"]
    aligned = align_partition(a, b, datasets)                # raises on anything short or empty
    sub_a = {ds: dict(zip(v[0], v[1])) for ds, v in aligned.items()}
    sub_b = {ds: dict(zip(v[0], v[2])) for ds, v in aligned.items()}
    res = boot.signflip_dep(sub_a, sub_b, R=sf["B"], seed=sf["seed"],
                            alternative=sf["alternative"], strict=True,
                            unit_of=boot.unit_key)           # explicit, as m9src/final_stats does
    return float(res["p"])


__all__ = ["cfg", "sequence", "align_partition", "bootstrap", "conjunct_rejects", "decide",
           "signflip_p", "draw_plan", "REJECTED", "NOT_REJECTED", "NOT_TESTED"]
