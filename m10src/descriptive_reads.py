"""The two DESCRIPTIVE reads ruled by gpt-6-astra on 2026-09-09 (`descriptive_runs`).

    ANCHOR-seed1 vs ANCHOR      decision 2: anchor sensitivity to the training seed
    A3-20M       vs F-bge-small decision 1: A4's corpus against A3's, at 20M

**Neither is a contrast and this module computes no interval.** That is deliberate, not an
omission: astra's ruling says *"compare their final checkpoints descriptively on authorized
development surfaces"*, and a bootstrap interval printed beside a point estimate is read as a test
no matter what the surrounding prose says. The twelve registered contrasts are computed, the
verdicts are written, and these two reads are inputs to neither. What is reported is exactly what
the ruling asked for: **both absolute scores, the signed difference, and its family
decomposition.**

Each read carries its registered label and its prohibitions out of the registry rather than from a
docstring here, so the thing a future session reads next to the number is the thing that was
registered before the number existed (`arms.ANCHOR-seed1._what_it_licenses` /
`._what_it_does_NOT_license`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cov_macro
from contrasts import cfg, check_against_arm_record, point_estimate, read_point
from run_arm import REGISTRY, RESULTS, sha256_file

# (arm, comparator, at_examples on BOTH sides, registry key holding the labels)
READS = {
    "seed_sensitivity": ("ANCHOR-seed1", "ANCHOR", None),
    "corpus_at_20M": ("A3-20M", "F-bge-small", 20_000_000),
}


def read(name, reg=None, verbose=True):
    """-> the descriptive record for one read, written to `results/m10_descriptive_<name>.json`."""
    reg = reg or cfg()
    arm, comparator, at = READS[name]
    entry = reg["arms"][arm]
    if not entry.get("descriptive"):
        raise ValueError(f"{arm} is not registered `descriptive: true`; this module reads only "
                         f"`descriptive_runs` members, never a screen arm")
    uf = dict(cov_macro.SURFACE)
    la, sa = read_point(arm, at)
    lb, sb = read_point(comparator, at)
    for nm, lab in ((arm, la), (comparator, lb)):
        check_against_arm_record(nm, lab)          # the same identity check the contrasts use
    delta, aligned = point_estimate(sa, sb, uf)
    ma, fa, ua = cov_macro.macro(sa, uf)
    mb, fb, ub = cov_macro.macro(sb, uf)
    w = cov_macro.weights(uf)
    per_unit = {u: float((x - y).mean()) for u, (_q, x, y) in aligned.items()}
    by_family = {f: sum(w[u] * per_unit[u] for u in per_unit if cov_macro.SURFACE[u] == f)
                 for f in sorted(cov_macro.FAMILIES)}

    out = {
        "read": name,
        "_what": entry["_what"],
        "arms": {"a": arm, "b": comparator, "at_examples": at,
                 "read_labels": {arm: la, comparator: lb}},
        "absolute": {arm: ma, comparator: mb},
        "signed_difference": delta,
        "_orientation": f"{arm} minus {comparator}",
        "by_family_absolute": {arm: fa, comparator: fb},
        "by_family_contribution_to_the_difference": by_family,
        "by_unit_difference": per_unit,
        "by_unit_absolute": {arm: ua, comparator: ub},
        "weights": w,
        "no_interval": "DELIBERATE. This read selects nothing and is not a contrast, so no "
                       "bootstrap interval is computed: an interval printed beside a point "
                       "estimate is read as a test whatever the prose says. The twelve registered "
                       "contrasts are in `results/m10_contrast_*.json`.",
        "selects": entry["selects"],
        "registry_sha256": sha256_file(REGISTRY),
    }
    for k in ("_what_it_licenses", "_what_it_does_NOT_license", "_parity"):
        if entry.get(k):
            out[k] = entry[k]
    if name == "seed_sensitivity":
        out["registered_sentence_filled"] = (
            f"At 5M examples, the second training seed changed ANCHOR's COV macro by "
            f"delta = {delta:+.6f}; this single between-run difference describes ANCHOR "
            f"SENSITIVITY, while the screen's query-resampling intervals condition on the trained "
            f"checkpoints and exclude training-seed variability.")
    p = RESULTS / f"m10_descriptive_{name}.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    if verbose:
        print(f"{name}: {arm} {ma:.6f} vs {comparator} {mb:.6f}  delta {delta:+.6f}")
        for f, v in by_family.items():
            print(f"    {f:16} contributes {v:+.6f}  "
                  f"({fa[f]:.4f} vs {fb[f]:.4f})")
        if name == "seed_sensitivity":
            print("  " + out["registered_sentence_filled"])
    return out


def main(argv=None):
    reg = cfg()
    names = argv or [n for n in READS]
    done = []
    for n in names:
        arm = READS[n][0]
        rec = RESULTS / f"m10_arm_{arm}.json"
        if not rec.exists() or json.loads(rec.read_text()).get("status") != "complete":
            print(f"{n}: {arm} has no COMPLETE arm record yet — skipped")
            continue
        done.append(read(n, reg))
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
