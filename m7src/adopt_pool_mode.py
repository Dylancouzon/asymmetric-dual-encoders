"""Write an adopted count-saturation pooling mode into a run's frozen preprocessing rule.

Capacity lever #4 changes the query RULE, not the weights: the same rows serve any mode, so
adoption is a metadata edit plus a regenerated release artifact, and the shipped bytes of the
table are unchanged. This script is the only sanctioned way to make that edit, and it refuses
unless the committed lever-4 result actually adopted that mode for that run id -- the alternative
is a session hand-editing a JSON and no one being able to tell afterwards which rule produced
which number.

The mode goes into the TRAINING checkpoint's metadata, not only the release sibling, because
`ensure_release` rebuilds the release meta from the training meta; writing it in one place only
would let a regeneration silently drop the adopted rule.

Usage: adopt_pool_mode.py <run_id> <mode>
"""
import json
import sys
from dataclasses import asdict

from _paths import REPO, WORK
from table import POOL_MODES, Preproc, ensure_release, read_meta


def main(run_id, mode):
    if mode not in POOL_MODES:
        raise SystemExit(f"unknown pool mode {mode!r}; known {POOL_MODES}")
    ev = REPO / "results" / "m7_lever4_pooling_full.json"
    if not ev.exists():
        raise SystemExit("results/m7_lever4_pooling_full.json missing; nothing has been adopted")
    b = json.loads(ev.read_text())
    if b.get("adopted") != mode or b.get("adjudicated_on") != run_id:
        raise SystemExit(f"lever 4 adopted {b.get('adopted')!r} on {b.get('adjudicated_on')!r}, "
                         f"not {mode!r} on {run_id!r}. The pre-registered bar decides this, "
                         "not the command line.")
    npz = WORK / "runs" / f"{run_id}.npz"
    mp = npz.parent / f"{run_id}.meta.json"
    meta = json.loads(mp.read_text())
    old = Preproc(**meta["preproc"])
    new = Preproc(**{**meta["preproc"], "pool_mode": mode})
    meta["preproc"] = asdict(new)
    meta["preproc_fingerprint"] = new.fingerprint()
    meta["pool_mode_adoption"] = {
        "mode": mode, "previous": old.pool_mode,
        "previous_fingerprint": old.fingerprint(),
        "evidence": "results/m7_lever4_pooling_full.json",
        "protocol": "m7/LEDGER.md 'Capacity lever #4', pre-registered before any number",
        "note": "rows and int8 codes are unchanged; this is a query-rule change with no bytes and "
                "no query-time cost attached"}
    mp.write_text(json.dumps(meta, indent=1, sort_keys=True))
    rel = npz.with_name(npz.stem + ".release.npz")
    for p in (rel, rel.parent / (rel.stem + ".meta.json")):
        if p.exists():
            p.unlink()                     # force a rebuild so the release meta carries the mode
    rel = ensure_release(npz)
    rm = read_meta(rel)
    assert rm["preproc"]["pool_mode"] == mode, rm["preproc"]
    print(f"{run_id}: pool_mode {old.pool_mode} -> {mode}; preproc fingerprint "
          f"{old.fingerprint()} -> {new.fingerprint()}; release rebuilt at {rel.name}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
