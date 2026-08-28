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

Usage: adopt_pool_mode.py <run_id> <mode> [--refresh-evidence]
"""
import json
import sys
from dataclasses import asdict

from _paths import REPO, WORK
from table import POOL_MODES, Preproc, ensure_release, read_meta


def evidence_path(run_id):
    """Named for the artifact it adjudicates. This USED to be a fixed `..._full.json`, and on
    2026-08-28 that bit: lever 4 was re-adjudicated on a new candidate, the fixed name came to hold
    that artifact's FAILED adjudication, and the shipping artifact's own metadata still cited it as
    the evidence for its adopted rule. A reader following the pointer would have found evidence
    contradicting the adoption. One file per run id cannot drift that way."""
    return REPO / "results" / f"m7_lever4_pooling_{run_id}.json"


def main(run_id, mode, refresh_evidence=False):
    if mode not in POOL_MODES:
        raise SystemExit(f"unknown pool mode {mode!r}; known {POOL_MODES}")
    ev = evidence_path(run_id)
    if not ev.exists():
        raise SystemExit(f"{ev.relative_to(REPO)} missing: lever 4 has not been adjudicated on "
                         f"{run_id!r}. Run lever4_readjudicate.py on it first.")
    b = json.loads(ev.read_text())
    if b.get("adopted") != mode or b.get("adjudicated_on") != run_id:
        raise SystemExit(f"lever 4 adopted {b.get('adopted')!r} on {b.get('adjudicated_on')!r}, "
                         f"not {mode!r} on {run_id!r}. The pre-registered bar decides this, "
                         "not the command line.")
    npz = WORK / "runs" / f"{run_id}.npz"
    mp = npz.parent / f"{run_id}.meta.json"
    meta = json.loads(mp.read_text())
    old = Preproc(**meta["preproc"])
    if refresh_evidence:
        # Repoint a stale evidence path WITHOUT touching the rule. Only legal when the artifact
        # already serves the adopted mode, so this can repair a renamed pointer and nothing else.
        if old.pool_mode != mode:
            raise SystemExit(f"--refresh-evidence refuses: {run_id} serves {old.pool_mode!r}, not "
                             f"{mode!r}. It repairs a pointer, it does not adopt a rule.")
        adopt = dict(meta.get("pool_mode_adoption") or {})
        adopt["evidence"] = str(ev.relative_to(REPO))
        adopt["evidence_repointed"] = ("was results/m7_lever4_pooling_full.json, a fixed name that "
                                       "re-pointed when lever 4 was re-adjudicated on another "
                                       "artifact; the rule and the bytes are unchanged")
        meta["pool_mode_adoption"] = adopt
        mp.write_text(json.dumps(meta, indent=1, sort_keys=True))
        print(f"{run_id}: evidence -> {adopt['evidence']} (rule unchanged, still "
              f"{old.pool_mode!r})")
        return
    new = Preproc(**{**meta["preproc"], "pool_mode": mode})
    meta["preproc"] = asdict(new)
    meta["preproc_fingerprint"] = new.fingerprint()
    meta["pool_mode_adoption"] = {
        "mode": mode, "previous": old.pool_mode,
        "previous_fingerprint": old.fingerprint(),
        "evidence": str(ev.relative_to(REPO)),
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
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], refresh_evidence="--refresh-evidence" in sys.argv)
