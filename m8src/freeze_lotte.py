"""LEDGER `E10-REMEDY` (`m8/LEDGER.md`, 2026-08-29 entry) -- the PIN. `m8src/protected_filter.py`
now holds the whole per-item removal + re-screen (its REMEDY section); this module's one remaining
job is to freeze the result: hash the surviving slices' remediated files and pin those hashes as
the protected partition's fingerprint, then let a later caller re-check the files on disk against
that pin.

WHY PIN IS SEPARATE FROM REMEDY, given both could in principle live in one file. `paths_guard`
permits exactly one `claim()` per process, so a module's allowlist entry is a hard ceiling on what
it may touch -- `m8src.protected_filter` claims `{"untouched_labels", "lotte", "m9reserve"}`
because REMEDY needs to read the protected query index (`untouched_labels`) as well as the LoTTE
corpora (`lotte`); this module's own entry is `{"lotte"}` only, because hashing already-remediated
files needs nothing else. Keeping PIN in its own file with its own narrower claim is not
bureaucracy, it is the whole point of the allowlist: the module that only ever needs to read
already-cleaned files cannot be tricked, refactored, or extended into reading a protected label it
never needed. One process, one claim; the instruments live where the capability lives -- REMEDY's
instruments (the CQADupStack index, the protected-query index) live in `protected_filter.py`
because that is the module allowed to hold them, and PIN's instrument (`_hash_slice`) lives here
because hashing needs none of them.

PIN IS THE HASH AUTHORITY, not a mirror of what `remedy()` thought it computed. `pin()` re-reads
the remediated `.tsv`/`.jsonl` files fresh from `work/lotte/remediated/<topic>/<split>/` and
recomputes every hash from those bytes -- it never trusts a number carried over from inside
`remedy()`'s own process. That is why `protected_filter.remedy()` does not compute or record a
"hashes" field at all: there is exactly one place a hash is computed for the record, and it is
this one, computed from what is actually on disk.

`pin()` refuses over any slice that is not in `results/m8_lotte_remedy.json`'s surviving-slices
list -- a pin over a slice that never passed the re-screen would launder a contaminated slice into
a protected partition, silently turning a screening failure into a fact nobody checks again.

Forum queries only, always. LoTTE's `search` queries (`questions.search.tsv`, `qas.search.jsonl`)
are GooAQ-licensed non-commercial-research-only; this module never opens them, and in fact never
opens raw LoTTE at all any more -- it reads only the already-remediated copies REMEDY produced.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import m8base
import paths_guard

paths_guard.claim("m8src.freeze_lotte",
                  note="E10-REMEDY PIN: hash the already-remediated LoTTE shadow slices and "
                       "freeze them as a protected partition")

import probe_guard                                    # noqa: E402

REPO = m8base.REPO
REMEDIATED = REPO / "work" / "lotte" / "remediated"


def sha(obj):
    """Same pattern as `scripts/freeze_eval_assets.py`: sha256 of the sorted-key JSON encoding,
    so key order in a dict never perturbs the hash but list order (doc/query sequence) does."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- readers for the REMEDIATED slices

def _read_collection(path):
    doc_ids, doc_texts = [], []
    with open(path) as fh:
        for line in fh:
            t = line.rstrip("\n").split("\t", 1)
            if len(t) == 2:
                doc_ids.append(t[0])
                doc_texts.append(t[1])
    return doc_ids, doc_texts


def _read_questions(path):
    q_ids, q_texts = [], []
    with open(path) as fh:
        for line in fh:
            t = line.rstrip("\n").split("\t", 1)
            if len(t) == 2:
                q_ids.append(t[0])
                q_texts.append(t[1])
    return q_ids, q_texts


def _read_qas(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _require(path, slice_key):
    if not path.exists():
        raise SystemExit(
            f"E10-REMEDY PIN: {path} is missing for slice {slice_key!r}. Refusing to silently "
            f"skip a slice -- fix the input or drop the slice by name, not by omission.")


def _hash_slice(doc_ids, doc_texts, q_ids, q_texts, qrels_obj):
    """Five separate hashes over doc ids, doc texts, query ids, query texts and qrels -- the
    `scripts/freeze_eval_assets.py` pattern: doc-side lists hash in their natural (file) order,
    query-side lists hash ordered by SORTED qid so the pin does not depend on file row order."""
    q_by_id = dict(zip(q_ids, q_texts))
    return {
        "doc_ids_sha256": sha(doc_ids),
        "doc_texts_sha256": sha(doc_texts),
        "query_ids_sha256": sha(sorted(q_ids)),
        "query_texts_sha256": sha([q_by_id[q] for q in sorted(q_ids)]),
        "qrels_sha256": sha(qrels_obj),
    }


# ---------------------------------------------------------------- pin / verify ------------------

def pin(keys=None):
    """Freeze: read the remediated slices already on disk, hash them, and pin the hashes as the
    protected partition's fingerprint. `keys` defaults to every slice `m8_lotte_remedy.json` marks
    SURVIVES; passing an explicit subset is still checked against that same list -- see the module
    docstring for why a slice outside it is refused rather than silently pinned."""
    remedy_p = REPO / "results" / "m8_lotte_remedy.json"
    if not remedy_p.exists():
        raise SystemExit(
            f"E10-REMEDY PIN: {remedy_p} does not exist -- run `protected_filter.py remedy` "
            f"first. A pin with no remedy record behind it cannot know which slices actually "
            f"passed the re-screen.")
    surviving = set(json.loads(remedy_p.read_text()).get("surviving_slices") or [])
    if not surviving:
        raise SystemExit(f"E10-REMEDY PIN: {remedy_p} lists no surviving slices -- nothing to pin.")

    targets = sorted(keys) if keys is not None else sorted(surviving)
    bad = [k for k in targets if k not in surviving]
    if bad:
        raise SystemExit(
            f"E10-REMEDY PIN: {bad} did not pass remedy + re-screen per {remedy_p} -- a pin over "
            f"a slice that never survived the re-screen would launder a contaminated slice into a "
            f"protected partition.")

    t0 = time.time()
    slices = {}
    for key in targets:
        topic, split = key.split("/")
        d = REMEDIATED / topic / split
        for name in ("collection.tsv", "questions.forum.tsv", "qas.forum.jsonl"):
            _require(d / name, key)
        doc_ids, doc_texts = _read_collection(d / "collection.tsv")
        q_ids, q_texts = _read_questions(d / "questions.forum.tsv")
        qas_rows = _read_qas(d / "qas.forum.jsonl")
        qrels_obj = {str(r["qid"]): sorted(str(p) for p in r["answer_pids"]) for r in qas_rows}
        slices[key] = {
            "n_docs": len(doc_ids), "n_queries": len(q_ids),
            "n_qrels_pairs": sum(len(v) for v in qrels_obj.values()),
            "hashes": _hash_slice(doc_ids, doc_texts, q_ids, q_texts, qrels_obj),
            "read_relpath": str(d.relative_to(REPO)),
        }

    out = {
        "_note": "LEDGER 2026-08-29 E10-REMEDY PIN. The hash authority: computed fresh from the "
                 "files remedy() wrote, never from a number carried over inside remedy()'s own "
                 "process -- see the module docstring.",
        "remedy_source": str(remedy_p.relative_to(REPO)),
        "pinned_slices": targets,
        "slices": slices,
        "seconds": round(time.time() - t0, 1),
    }
    dest = REPO / "results" / "m8_lotte_pin.json"
    probe_guard.write_result(dest, out, "E10-REMEDY")
    print(f"pinned {len(slices)} slices -> {dest}", flush=True)
    return out


def verify():
    """Re-read what `pin()` wrote and recompute the five hashes fresh from
    `work/lotte/remediated/` -- refuses on ANY mismatch. Does not re-screen (that already happened
    in `protected_filter.remedy()`); this only confirms the files on disk still hash to what
    `pin()` recorded."""
    pin_p = REPO / "results" / "m8_lotte_pin.json"
    if not pin_p.exists():
        raise SystemExit(f"E10-REMEDY verify: {pin_p} does not exist -- run `pin` first.")
    data = json.loads(pin_p.read_text())

    checked, problems = [], []
    for key, info in data["slices"].items():
        topic, split = key.split("/")
        d = REMEDIATED / topic / split
        for name in ("collection.tsv", "questions.forum.tsv", "qas.forum.jsonl"):
            _require(d / name, key)
        doc_ids, doc_texts = _read_collection(d / "collection.tsv")
        q_ids, q_texts = _read_questions(d / "questions.forum.tsv")
        qas_rows = _read_qas(d / "qas.forum.jsonl")
        qrels_obj = {str(r["qid"]): sorted(str(p) for p in r["answer_pids"]) for r in qas_rows}
        recomputed = _hash_slice(doc_ids, doc_texts, q_ids, q_texts, qrels_obj)
        for k, v in recomputed.items():
            pinned = info["hashes"][k]
            if pinned != v:
                problems.append(f"{key}: {k} mismatch (pinned {pinned[:12]} recomputed {v[:12]})")
        checked.append(key)

    if problems:
        raise SystemExit("E10-REMEDY verify FAILED:\n  " + "\n  ".join(problems))
    print(f"verify OK: {len(checked)} pinned slices re-hashed, all pins match: {checked}",
          flush=True)
    return {"verified": True, "n_slices_checked": len(checked), "slices": checked}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pin", "verify"])
    a = ap.parse_args()
    if a.step == "pin":
        pin()
    else:
        verify()
    return 0


if __name__ == "__main__":
    sys.exit(main())
