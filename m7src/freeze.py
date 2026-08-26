"""The freeze manifest: what the pushed freeze commit must pin, beyond the code itself.

The mandate freezes "config, code, fusion params, preprocessing, dev-suite hashes" in a commit
pushed to GitHub. A commit hash alone does not do that: the trained table lives under the
gitignored work/ tree, and preprocessing / fusion / released-system would otherwise be free
choices made on the final-run command line, after the freeze. So the freeze commit must also
contain m7/FREEZE.json, which pins:

  * sha256 of the released table .npz and of its .meta.json
  * the preprocessing fingerprint (read from the table's own metadata, never from the CLI)
  * the fusion family and parameter selected on dev, and which system the Tier-1 claim is judged on
  * the dev-suite component hashes the selection was made against

`final_run.py` reads every one of these from the committed file and recomputes the table hash,
refusing to start on any mismatch.
"""
import hashlib
import json
from pathlib import Path

from _paths import REPO, WORK

FREEZE = REPO / "m7" / "FREEZE.json"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(run_id, fusion_spec, released_system, dev_macro=None, notes=None):
    npz = WORK / "runs" / f"{run_id}.npz"
    meta_p = WORK / "runs" / f"{run_id}.meta.json"
    meta = json.loads(meta_p.read_text())
    pre = meta["preproc"]
    blob = {
        "_note": "Written before the freeze commit. final_run.py reads preprocessing, fusion and "
                 "the released-system choice from here, never from the command line, and "
                 "recomputes table_sha256 before scoring anything.",
        "run_id": run_id,
        "table_relpath": f"work/runs/{run_id}.npz",
        "table_sha256": sha256_file(npz),
        "table_meta_sha256": sha256_file(meta_p),
        "table_bytes": npz.stat().st_size,
        "preproc": pre,
        "preproc_fingerprint": meta["preproc_fingerprint"],
        "learned_weights": meta.get("learned_weights"),
        "teacher": meta.get("teacher"),
        "teacher_revision": meta.get("teacher_revision"),
        "fusion": fusion_spec,
        "released_system": released_system,
        "dev_macro_at_freeze": dev_macro,
        "dev_manifest_sha256": sha256_file(REPO / "results" / "m7_dev_manifest.json"),
        "eval_manifest_sha256": sha256_file(REPO / "results" / "eval_manifest.json"),
        "perquery_sha256": sha256_file(REPO / "results" / "perquery.json"),
        "notes": notes or "",
    }
    FREEZE.write_text(json.dumps(blob, indent=1))
    print(json.dumps(blob, indent=1))
    return blob


def load_and_verify():
    """-> the freeze blob, with the table's bytes and metadata re-hashed against it."""
    if not FREEZE.exists():
        raise SystemExit(f"FINAL RUN REFUSED: {FREEZE} does not exist. Write it with "
                         "freeze.write(...) and commit it as the freeze commit.")
    b = json.loads(FREEZE.read_text())
    npz = REPO / b["table_relpath"]
    meta_p = npz.with_suffix("").with_suffix(".meta.json") if npz.suffixes else None
    meta_p = npz.parent / (npz.stem + ".meta.json")
    problems = []
    if not npz.exists():
        problems.append(f"table {npz} is missing")
    else:
        if sha256_file(npz) != b["table_sha256"]:
            problems.append(f"table sha256 mismatch: {npz} is not the frozen artifact")
        if npz.stat().st_size != b["table_bytes"]:
            problems.append("table byte size mismatch")
    if not meta_p.exists():
        problems.append(f"table metadata {meta_p} is missing")
    elif sha256_file(meta_p) != b["table_meta_sha256"]:
        problems.append("table metadata sha256 mismatch")
    else:
        meta = json.loads(meta_p.read_text())
        if meta["preproc_fingerprint"] != b["preproc_fingerprint"]:
            problems.append("the table's own preprocessing fingerprint disagrees with FREEZE.json")
    for name, key in (("results/m7_dev_manifest.json", "dev_manifest_sha256"),
                      ("results/eval_manifest.json", "eval_manifest_sha256"),
                      ("results/perquery.json", "perquery_sha256")):
        if sha256_file(REPO / name) != b[key]:
            problems.append(f"{name} changed after the freeze")
    if b["released_system"] == "fusion" and not b.get("fusion"):
        problems.append("released_system is 'fusion' but no fusion spec is frozen")
    if problems:
        raise SystemExit("FINAL RUN REFUSED:\n  - " + "\n  - ".join(problems))
    return b
