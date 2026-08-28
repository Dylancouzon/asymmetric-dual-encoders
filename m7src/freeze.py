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
import os
from pathlib import Path

import encoders
from _paths import REPO, WORK

FREEZE = REPO / "m7" / "FREEZE.json"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


NON_COMMERCIAL_SOURCES = ("msmarco", "ms-marco", "ms_marco")


def assert_releasable(run_id):
    """Refuse to freeze an artifact trained on data that cannot be released commercially.

    MS MARCO is "non-commercial research purposes only" and is permanently excluded from the
    RELEASE stack (m7/LEDGER.md). The final M7 task deliberately trains ONE variant on it to
    measure what the exclusion costs, and that variant must never reach a freeze, a fusion, or an
    upload. Enforced here rather than by intention, because the two artifacts are otherwise
    indistinguishable on disk and the mistake would be irreversible once published."""
    import mix
    # `sources: []` in a Cfg means "every available source", so an empty list must NOT read as
    # "no non-commercial data" -- that is precisely how the research-only variant would slip
    # through. Resolve it against what is on disk now, and say so: this reflects the CURRENT mix,
    # which is the conservative direction (a source added later still trips the guard).
    chain, seen = [run_id], set()
    while chain:                       # a checkpoint init inherits its parent's training sources
        rid = chain.pop()
        if rid in seen:
            continue
        seen.add(rid)
        p = WORK / "runs" / f"{rid}.json"
        if not p.exists():
            continue
        c = json.loads(p.read_text())["cfg"]
        srcs = [str(s).lower() for s in (c.get("sources") or mix.available_sources())]
        bad = [s for s in srcs if any(n in s for n in NON_COMMERCIAL_SOURCES)]
        if bad:
            raise SystemExit(
                f"FREEZE REFUSED: {run_id} inherits non-commercially-licensed training data "
                f"{bad} via {rid}. MS MARCO is research-only; it may not enter a released "
                f"artifact (m7/LEDGER.md, 'the clean-stack tax').")
        init = str(c.get("init", ""))
        if init.startswith("run:"):
            chain.append(init.split(":", 1)[1])
    return sorted(seen)


def write(run_id, fusion_spec, released_system, dev_macro=None, notes=None):
    # Freeze the RELEASE artifact (weights folded), never the raw training checkpoint: the tier
    # claims are about what ships (review #2 BLOCKER 2). ensure_release is idempotent.
    from table import ensure_release
    lineage = assert_releasable(run_id)
    npz = ensure_release(WORK / "runs" / f"{run_id}.npz")
    meta_p = npz.with_name(npz.stem + ".meta.json")
    meta = json.loads(meta_p.read_text())
    pre = meta["preproc"]
    blob = {
        "_note": "Written before the freeze commit. final_run.py reads preprocessing, fusion and "
                 "the released-system choice from here, never from the command line, and "
                 "recomputes table_sha256 before scoring anything.",
        "run_id": run_id,
        "training_lineage": lineage,
        "release_licence_check": "no non-commercial training source in the lineage "
                                 "(freeze.assert_releasable)",
        "table_relpath": f"work/runs/{npz.name}",
        "training_checkpoint_sha256": sha256_file(WORK / "runs" / f"{run_id}.npz"),
        "table_sha256": sha256_file(npz),
        "table_meta_sha256": sha256_file(meta_p),
        "table_bytes": npz.stat().st_size,
        "preproc": pre,
        "preproc_fingerprint": meta["preproc_fingerprint"],
        "learned_weights": meta.get("learned_weights"),
        "teacher": meta.get("teacher"),
        "teacher_revision": meta.get("teacher_revision"),
        # The FULL encoder identity, not just repo+revision. Every field here changes the vectors
        # the table is scored against, and load_and_verify refuses if the running process's active
        # encoder differs on any of them. Codex gate 2026-08-26, BLOCKER 4: a stella table could be
        # evaluated against default bge-base with every existing hash guard passing, because
        # M7_ENCODER selects the encoder from the environment and nothing compared it to the freeze.
        "encoder_spec": encoder_fingerprint(),
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


def encoder_fingerprint(spec=None):
    """Every field of the active Spec that changes the vectors. Compared field by field at final-run
    time so a mismatch names the field rather than just failing a hash."""
    sp = spec or encoders.active()
    return {"name": sp.name, "repo": sp.repo, "revision": sp.revision, "dim": sp.dim,
            "pooling": sp.pooling, "post_dense": sp.post_dense,
            "query_prefix": sp.query_prefix, "doc_prefix": sp.doc_prefix,
            "max_length": sp.max_length, "tokenizer_id": sp.tokenizer_id,
            "vocab": sp.vocab, "cls_id": sp.cls_id,
            "config_kwargs": dict(sorted(sp.config_kwargs.items()))}


def encoder_drift(frozen_spec, spec=None):
    """{field: (frozen, live)} for every Spec field that disagrees. Separated from load_and_verify
    so it is testable without a FREEZE.json (test_freeze_guard.py)."""
    live = encoder_fingerprint(spec)
    return {k: (frozen_spec.get(k), live.get(k)) for k in live if frozen_spec.get(k) != live.get(k)}


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
    # The teacher the artifact was frozen with must be the teacher this process is running.
    frozen_spec = b.get("encoder_spec")
    if frozen_spec is None:
        problems.append("FREEZE.json predates encoder_spec: rewrite the freeze with the current "
                        "freeze.write() so the teacher identity is bound to the artifact")
    else:
        drift = encoder_drift(frozen_spec)
        if drift:
            problems.append("active encoder differs from the frozen one (M7_ENCODER is "
                            f"{os.environ.get('M7_ENCODER', '<unset>')!r}): "
                            + "; ".join(f"{k}: frozen={a!r} live={b2!r}" for k, (a, b2) in drift.items()))
    if b["released_system"] == "fusion" and not b.get("fusion"):
        problems.append("released_system is 'fusion' but no fusion spec is frozen")
    if problems:
        raise SystemExit("FINAL RUN REFUSED:\n  - " + "\n  - ".join(problems))
    return b
