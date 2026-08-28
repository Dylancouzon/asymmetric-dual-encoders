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
            # NOT `continue`. The manifest this guard writes asserts "no non-commercial training
            # source in the lineage"; with an ancestor's record missing, that assurance is
            # unsupported. Failing open let the freeze record a claim it could not prove.
            raise SystemExit(
                f"FREEZE REFUSED: {run_id}'s lineage includes {rid}, whose run record {p} is "
                "missing, so the training sources of that ancestor cannot be established. The "
                "releasability claim is unprovable, not satisfied.")
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


def assert_encoder_matches_artifact(meta, where):
    """The table's own `teacher` must be the encoder this process is running.

    `encoder_spec` is read from the AMBIENT environment (`M7_ENCODER`) while `teacher` comes from
    the artifact's metadata, and nothing compared them. `run_freeze_prep.sh` exports the encoder
    only inside its own child shell, so a `freeze.write(...)` from an ordinary shell records
    `teacher: stella` beside `encoder_spec: bge-base` -- and a final run in that same shell then
    passes `encoder_drift` (frozen bge == live bge) while serving stella's table against bge
    documents. With bge the dimensions differ and it crashes; with any other 1024-d encoder
    (arctic-l, bge-large, e5-large) it produces a plausible wrong number instead.
    Codex one-shot-path review 2026-08-28, BLOCKER 1.
    """
    sp = encoders.active()
    want = (meta.get("teacher"), meta.get("teacher_revision"))
    got = (sp.repo, sp.revision)
    if want != got and want != (None, None):
        raise SystemExit(
            f"{where} REFUSED: the artifact was built with teacher {want[0]}@{str(want[1])[:12]} "
            f"but the active encoder is {sp.name} = {got[0]}@{str(got[1])[:12]} "
            f"(M7_ENCODER={os.environ.get('M7_ENCODER', '<unset>')!r}). Set M7_ENCODER to the "
            "encoder the table was distilled from; a table is only valid against its own "
            "teacher's document vectors.")
    if meta.get("dim") is not None and int(meta["dim"]) != int(sp.dim):
        raise SystemExit(f"{where} REFUSED: table dim {meta['dim']} != active encoder dim {sp.dim}")


RELEASED_SYSTEMS = ("dense", "fusion")


def load_selected_fusion(run_id, npz, meta_p, meta):
    """Read the fusion spec from the SELECTION's own output and prove it describes this artifact.

    `write` used to take whatever spec its caller passed, and `select_fusion` recorded nothing
    about what it had been fitted on -- so a parameter selected on artifact A could be frozen with
    artifact B, undetected, and applied in the one-shot run (Codex one-shot-path review
    2026-08-28, MAJOR 1). Every field here is re-derived from the artifact on disk and from the
    live environment; none of it is taken on trust.
    """
    import fusion
    p = WORK / "runs" / f"{run_id}.fusion.json"
    if not p.exists():
        raise SystemExit(f"FREEZE REFUSED: no fusion selection at {p}. Run "
                         f"`select_fusion.py {run_id}` (step 1 of run_freeze_prep.sh) first; a "
                         "fusion parameter may not be supplied by hand.")
    spec = json.loads(p.read_text())
    problems = []
    sel = spec.get("selected_on")
    if not isinstance(sel, dict):
        problems.append("the spec carries no `selected_on` block: it predates provenance binding "
                        "and cannot be shown to have been fitted on this artifact. Re-run "
                        f"select_fusion.py {run_id}.")
    else:
        want = {"run_id": run_id,
                "table_sha256": sha256_file(npz),
                "table_meta_sha256": sha256_file(meta_p),
                "preproc_fingerprint": meta["preproc_fingerprint"],
                "dev_manifest_sha256": sha256_file(REPO / "results" / "m7_dev_manifest.json")}
        for k, v in want.items():
            if sel.get(k) != v:
                problems.append(f"fusion spec was selected on {k}={str(sel.get(k))[:16]!r} but "
                                f"the artifact being frozen has {str(v)[:16]!r}")
        if sel.get("preproc") != meta["preproc"]:
            problems.append("fusion spec's preproc differs from the artifact's own metadata")
        drift = encoder_drift(sel.get("encoder_spec") or {})
        if drift:
            problems.append("the fusion selection ran under a different encoder: "
                            + "; ".join(f"{k}: selected={a!r} live={b!r}" for k, (a, b) in drift.items()))
    if spec.get("family") not in fusion.FAMILIES:
        problems.append(f"unknown fusion family {spec.get('family')!r}; known: {fusion.FAMILIES}")
    if not isinstance(spec.get("param"), (int, float)) or isinstance(spec.get("param"), bool):
        problems.append(f"fusion param {spec.get('param')!r} is not a number")
    if spec.get("depth") != fusion.DEPTH:
        problems.append(f"fusion was selected at depth {spec.get('depth')} but the final run "
                        f"retrieves to {fusion.DEPTH}; both families are depth-sensitive")
    # The committed copy is what a reviewer reads; it must be the same object as the one frozen.
    pub = REPO / "results" / f"m7_fusion_{run_id}.json"
    if not pub.exists():
        problems.append(f"{pub.relative_to(REPO)} is missing: the selection's committed copy")
    elif json.loads(pub.read_text()) != spec:
        problems.append(f"{pub.relative_to(REPO)} differs from {p.relative_to(REPO.parent)}; one "
                        "of the two has been edited since the selection ran")
    if problems:
        raise SystemExit("FREEZE REFUSED:\n  - " + "\n  - ".join(problems))
    return spec


def assert_gate_passed(run_id, npz, meta_p):
    """The freeze must be preceded by a PASSing gate ON THIS ARTIFACT.

    `gate.py` writes the artifact's hashes into its result; nothing consulted them, so a freeze
    could follow a gate run on a different (or an older) table, or no gate at all."""
    p = REPO / "results" / f"m7_gate_{run_id}.json"
    if not p.exists():
        raise SystemExit(f"FREEZE REFUSED: no gate result at {p.relative_to(REPO)}. The gate is "
                         "the mechanical eligibility audit that must precede the freeze "
                         "(run_freeze_prep.sh step 3).")
    g = json.loads(p.read_text())
    art = g.get("artifact") or {}
    problems = []
    if not g.get("PASS"):
        failed = [k for k, v in (g.get("conditions") or {}).items() if not v.get("pass")]
        problems.append(f"the gate for {run_id} is NO-GO (failed: {failed or 'unknown'})")
    if art.get("sha256") != sha256_file(npz):
        problems.append("the gate ran on a different table than the one being frozen "
                        f"(gate {str(art.get('sha256'))[:16]!r} vs {sha256_file(npz)[:16]!r})")
    if art.get("meta_sha256") != sha256_file(meta_p):
        problems.append("the gate ran against different table metadata than the one being frozen")
    if g.get("run_id") != run_id:
        problems.append(f"gate file names run_id {g.get('run_id')!r}, not {run_id!r}")
    if problems:
        raise SystemExit("FREEZE REFUSED:\n  - " + "\n  - ".join(problems))
    return {"path": p.name, "PASS": True,
            "conditions": {k: bool(v.get("pass")) for k, v in (g.get("conditions") or {}).items()}}


def write(run_id, released_system=None, dev_macro=None, notes=None):
    """Write m7/FREEZE.json for `run_id`.

    `released_system` is DERIVED from the fusion selection (the grid contains the dense-only
    endpoint, so "do not fuse" is decided mechanically); passing it is an optional cross-check,
    and a disagreement is fatal rather than an override.
    """
    # Freeze the RELEASE artifact (weights folded), never the raw training checkpoint: the tier
    # claims are about what ships (review #2 BLOCKER 2). ensure_release is idempotent.
    import fusion
    from table import ensure_release
    lineage = assert_releasable(run_id)
    npz = ensure_release(WORK / "runs" / f"{run_id}.npz")
    meta_p = npz.with_name(npz.stem + ".meta.json")
    meta = json.loads(meta_p.read_text())
    assert_encoder_matches_artifact(meta, "FREEZE")
    # The teacher runs under `trust_remote_code`, so the doc side of every reproduction executes
    # code the weights revision alone did not record. Re-hash the whole snapshot against the
    # committed pin and refuse if it has moved (m7src/teacher_code.py).
    import teacher_code
    code_ok, code_problems = teacher_code.verify()
    if not code_ok:
        raise SystemExit("FREEZE REFUSED: the teacher's pinned files do not match "
                         "results/m7_teacher_code_pin.json:\n  - " + "\n  - ".join(code_problems))
    gate_evidence = assert_gate_passed(run_id, npz, meta_p)
    fusion_spec = load_selected_fusion(run_id, npz, meta_p, meta)
    derived = "dense" if fusion.is_dense_only(fusion_spec) else "fusion"
    if released_system is not None and released_system != derived:
        raise SystemExit(
            f"FREEZE REFUSED: released_system={released_system!r} was passed, but the dev "
            f"selection picked {fusion_spec['family']} param={fusion_spec['param']}, which is "
            f"{derived!r}. Whether the released system fuses is decided by the selection grid "
            "(it contains the dense-only endpoint), not on the freeze command line.")
    released_system = derived
    if released_system not in RELEASED_SYSTEMS:      # unreachable via `derived`; a guard, not a branch
        raise SystemExit(f"FREEZE REFUSED: released_system must be one of {RELEASED_SYSTEMS}")
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
        # The trust_remote_code payload, verified above, not merely referenced.
        "teacher_code_pin_sha256": sha256_file(REPO / "results" / "m7_teacher_code_pin.json"),
        "teacher_code_verified": True,
        "fusion": fusion_spec,
        "fusion_spec_relpath": f"work/runs/{run_id}.fusion.json",
        "released_system": released_system,
        "released_system_derivation": (
            f"derived from the dev selection: family={fusion_spec['family']} "
            f"param={fusion_spec['param']}; the convex grid contains the dense-only endpoint "
            "w=1.0, so whether the released system fuses is decided by the same mechanical "
            "selection as the parameter"),
        "gate": gate_evidence,
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
        # catches a freeze written under the wrong ambient encoder: `teacher` came from the
        # artifact and `encoder_spec` from the environment, and they were never compared.
        for f, k in (("teacher", "teacher"), ("teacher_revision", "teacher_revision")):
            if meta.get(f) != b.get(k):
                problems.append(f"the table's {f} ({meta.get(f)}) disagrees with FREEZE.json "
                                f"({b.get(k)})")
        fs = b.get("encoder_spec") or {}
        if fs.get("repo") and meta.get("teacher") and fs["repo"] != meta["teacher"]:
            problems.append(f"FREEZE.json binds encoder {fs['repo']} but the frozen table was "
                            f"distilled from {meta['teacher']} -- the freeze was written with the "
                            "wrong M7_ENCODER")
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
    # An ENUM, not a free string. `final_run` selects the released system with
    # `"fusion" if released_system == "fusion" else "int8-table"`, so any typo -- "fused",
    # "Fusion", "dense+bm25" -- silently meant DENSE and the Tier-1 claim would have been made
    # about the wrong system (Codex one-shot-path review 2026-08-28, MAJOR 1).
    rs = b.get("released_system")
    if rs not in RELEASED_SYSTEMS:
        problems.append(f"released_system {rs!r} is not one of {RELEASED_SYSTEMS}")
    elif rs == "fusion" and not b.get("fusion"):
        problems.append("released_system is 'fusion' but no fusion spec is frozen")
    import fusion as _fusion
    spec = b.get("fusion") or {}
    if spec and spec.get("family") not in _fusion.FAMILIES:
        problems.append(f"frozen fusion family {spec.get('family')!r} is not one of "
                        f"{_fusion.FAMILIES}")
    elif spec and spec.get("depth") != _fusion.DEPTH:
        problems.append(f"fusion was selected at depth {spec.get('depth')} but this process "
                        f"retrieves to {_fusion.DEPTH}")
    elif spec and rs in RELEASED_SYSTEMS:
        derived = "dense" if _fusion.is_dense_only(spec) else "fusion"
        if derived != rs:
            problems.append(f"released_system is {rs!r} but the frozen selection "
                            f"({spec.get('family')} param={spec.get('param')}) is {derived!r}")
    if problems:
        raise SystemExit("FINAL RUN REFUSED:\n  - " + "\n  - ".join(problems))
    return b
