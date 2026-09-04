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


# Dylan 2026-09-04 permitted non-commercial data for VALIDATION, not training. This guard is
# training-scope only (it walks a run's cfg.sources), so the rule change does not relax it and it
# must not be weakened: validation caches simply never appear here. See
# research/m7-data-licensing.md, "Rule change 2026-09-04".
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
    chain, hashes = [run_id], {}
    while chain:                       # a checkpoint init inherits its parent's training sources
        rid = chain.pop()
        if rid in hashes:
            continue
        p = WORK / "runs" / f"{rid}.json"
        if not p.exists():
            # NOT `continue`. The manifest this guard writes asserts "no non-commercial training
            # source in the lineage"; with an ancestor's record missing, that assurance is
            # unsupported. Failing open let the freeze record a claim it could not prove.
            raise SystemExit(
                f"FREEZE REFUSED: {run_id}'s lineage includes {rid}, whose run record {p} is "
                "missing, so the training sources of that ancestor cannot be established. The "
                "releasability claim is unprovable, not satisfied.")
        # Read the bytes ONCE, hash THOSE bytes, parse THOSE bytes. The first version validated
        # `p.read_text()` and then hashed the file again afterwards, so a record replaced between
        # the two reads was validated as one content and recorded under another's hash (Codex
        # pre-freeze review 2026-08-28, MAJOR 6).
        raw = p.read_bytes()
        hashes[rid] = hashlib.sha256(raw).hexdigest()
        c = json.loads(raw)["cfg"]
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
    # The hash of every record inspected, not just the ids: the guard reads gitignored, mutable
    # `work/runs/<id>.json` files, and the freeze recorded only which ids it walked -- so nothing
    # pinned WHAT it read (Codex review #4, MAJOR 11).
    return {rid: hashes[rid] for rid in sorted(hashes)}


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


def load_selected_fusion(run_id, table_sha, meta_sha, meta):
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
                "table_sha256": table_sha,
                "table_meta_sha256": meta_sha,
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
    # The recorded BM25 run keys were written and never read, so a bm25s/PyStemmer upgrade between
    # selection and freeze went unnoticed: the parameter was fitted under one lexical function and
    # applied under another (Codex review #4, MAJOR 2). The full content keys cannot be re-derived
    # here without loading the dev corpora, but the parts that make the FUNCTION are checked, and
    # an empty block is now refused rather than normalised.
    keys = (sel or {}).get("bm25_run_keys")
    comps = spec.get("components") or []
    if not isinstance(keys, dict) or not keys:
        problems.append("the fusion spec records no `bm25_run_keys`; re-run select_fusion.py so "
                        "the lexical runs the parameter was fitted against are pinned")
    else:
        if sorted(keys) != sorted(comps):
            problems.append(f"bm25_run_keys cover {sorted(keys)} but the selection ran on "
                            f"{sorted(comps)}")
        live_v, live_c = fusion._pkg_versions(), fusion.BM25_CONFIG
        for c, k in keys.items():
            if not isinstance(k, dict):
                problems.append(f"bm25_run_keys[{c}] is not a key object")
                continue
            if k.get("config") != live_c:
                problems.append(f"bm25_run_keys[{c}] was built with BM25 parameters "
                                f"{k.get('config')} but this environment has {live_c}")
            if k.get("versions") != live_v:
                problems.append(f"bm25_run_keys[{c}] was built with {k.get('versions')} but this "
                                f"environment has {live_v}; the lexical function has changed since "
                                "the parameter was selected")
            if k.get("depth") != fusion.DEPTH:
                problems.append(f"bm25_run_keys[{c}] was built at depth {k.get('depth')}")
    # The frozen (family, param) must be the argmax of the grid the selection actually searched --
    # not merely a well-formed pair. Without this, editing the spec (or FREEZE.json) to any other
    # grid point passed every check (Codex review #4, MAJOR 3).
    grid = spec.get("grid") or []
    if not grid:
        problems.append("the fusion spec records no `grid`, so its winner cannot be checked")
    else:
        try:
            best = max(grid, key=lambda r: (float(r["macro"]), 0))
        except (KeyError, TypeError, ValueError):
            best = None
            problems.append("the fusion spec's `grid` is malformed")
        if best is not None:
            top = [r for r in grid if float(r["macro"]) == float(best["macro"])]
            if not any(r["family"] == spec.get("family") and r["param"] == spec.get("param")
                       for r in top):
                problems.append(
                    f"the frozen point ({spec.get('family')}, {spec.get('param')}) is not the "
                    f"argmax of its own grid; the best row is ({best['family']}, {best['param']}) "
                    f"at macro {best['macro']}")
            if spec.get("dev_macro") is not None and \
                    abs(float(spec["dev_macro"]) - float(best["macro"])) > 1e-12:
                problems.append("the spec's dev_macro is not the grid's best macro")
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


REQUIRED_GATE_CONDITIONS = ("G1_stage0_above_potion", "G2_capacity_probe",
                            "G3_candidate_above_bm25", "G4_int8_equivalence")


def assert_gate_passed(run_id, table_sha, meta_sha):
    """The freeze must be preceded by a PASSing gate ON THIS ARTIFACT.

    `gate.py` writes the artifact's hashes into its result; nothing consulted them, so a freeze
    could follow a gate run on a different (or an older) table, or no gate at all. Takes the
    hashes rather than the paths so the caller can hash the bytes ONCE -- re-hashing here and
    again while building the manifest left a window in which the table could be replaced between
    the two (Codex review #4, MAJOR 4).

    Hardened per the 2026-08-28 pre-freeze review (BLOCKER 6): a nonempty all-true condition
    mapping was ALL this required, so a diagnostic subset run -- which overwrote the official gate
    file -- or a hand-edited one-condition file was accepted as GO. Now the exact registered
    condition set, the pinned component list, a real (non-substituted) Stage-0 for G1, the
    per-query dump's bytes, and a clean committed evaluator identity are all required."""
    p = REPO / "results" / f"m7_gate_{run_id}.json"
    if not p.exists():
        raise SystemExit(f"FREEZE REFUSED: no gate result at {p.relative_to(REPO)}. The gate is "
                         "the mechanical eligibility audit that must precede the freeze "
                         "(run_freeze_prep.sh step 3).")
    g = json.loads(p.read_text())
    art = g.get("artifact")
    problems = []
    if set(g.get("conditions") or {}) != set(REQUIRED_GATE_CONDITIONS):
        problems.append(f"gate conditions are {sorted(g.get('conditions') or {})}, not the "
                        f"registered set {sorted(REQUIRED_GATE_CONDITIONS)}")
    if g.get("diagnostic_subset"):
        problems.append("the gate result is marked diagnostic_subset; a freeze needs the full "
                        "pinned suite")
    import dev_eval
    pinned = dev_eval.dev_components()
    if g.get("components") != pinned:
        problems.append(f"the gate ran on components {g.get('components')}, not the pinned dev "
                        f"suite {pinned}")
    g1 = (g.get("conditions") or {}).get("G1_stage0_above_potion") or {}
    if g1.get("checkpoint") in (None, run_id) or "SUBSTITUTED" in str(g1.get("note", "")):
        problems.append("G1 was not judged on a real Stage-0 checkpoint (substituted or missing); "
                        "the registered gate defines G1 on the Stage-0 distilled table")
    dump = g.get("per_query_dump") or {}
    dp = REPO / "results" / str(dump.get("path", ""))
    if not dump or not dp.exists():
        problems.append("the gate's unrounded per-query dump is missing")
    elif sha256_file(dp) != dump.get("file_sha256"):
        problems.append(f"the gate's per-query dump {dp.name} does not match the hash the gate "
                        "recorded; it has been replaced or edited")
    ci = g.get("code_identity") or {}
    if not ci.get("git_head"):
        problems.append("the gate records no code identity (git_head)")
    if ci.get("m7src_dirty") is not False:
        problems.append("the gate ran from a DIRTY m7src tree (code_identity.m7src_dirty is not "
                        "false); re-run it from a clean commit so its code identity is real")
    # `is True`, not truthiness: `"PASS": "false"` is a non-empty string and used to pass. And the
    # summary flag is not enough on its own -- every condition must independently say `pass: true`,
    # so a hand-set or stale summary cannot carry a failing condition through
    # (Codex review #4, MAJOR 5).
    conds = g.get("conditions") or {}
    if g.get("PASS") is not True:
        failed = [k for k, v in conds.items() if v.get("pass") is not True]
        problems.append(f"the gate for {run_id} is NO-GO (PASS={g.get('PASS')!r}; "
                        f"failed: {failed or 'unknown'})")
    if not conds:
        problems.append("the gate result records no conditions")
    else:
        bad = [k for k, v in conds.items() if not isinstance(v, dict) or v.get("pass") is not True]
        if bad:
            problems.append(f"gate conditions not passing: {bad}")
    if not isinstance(art, dict):
        problems.append(f"the gate result has no `artifact` block (got {type(art).__name__})")
        art = {}
    if art.get("sha256") != table_sha:
        problems.append("the gate ran on a different table than the one being frozen "
                        f"(gate {str(art.get('sha256'))[:16]!r} vs {table_sha[:16]!r})")
    if art.get("meta_sha256") != meta_sha:
        problems.append("the gate ran against different table metadata than the one being frozen")
    if g.get("run_id") != run_id:
        problems.append(f"gate file names run_id {g.get('run_id')!r}, not {run_id!r}")
    if problems:
        raise SystemExit("FREEZE REFUSED:\n  - " + "\n  - ".join(problems))
    # Keep the gate's own artifact hashes in the record. Discarding them hid the case where the
    # gate, the selection and the frozen bytes were not all the same object.
    return {"path": p.name, "PASS": True, "artifact": art,
            "conditions": {k: bool(v.get("pass")) for k, v in conds.items()}}


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
    # Hash the bytes ONCE and use that hash everywhere below. Verifying the gate and the selection
    # against a fresh hash, then re-hashing while building the manifest, left a window in which the
    # release could be replaced: the freeze would then carry gate evidence and a fusion spec for A
    # beside table hashes for B, and `load_and_verify` accepts B when A and B share preprocessing
    # and encoder -- as two checkpoints normally do (Codex review #4, MAJOR 4).
    table_sha, meta_sha = sha256_file(npz), sha256_file(meta_p)
    table_bytes = npz.stat().st_size
    gate_evidence = assert_gate_passed(run_id, table_sha, meta_sha)
    fusion_spec = load_selected_fusion(run_id, table_sha, meta_sha, meta)
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
        "training_lineage": sorted(lineage),
        "training_lineage_record_sha256": lineage,
        "release_licence_check": "no non-commercial training source in the lineage "
                                 "(freeze.assert_releasable); the sha256 of every run record "
                                 "inspected is above, because those records are gitignored and "
                                 "mutable",
        "table_relpath": f"work/runs/{npz.name}",
        "training_checkpoint_sha256": sha256_file(WORK / "runs" / f"{run_id}.npz"),
        "table_sha256": table_sha,
        "table_meta_sha256": meta_sha,
        "table_bytes": table_bytes,
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
    # Close the window from the other side too: if the release moved while this function ran, the
    # gate evidence and the fusion binding above describe bytes that are no longer there.
    if sha256_file(npz) != table_sha or sha256_file(meta_p) != meta_sha:
        raise SystemExit("FREEZE REFUSED: the release artifact changed while the freeze was being "
                         "written. Nothing was frozen; re-run the gate and the fusion selection.")
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
    # RE-BIND TO THE SELECTION ITSELF. Everything above reads FREEZE.json, which is a committed
    # text file: editing `fusion.param` and `released_system` consistently after `freeze.write`
    # passed every check, so "the released system is decided by the dev grid" was false at
    # final-run time (Codex review #4, MAJOR 3). Compare against the selection's own output and
    # against the grid it searched.
    if spec:
        sel_p = WORK / "runs" / f"{b.get('run_id')}.fusion.json"
        if not sel_p.exists():
            problems.append(f"the fusion selection {sel_p} is gone, so FREEZE.json's fusion block "
                            "cannot be shown to be the one that was selected")
        elif json.loads(sel_p.read_text()) != spec:
            problems.append(f"FREEZE.json's fusion block differs from {sel_p.name}; it has been "
                            "edited since the selection ran")
        grid = spec.get("grid") or []
        if grid:
            best = max(grid, key=lambda r: float(r["macro"]))
            top = [r for r in grid if float(r["macro"]) == float(best["macro"])]
            if not any(r["family"] == spec.get("family") and r["param"] == spec.get("param")
                       for r in top):
                problems.append(f"the frozen fusion point ({spec.get('family')}, "
                                f"{spec.get('param')}) is not the argmax of its own grid")
        else:
            problems.append("the frozen fusion spec records no grid")
        # THE LEXICAL FUNCTION MUST BE THE ONE THE PARAMETER WAS SELECTED UNDER. The final run
        # builds BM25 uncached, and nothing compared the installed bm25s/PyStemmer to the versions
        # the selection's cache keys recorded -- so a package upgrade between freeze and final run
        # silently changed the fused system, i.e. C3 would judge a function never selected on dev
        # (Codex pre-freeze review 2026-08-28, BLOCKER 1).
        sel_keys = (spec.get("selected_on") or {}).get("bm25_run_keys") or {}
        if not sel_keys:
            problems.append("the frozen fusion spec records no bm25_run_keys, so the lexical "
                            "function it was fitted against cannot be re-verified")
        else:
            live_v = _fusion._pkg_versions()
            for comp, k in sorted(sel_keys.items()):
                if k.get("versions") != live_v:
                    problems.append(f"BM25 package drift since the fusion selection ({comp}: "
                                    f"selected under {k.get('versions')}, running {live_v}); the "
                                    "fused system would not be the function selected on dev")
                    break
                if k.get("config") != _fusion.BM25_CONFIG:
                    problems.append(f"BM25 config drift since the fusion selection ({comp}: "
                                    f"{k.get('config')} vs live {_fusion.BM25_CONFIG})")
                    break
    # The teacher's remote code is re-verified by final_run before any protected access; here we
    # only check the pin file itself has not been swapped since the freeze.
    pin = REPO / "results" / "m7_teacher_code_pin.json"
    if b.get("teacher_code_pin_sha256"):
        if not pin.exists():
            problems.append("results/m7_teacher_code_pin.json is missing")
        elif sha256_file(pin) != b["teacher_code_pin_sha256"]:
            problems.append("results/m7_teacher_code_pin.json changed after the freeze")
    # RE-RUN THE PREDICATES, do not trust their recorded verdicts. FREEZE.json is a committed
    # text file: a hand-authored or edited one carrying valid table/fusion/manifest hashes but no
    # real gate or licence evidence passed everything above, so the final run could bypass
    # `freeze.write`'s two protections entirely (Codex pre-freeze review 2026-08-28, BLOCKER 7).
    # Both predicates read committed or content-hashed evidence and raise with their own reasons.
    if not problems:
        try:
            lineage = assert_releasable(b["run_id"])
        except SystemExit as e:
            problems.append(f"releasability re-check failed: {e}")
        else:
            if lineage != b.get("training_lineage_record_sha256"):
                problems.append("the training-lineage run records no longer match the hashes "
                                "frozen in FREEZE.json")
        try:
            assert_gate_passed(b["run_id"], b["table_sha256"], b["table_meta_sha256"])
        except SystemExit as e:
            problems.append(f"gate re-check failed: {e}")
    if problems:
        raise SystemExit("FINAL RUN REFUSED:\n  - " + "\n  - ".join(problems))
    return b
