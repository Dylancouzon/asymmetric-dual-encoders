"""M13 — LoTTE read #1 as a small script (ruling R16): the pre-build veto and the observational row.

    .venv/bin/python m13src/lotte_gate13.py --write-manifest    # after both E records: writes m13/LOTTE_GATE_MANIFEST.json to commit and push
    .venv/bin/python m13src/lotte_gate13.py --preflight-only    # opens no LoTTE path, writes nothing
    .venv/bin/python m13src/lotte_gate13.py [--device cuda]     # THE read; writes m13/LOTTE_GATE.json
    .venv/bin/python m13src/lotte_gate13.py --recover           # complete a crashed read from its persisted slices

What it executes is registered, not decided here: `m13/LOTTE_GATE_REGISTRATION.json` (ruling R8)
names the seven remediated slices and their counts, nDCG@10 as the veto metric with Success@5 beside
it, the veto constants (margin 0.004, paired bootstrap B = 10,000 seed 903, one-sided 97.5% upper
bound, `inverted_cdf` by R17) and the identities; `m10/LOTTE_LOCK.md` fixes WHEN (after both 5M E
arms, before the build) and the two branches:

  * E1 selected **bs32** -> the selected recipe IS the anchor recipe: the veto is SKIPPED and
    forfeited, and the observational row is still read on `E-bs32`'s cycle-3 checkpoint.
  * E1 selected **bs128** -> the veto RUNS: candidate `E-bs128`, comparator `E-bs32`, both the A100
    cycle-3 checkpoints. A veto means the 200M build trains bs32 (`build13.check_gate` recomputes
    the decision from the recorded bootstrap and overrides the E1 batch).

Three committed AND PUSHED artifacts bind the read, and the read refuses without any of them:

  * **the manifest** `m13/LOTTE_GATE_MANIFEST.json` — the lock's "second manifest commit": both
    checkpoint shas, the arm records' shas, the full registered recipe, the student's dependency
    identity (tokenizer and backbone configuration hashes), the branch and the E1 verdict sha,
    written by `--write-manifest` and then committed and pushed by the operator. The gate requires
    it tracked, unmodified, with a commit present on a remote branch, and re-checks every field
    against the live records, the registered recipe knobs and the bytes on disk;
  * **the pin** `results/m8_lotte_pin.json` — `m8src/freeze_lotte.py pin`'s five hashes per slice
    (ruling R18: run immediately before the gate), under the same git provenance checks. Every slice
    is compared hash by hash before it is scored;
  * **the registration**, whose sha the record carries.

The read itself: under this module's own `m8src/paths_guard` entry (`m13src.lotte_gate13`, LEDGER 15
amendment 2026-09-10) it opens only `collection.tsv`, `questions.forum.tsv` and `qas.forum.jsonl` under
`work/lotte/remediated/<topic>/<split>/` — never the GooAQ-licensed search split, never the raw
archive. Stella encodes each collection once, shard-resumable and VERIFIED on reuse, into
`work/lotte/enc/`; the students are rebuilt from the exact checkpoint bytes that were hashed (one
read of the file, hashed and deserialised from the same buffer). Exact search, per-query nDCG@10 and
Success@5, the paired within-slice bootstrap and the veto rule.

One read. An exclusive `flock` under `work/lotte/gate13/` is held for the whole attempt, so two
processes cannot both read, and a receipt binding every input is created (O_EXCL, directory fsynced)
before the first LoTTE open. Each slice's outputs are persisted as it completes and their digests
journaled. A plain re-run REFUSES while the receipt exists; `--recover` takes the same lock and
completes the same read under the identical identity, accepting a persisted slice only if its digest
is journaled, its content is well-formed and its hashes are the pin's, and reading only the slices
without one. An existing `m13/LOTTE_GATE.json` refuses everything: there is no third read.

The record carries no query text, label or per-query row (those stay under `work/lotte/gate13/`),
and no refusal message prints an identifier from the surface. Two pitfalls this file exists to get
right: LoTTE's qids and pids are BOTH small integers, and `evalkit.run_from_arrays` drops a hit whose
doc id equals the query id, so queries are scored under a `q:` namespace; and `pytrec_eval` silently
omits a run qid without qrels, so the scored qid set is asserted every slice.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in ("m7src", "m8src", "m10src", "m13src"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

import build_lock as BL                         # noqa: E402
import run_arm as R                             # noqa: E402

refuse = R.refuse
sha256_file = R.sha256_file
slug = R.slug

E_ARMS = ("E-bs32", "E-bs128")
SCREEN_DOSE = 5_000_000
GUARD_ENTRY = "m13src.lotte_gate13"
SLICE_FILES = ("collection.tsv", "questions.forum.tsv", "qas.forum.jsonl")
CODE_IDENTITY_FILES = ("m13src/lotte_gate13.py", "m10src/nano10.py", "m7src/evalkit.py",
                       "m7src/teacher.py")
QNS = "q:"                                      # the query-id namespace (see the module docstring)
QUANTILE_METHOD = "inverted_cdf"                # ruling R17: the ONLY admissible upper-bound method
RECIPE_FIELDS = ("student", "n_layers", "head", "objective", "pattern", "dose_examples", "batch")
IDENTITY_FIELDS = ("branch", "e1_batch", "e1_verdict_sha256", "candidate_sha256", "comparator_sha256",
                   "registration_sha256", "manifest_sha256", "manifest_commit", "pin_sha256",
                   "code_identity", "device", "slice_order")
HASH_FIELDS = ("doc_ids_sha256", "doc_texts_sha256", "query_ids_sha256", "query_texts_sha256",
               "qrels_sha256")


@dataclass
class Config:
    """Every path and injected component the read touches. Production defaults are the real ones;
    the tests build one pointing at a synthetic git tree (with a bare origin) and inject the
    document encoder, the student loader and the dependency identity."""

    repo: Path = REPO
    registration_path: Path = REPO / "m13" / "LOTTE_GATE_REGISTRATION.json"
    build_config_path: Path = REPO / "m13" / "build_config.json"
    record_path: Path = REPO / "m13" / "LOTTE_GATE.json"
    manifest_path: Path = REPO / "m13" / "LOTTE_GATE_MANIFEST.json"
    pin_path: Path = REPO / "results" / "m8_lotte_pin.json"
    verdicts_path: Path = REPO / "results" / "m10_screen_verdicts.json"
    registry_path: Path = REPO / "m10" / "screen_registry.json"
    arm_records_dir: Path = REPO / "results"
    # the protected tree: the slices are read here and everything derived from them stays inside it
    remediated_dir: Path = REPO / "work" / "lotte" / "remediated"
    enc_root: Path = REPO / "work" / "lotte" / "enc"
    gate_work_dir: Path = REPO / "work" / "lotte" / "gate13"
    device: str = "cuda"
    allow_cpu: bool = False                      # tests only; the real encode is a GPU job
    topk: int = 100
    chunk: int = 250_000
    # (cache name, doc_texts) -> (vectors, cache record). None -> stella through teacher.encode_cached
    doc_encoder: object = None
    # (arm record summary, checkpoint bytes, device) -> object with encode_queries(texts).
    # None -> nano10 from the recipe, loaded from the hashed bytes
    load_student: object = None
    # (student key) -> {"repo", "sha256"}: the tokenizer and backbone configuration the student
    # loads by repository name. None -> hashed from the Hugging Face cache
    dependency_identity: object = None
    claim_guard: bool = True                     # a synthetic tree has no protected path to claim
    require_pushed: bool = True                  # the manifest and pin commits must be on a remote branch
    # the document tower's identity, the same pin `access13.Config` carries
    teacher_model_id: str = "NovaSearch/stella_en_400M_v5"
    teacher_revision: str = "ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20"


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")


def sha_obj(obj):
    """`m8src/freeze_lotte.sha`: sha256 of the sorted-key JSON encoding, so the slice hashes here
    are comparable with the pin that module writes."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def slice_hashes(doc_ids, doc_texts, q_ids, q_texts, qrels_sorted):
    """`m8src/freeze_lotte._hash_slice`, reimplemented byte-for-byte (that module claims its guard
    entry at import and cannot be imported beside this one)."""
    q_by_id = dict(zip(q_ids, q_texts))
    return {"doc_ids_sha256": sha_obj(list(doc_ids)), "doc_texts_sha256": sha_obj(list(doc_texts)),
            "query_ids_sha256": sha_obj(sorted(q_ids)),
            "query_texts_sha256": sha_obj([q_by_id[q] for q in sorted(q_ids)]),
            "qrels_sha256": sha_obj(qrels_sorted)}


def code_identity():
    """sha256 of the bytes of the code that RUNS the read — always this source tree. Computed at
    preflight (into the receipt) and again before the record is written; a difference refuses."""
    h = hashlib.sha256()
    for name in CODE_IDENTITY_FILES:
        h.update((REPO / name).read_bytes())
    return h.hexdigest()


def _fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_atomic(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    _fsync_dir(p.parent)
    return p


def _is_sha(x):
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def _git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo)] + list(args), capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def committed_and_pushed(path, what, *, repo=REPO, require_pushed=True):
    """-> the last commit touching `path`. Tracked, unmodified (staged or not), committed, and —
    unless `require_pushed` is off — that commit is on a remote branch. Local git state alone
    licenses nothing irreversible (Sol 2026-09-10, finding 5). Shared with `build13`."""
    p = Path(path)
    rc, _o, err = _git(repo, "ls-files", "--error-unmatch", str(p))
    if rc != 0:
        refuse(f"{what} {p} is not tracked by git; commit and push it first")
    _rc, status, _e = _git(repo, "status", "--porcelain", "--", str(p))
    if status.strip():
        refuse(f"{what} {p} has uncommitted changes; only the committed file binds")
    _rc, commit, _e = _git(repo, "log", "-1", "--format=%H", "--", str(p))
    if not commit:
        refuse(f"{what} {p} has no commit in this repository's history")
    if require_pushed:
        _rc, remotes, _e = _git(repo, "branch", "-r", "--contains", commit)
        if not remotes.strip():
            refuse(f"{what} {p}: commit {commit[:12]} is not on any remote branch; push it first "
                   f"(a commit only this machine holds is not provenance)")
    return commit


# ------------------------------------------------------------------------------- preflight ----

def registration(cfg):
    """-> (the registration, its sha). Refuses a file that is not the R8 registration, or one whose
    upper-bound quantile method is not R17's `inverted_cdf`."""
    p = Path(cfg.registration_path)
    if not p.exists():
        refuse(f"no LoTTE gate registration at {p} (ruling R8)")
    reg = json.loads(p.read_text())
    if reg.get("executed") is not False:
        refuse(f"{p}: `executed` is {reg.get('executed')!r}; the registration is written while "
               f"LoTTE is unread and never flips — the gate RECORD carries the execution")
    surf = reg.get("surface") or {}
    slices = surf.get("slices") or {}
    if len(slices) != 7 or surf.get("n_slices") != 7:
        refuse(f"{p}: the registration names {len(slices)} slices, not the registered seven")
    for key, exp in slices.items():
        if key.count("/") != 1:
            refuse(f"{p}: slice key {key!r} is not <topic>/<split>")
        for f in ("queries_after_remedy", "docs_after_remedy", "qrels_pairs_after_remedy"):
            if not isinstance(exp.get(f), int) or exp[f] <= 0:
                refuse(f"{p}: slice {key} lacks a positive integer {f}")
    if sum(e["queries_after_remedy"] for e in slices.values()) != surf.get("total_queries"):
        refuse(f"{p}: the per-slice query counts do not sum to `total_queries`")
    if not str((reg.get("metric") or {}).get("veto", "")).startswith("nDCG@10"):
        refuse(f"{p}: the veto metric is not nDCG@10 (ruling R8)")
    veto = reg.get("veto") or {}
    boot = veto.get("bootstrap") or {}
    if veto.get("margin") != 0.004:
        refuse(f"{p}: veto.margin is {veto.get('margin')!r}, not the locked 0.004")
    for f, want in (("B", 10000), ("seed", 903), ("paired_within_slice", True)):
        if boot.get(f) != want:
            refuse(f"{p}: veto.bootstrap.{f} is {boot.get(f)!r}, not the locked {want}")
    if boot.get("quantile_method") != QUANTILE_METHOD:
        refuse(f"{p}: veto.bootstrap.quantile_method is {boot.get('quantile_method')!r}; ruling R17 "
               f"pins {QUANTILE_METHOD!r} (order statistic 9,750 of 10,000) and admits nothing else")
    contract = (reg.get("gate_record_contract") or {}).get("path")
    if contract and (Path(cfg.repo) / contract).resolve() != Path(cfg.record_path).resolve():
        refuse(f"{p}: the registration's record path is {contract!r}, not {cfg.record_path}")
    return reg, sha256_file(p)


def registered_recipe(cfg):
    """The recipe every E arm must carry, from `m13/build_config.json`'s registered knobs (which
    `build_lock.validate` holds to the live registry): a record's self-declared student, layers,
    head, objective and mix are compared against these, not merely required to exist (Sol 2026-09-10,
    finding 3)."""
    k = json.loads(Path(cfg.build_config_path).read_text())["data_knobs_equal_to_anchor"]
    return {"student": k["student"], "n_layers": int(k["feature_layers"]), "head": k["head"],
            "objective": k["objective"], "pattern": k["mix"]}


def branch_of(cfg):
    """-> (branch, e1 batch, verdict sha). The E1 verdict, bound to the live registry."""
    v = BL.verdicts(cfg.verdicts_path)
    BL.check_verdict_binding(v, registry_path=cfg.registry_path)
    sel = (v.get("selected") or {}).get("batch")
    if sel is None or str(sel).upper() == BL.PENDING:
        refuse("the E1 verdict is PENDING: results/m10_screen_verdicts.json:selected.batch is not a "
               "batch size. Both E arms run first; read #1 sits after them and before the build "
               "(m10/LOTTE_LOCK.md).")
    try:
        b = int(sel)
    except (TypeError, ValueError):
        refuse(f"`selected.batch` is {sel!r}, not a batch size")
    if b not in (32, 128):
        refuse(f"`selected.batch` is {b}; the registered branches are bs32 and bs128")
    return ("bs128" if b == 128 else "bs32"), b, sha256_file(Path(cfg.verdicts_path))


def arm_record(cfg, arm, registry, want_recipe):
    """-> the summary of one E arm's published record, or refuses. The record must be THIS arm's
    (name, seed, registry binding) and carry the REGISTERED recipe; the checkpoint the gate reads is
    the one `build13._registered_e_checkpoints` will hold it to: the record's cycle-3 sha."""
    p = Path(cfg.arm_records_dir) / f"m10_arm_{slug(arm)}.json"
    if not p.exists():
        refuse(f"no published arm record at {p}: read #1 happens after BOTH 5M E arms finish "
               f"(m10/LOTTE_LOCK.md), and the gate compares their cycle-3 checkpoints")
    rec = json.loads(p.read_text())
    if rec.get("arm") != arm:
        refuse(f"{p}: the record is for arm {rec.get('arm')!r}, not {arm!r}")
    if rec.get("status") != "complete" or rec.get("complete") is not True:
        refuse(f"{p}: status {rec.get('status')!r}, complete {rec.get('complete')!r}; a failed or "
               f"unfinished arm has no final checkpoint to read")
    if rec.get("smoke"):
        refuse(f"{p} is a SMOKE record; the gate reads registered arms only")
    ck, sha = rec.get("final_checkpoint"), rec.get("final_checkpoint_sha256")
    c3 = ((rec.get("checkpoints") or {}).get("cycle3") or {}).get("sha256")
    if not ck or not _is_sha(sha) or c3 != sha:
        refuse(f"{p}: final_checkpoint {ck!r} / sha {str(sha)[:12]!r} / cycle3 sha "
               f"{str(c3)[:12]!r} do not name one cycle-3 checkpoint")
    recipe = rec.get("recipe") or {}
    missing = [f for f in RECIPE_FIELDS if f not in recipe]
    if missing:
        refuse(f"{p}: recipe lacks {missing}; the student cannot be rebuilt or checked from the record")
    entry = (registry.get("arms") or {}).get(arm) or {}
    if int(recipe["dose_examples"]) != SCREEN_DOSE or \
            int(entry.get("dose_examples", -1)) != SCREEN_DOSE:
        refuse(f"{p}: dose {recipe['dose_examples']} / registry {entry.get('dose_examples')} is not "
               f"the registered screen dose {SCREEN_DOSE} (m10/LOTTE_LOCK.md)")
    if int(recipe["batch"]) != int(entry.get("batch")):
        refuse(f"{p}: batch {recipe['batch']} is not the registry's {entry.get('batch')} for {arm}")
    bad = {f: (recipe[f], want_recipe[f]) for f in want_recipe
           if (int(recipe[f]) if f == "n_layers" else recipe[f]) != want_recipe[f]}
    if bad:
        refuse(f"{p}: recipe differs from the registered knobs on {bad} (record, registered); a "
               f"checkpoint of another recipe is not this arm's")
    want_seed = int(entry["seed"]) if entry.get("seed") is not None else 0     # seed_rule: seed 0
    if int(rec.get("seed", -1)) != want_seed:
        refuse(f"{p}: seed {rec.get('seed')!r} is not the registered {want_seed} for {arm}")
    live_reg = sha256_file(Path(cfg.registry_path))
    if rec.get("registry_sha256") != live_reg:
        refuse(f"{p}: the arm trained under registry sha {str(rec.get('registry_sha256'))[:12]} but "
               f"the live m10/screen_registry.json is {live_reg[:12]}; a record about one registry "
               f"cannot feed a gate under another")
    return {"arm": arm, "record": str(p), "record_sha256": sha256_file(p), "checkpoint": ck,
            "sha256": sha, "recipe": {k: recipe[k] for k in RECIPE_FIELDS},
            "seed": want_seed, "params": rec.get("params"), "device": rec.get("device"),
            "git_head": rec.get("git_head")}


def checkpoint_bytes(cfg, ar):
    """-> the checkpoint's bytes, read ONCE and hashed; the student is deserialised from this very
    buffer, so a file replaced between hashing and loading cannot be scored."""
    p = Path(cfg.repo) / ar["checkpoint"]
    if not p.exists():
        refuse(f"{ar['arm']}: checkpoint {p} is not on this machine; the gate reads the exact "
               f"bytes the arm record names")
    data = p.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != ar["sha256"]:
        refuse(f"{ar['arm']}: {p} hashes {got[:12]}, the record says {ar['sha256'][:12]}; "
               f"refusing to read a checkpoint that is not the published one")
    return data


def dependency_identity(cfg, student_key):
    """-> {"repo", "sha256"}: what `nano10.Nano10` loads BY REPOSITORY NAME beside the checkpoint
    — the tokenizer and the backbone configuration. Recorded in the manifest at write time and
    compared at load time, so a changed dependency cannot change the scoring under an unchanged
    checkpoint sha (Sol 2026-09-10, finding 4). `nano10` itself pins no revision; that is M10 code."""
    if cfg.dependency_identity is not None:
        return cfg.dependency_identity(student_key)
    import nano10 as N
    from transformers import AutoConfig, AutoTokenizer
    repo = N.REPOS[student_key]
    tok = AutoTokenizer.from_pretrained(repo)
    conf = AutoConfig.from_pretrained(repo)
    h = hashlib.sha256()
    h.update(repo.encode())
    h.update(tok.backend_tokenizer.to_str().encode())
    h.update(conf.to_json_string().encode())
    return {"repo": repo, "sha256": h.hexdigest()}


def manifest_of(cfg, arms, branch, vsha):
    """-> (manifest, sha, commit). `m13/LOTTE_GATE_MANIFEST.json` is the lock's second manifest
    commit. It must be committed and pushed, and agree field by field with the live arm records,
    the branch, the E1 verdict sha and the registry."""
    p = Path(cfg.manifest_path)
    if not p.exists():
        refuse(f"no checkpoint manifest at {p}. After BOTH E records are pushed, run "
               f"`lotte_gate13.py --write-manifest`, then commit and push the file (the second "
               f"manifest commit, m10/LOTTE_LOCK.md); the read refuses without it.")
    m = json.loads(p.read_text())
    cand_arm = "E-bs128" if branch == "bs128" else "E-bs32"
    problems = []
    if m.get("branch") != branch:
        problems.append(f"branch {m.get('branch')!r} vs E1 {branch!r}")
    if m.get("e1_verdict_sha256") != vsha:
        problems.append("e1_verdict_sha256 differs from the live verdict")
    if m.get("registry_sha256") != sha256_file(Path(cfg.registry_path)):
        problems.append("registry_sha256 differs from the live registry")
    for role, arm in (("candidate", cand_arm), ("comparator", "E-bs32" if branch == "bs128" else None)):
        got = m.get(role)
        if arm is None:
            if got is not None:
                problems.append("bs32 branch with a comparator")
            continue
        if not isinstance(got, dict):
            problems.append(f"{role} missing")
            continue
        for f in ("arm", "sha256", "record_sha256", "checkpoint", "recipe", "seed"):
            if got.get(f) != arms[arm][f]:
                problems.append(f"{role}.{f} differs from the live {arm} record")
        if not _is_sha((got.get("dependencies") or {}).get("sha256")):
            problems.append(f"{role}.dependencies.sha256 missing")
    if problems:
        refuse(f"{p} does not describe the live inputs: " + "; ".join(problems) +
               ". A manifest is written once from the published records and committed; if the "
               "records legitimately changed, that is a new second manifest commit, not a silent read.")
    commit = committed_and_pushed(p, "the checkpoint manifest", repo=cfg.repo,
                                  require_pushed=cfg.require_pushed)
    return m, sha256_file(p), commit


def pin_of(cfg, reg):
    """-> (pin, sha, commit). `results/m8_lotte_pin.json` (m8src/freeze_lotte.py pin) must cover
    every registered slice with its three counts and five hashes, committed and pushed (R18)."""
    p = Path(cfg.pin_path)
    if not p.exists():
        refuse(f"no LoTTE pin at {p}. Ruling R18: run `.venv/bin/python m8src/freeze_lotte.py pin` "
               f"immediately before the gate, commit and push the pin; the gate compares every "
               f"slice's five hashes and counts against it, and counts alone authenticate nothing.")
    pin = json.loads(p.read_text())
    slices = pin.get("slices") or {}
    missing = [k for k in reg["surface"]["slices"] if k not in slices]
    if missing:
        refuse(f"{p} does not pin {missing}; every registered slice must be pinned")
    for k in reg["surface"]["slices"]:
        s = slices[k]
        for f in HASH_FIELDS:
            if not _is_sha((s.get("hashes") or {}).get(f)):
                refuse(f"{p}: slice {k} lacks a sha256 for {f}")
        for f in ("n_docs", "n_queries", "n_qrels_pairs"):
            if not isinstance(s.get(f), int):
                refuse(f"{p}: slice {k} lacks the integer count {f}")
    commit = committed_and_pushed(p, "the LoTTE pin", repo=cfg.repo, require_pushed=cfg.require_pushed)
    return pin, sha256_file(p), commit


def identity_of(plan):
    return {k: plan[k] for k in IDENTITY_FIELDS}


def preflight(cfg, *, recover=False, verbose=True):
    """Everything that must be true before a LoTTE path is opened. Opens none (the receipt's
    EXISTENCE is a stat, not an open; its contents are compared after the claim)."""
    rec_p = Path(cfg.record_path)
    if rec_p.exists():
        refuse(f"{rec_p} already exists. Read #1 is ONE access and there is no third read "
               f"(m10/LOTTE_LOCK.md); a re-run is a protocol change, not a retry. Move the record "
               f"aside deliberately if it is not the executed gate.")
    receipt_p = Path(cfg.gate_work_dir) / "receipt.json"
    if receipt_p.exists() and not recover:
        refuse(f"{receipt_p} exists: a read has already STARTED. If it crashed, pass --recover to "
               f"complete it from its persisted slices under the identical identity; nothing else "
               f"may open the surface again.")
    if recover and not receipt_p.exists():
        refuse(f"--recover: no receipt at {receipt_p}; there is no started read to complete")
    reg, reg_sha = registration(cfg)
    want_recipe = registered_recipe(cfg)
    branch, batch, vsha = branch_of(cfg)
    registry = json.loads(Path(cfg.registry_path).read_text())
    arms = {a: arm_record(cfg, a, registry, want_recipe) for a in E_ARMS}
    cand = arms["E-bs128" if branch == "bs128" else "E-bs32"]
    comp = arms["E-bs32"] if branch == "bs128" else None
    for ar in (cand, comp):
        if ar is not None:
            checkpoint_bytes(cfg, ar)            # exists and hashes now; the bytes are re-read at load
    man, man_sha, man_commit = manifest_of(cfg, arms, branch, vsha)
    pin, pin_sha, pin_commit = pin_of(cfg, reg)
    import torch
    if cfg.device == "cuda" and not torch.cuda.is_available():
        refuse("--device cuda but no CUDA device is visible; a CUDA torch installation is not a GPU")
    if cfg.device != "cuda" and not cfg.allow_cpu:
        refuse(f"device {cfg.device!r}: the stella encode of ~2.7M passages is a GPU job")
    teacher = None
    if cfg.doc_encoder is None:
        os.environ.setdefault("M7_ENCODER", "stella-400M-v5")
        import teacher as T
        if (T.TEACHER, T.TEACHER_REV) != (cfg.teacher_model_id, cfg.teacher_revision):
            refuse(f"the active teacher is {T.TEACHER}@{T.TEACHER_REV[:12]}, not the pinned "
                   f"{cfg.teacher_model_id}@{cfg.teacher_revision[:12]}; a different document tower "
                   f"is a different index")
        teacher = {"model_id": T.TEACHER, "revision": T.TEACHER_REV, "encode_dtype": "fp16",
                   "doc_prefix": "", "max_length": 512}
    veto = reg["veto"]
    order = sorted(reg["surface"]["slices"],
                   key=lambda k: reg["surface"]["slices"][k]["docs_after_remedy"])
    plan = {"branch": branch, "e1_batch": batch, "e1_verdict_sha256": vsha,
            "candidate": cand, "comparator": comp,
            "candidate_sha256": cand["sha256"],
            "comparator_sha256": comp["sha256"] if comp else None,
            "veto_runs": comp is not None,
            "registration_sha256": reg_sha, "registration": reg,
            "manifest": man, "manifest_sha256": man_sha, "manifest_commit": man_commit,
            "pin": pin, "pin_sha256": pin_sha, "pin_commit": pin_commit,
            "code_identity": code_identity(),
            "slice_order": order, "veto": veto, "device": cfg.device, "teacher": teacher,
            "record_path": str(rec_p), "receipt_path": str(receipt_p), "recover": recover}
    if verbose:
        print(f"LoTTE read #1 preflight: branch {branch} (E1 batch {batch}); veto "
              f"{'RUNS' if comp else 'SKIPPED and forfeited (identical recipe and action)'}"
              + ("; RECOVER from a started read" if recover else ""))
        print(f"  candidate  {cand['arm']}: {cand['checkpoint']} {cand['sha256'][:12]}")
        if comp:
            print(f"  comparator {comp['arm']}: {comp['checkpoint']} {comp['sha256'][:12]}")
        print(f"  manifest {man_sha[:12]} @ {man_commit[:12]}; pin {pin_sha[:12]} @ {pin_commit[:12]}; "
              f"registration {reg_sha[:12]}; code {plan['code_identity'][:12]}")
        print(f"  slices (smallest first): {order}")
        print(f"  veto: margin {veto['margin']}, B {veto['bootstrap']['B']}, seed "
              f"{veto['bootstrap']['seed']}, {veto['bootstrap']['interval']}, {QUANTILE_METHOD}")
        print(f"  device {cfg.device}; teacher {teacher}; no LoTTE path opened", flush=True)
    return plan


def write_manifest(cfg, verbose=True):
    """`--write-manifest`: the lock's second manifest commit, materialised from the published arm
    records and the E1 verdict, after both E arms finished and before the read. Opens no LoTTE
    path, claims nothing. The operator commits and pushes the file; the gate then binds to it."""
    p = Path(cfg.manifest_path)
    if p.exists():
        refuse(f"{p} already exists; a manifest is written once. If the E records legitimately "
               f"changed, move it aside deliberately and record why in m13/EXECUTION.md.")
    reg, reg_sha = registration(cfg)
    want_recipe = registered_recipe(cfg)
    branch, batch, vsha = branch_of(cfg)
    registry = json.loads(Path(cfg.registry_path).read_text())
    arms = {a: arm_record(cfg, a, registry, want_recipe) for a in E_ARMS}
    for ar in arms.values():
        checkpoint_bytes(cfg, ar)
        ar["dependencies"] = dependency_identity(cfg, ar["recipe"]["student"])
    cand = arms["E-bs128" if branch == "bs128" else "E-bs32"]
    comp = arms["E-bs32"] if branch == "bs128" else None
    fields = ("arm", "checkpoint", "sha256", "record_sha256", "recipe", "seed", "git_head",
              "dependencies")
    man = {"_what": "LoTTE read #1 checkpoint manifest — m10/LOTTE_LOCK.md's second manifest commit, "
                    "written by m13src/lotte_gate13.py --write-manifest from the published E arm "
                    "records after both finished and before the build. Commit and push it; the "
                    "gate refuses to read without the pushed file and re-checks every field.",
           "written_at": utcnow(), "branch": branch, "e1_batch": batch,
           "e1_verdict_sha256": vsha, "registry_sha256": sha256_file(Path(cfg.registry_path)),
           "registration_sha256": reg_sha,
           "candidate": {k: cand[k] for k in fields},
           "comparator": None if comp is None else {k: comp[k] for k in fields},
           "git_head": R.git_head()}
    write_atomic(p, man)
    if verbose:
        print(f"wrote {p} (branch {branch}; candidate {cand['arm']} {cand['sha256'][:12]}"
              + (f"; comparator E-bs32 {comp['sha256'][:12]}" if comp else "")
              + "). Now: git add, commit and PUSH it — the second manifest commit.", flush=True)
    return man


# ------------------------------------------------------------------------------ the guard ----

def claim_lotte(note="LoTTE read #1 (m13/LOTTE_GATE_REGISTRATION.json)"):
    """Install `paths_guard` and claim THIS module's entry. `claim()` verifies the caller is
    physically `m13src/lotte_gate13.py`, so this cannot be borrowed by another file."""
    import paths_guard
    paths_guard.install()
    return paths_guard.claim(GUARD_ENTRY, note=note)


# -------------------------------------------------------------------------------- slices ----

def _read_tsv(path):
    ids, texts = [], []
    with open(path) as fh:
        for line in fh:
            t = line.rstrip("\n").split("\t", 1)
            if len(t) == 2:
                ids.append(t[0])
                texts.append(t[1])
    return ids, texts


def read_slice(cfg, key, expected, pinned):
    """-> the remediated slice, checked against the registration's counts AND the pin's counts and
    five hashes. Refuses on any mismatch, duplicate qrel row or duplicate positive."""
    topic, split = key.split("/")
    d = Path(cfg.remediated_dir) / topic / split
    for name in SLICE_FILES:
        if not (d / name).exists():
            refuse(f"{key}: {d / name} is missing; refusing to skip a registered slice")
    doc_ids, doc_texts = _read_tsv(d / "collection.tsv")
    q_ids, q_texts = _read_tsv(d / "questions.forum.tsv")
    qrels, problems = {}, []
    with open(d / "qas.forum.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = str(r["qid"])
            pids = [str(p) for p in r["answer_pids"]]
            if qid in qrels:
                problems.append("duplicate qrel rows for one query")
            if len(set(pids)) != len(pids):
                problems.append("duplicate positives within one qrel row")
            qrels[qid] = sorted(pids)
    n_pairs = sum(len(v) for v in qrels.values())
    if len(doc_ids) != expected["docs_after_remedy"]:
        problems.append(f"{len(doc_ids)} documents, registered {expected['docs_after_remedy']}")
    if len(q_ids) != expected["queries_after_remedy"]:
        problems.append(f"{len(q_ids)} queries, registered {expected['queries_after_remedy']}")
    if n_pairs != expected["qrels_pairs_after_remedy"]:
        problems.append(f"{n_pairs} qrel pairs, registered {expected['qrels_pairs_after_remedy']}")
    if len(set(doc_ids)) != len(doc_ids):
        problems.append("duplicate document ids")
    if len(set(q_ids)) != len(q_ids):
        problems.append("duplicate query ids")
    if set(qrels) != set(q_ids):
        problems.append("qas.forum.jsonl and questions.forum.tsv name different query sets")
    doc_set = set(doc_ids)
    if any(p not in doc_set for v in qrels.values() for p in v):
        problems.append("a qrel names a document outside the collection")
    if any(len(v) == 0 for v in qrels.values()):
        problems.append("a query has no positive")
    if problems:
        refuse(f"{key} under {d} is not the registered slice: " + "; ".join(sorted(set(problems))))
    hashes = slice_hashes(doc_ids, doc_texts, q_ids, q_texts, qrels)
    bad = [f for f in HASH_FIELDS if pinned["hashes"].get(f) != hashes[f]]
    for f, v in (("n_docs", len(doc_ids)), ("n_queries", len(q_ids)), ("n_qrels_pairs", n_pairs)):
        if int(pinned[f]) != v:
            bad.append(f)
    if bad:
        refuse(f"{key}: the slice on disk does not match results/m8_lotte_pin.json on {sorted(bad)}; "
               f"the bytes changed since the pin, and a changed slice is not the registered one")
    return {"key": key, "doc_ids": doc_ids, "doc_texts": doc_texts, "q_ids": q_ids,
            "q_texts": q_texts, "qrels": {q: {p: 1 for p in v} for q, v in qrels.items()},
            "n_docs": len(doc_ids), "n_queries": len(q_ids), "n_qrels_pairs": n_pairs,
            "hashes": hashes, "read_relpath": str(d)}


def encode_docs(cfg, key, doc_texts, verbose=True):
    """-> (document vectors, cache record). stella once per slice, shard-resumable, cached INSIDE
    the protected tree (`teacher.ENC` is rebound to `cfg.enc_root`), prefix "" like the shared
    index's own caches, stored fp16 like the DEV-6 caches. `verify=True`: a shard or combined file
    reused from an earlier attempt is re-hashed, and an unrecorded one refuses."""
    topic, split = key.split("/")
    name = f"lotte-{topic}-{split}-docs"
    t0 = time.time()
    if cfg.doc_encoder is not None:
        vecs, rec = cfg.doc_encoder(name, doc_texts)
    else:
        import torch
        import teacher as T
        T.ENC = Path(cfg.enc_root)
        vecs = T.encode_cached(name, doc_texts, prefix="", dtype=torch.float16, verbose=verbose,
                               device=cfg.device, verify=True)
        rec = dict(T.PROVENANCE.get(name) or {})
        rec.pop("shard_sha256", None)             # the combined sha carries the identity
    sec = time.time() - t0
    rec.update({"name": name, "n_rows": len(doc_texts), "dim": int(vecs.shape[1]),
                "seconds": round(sec, 1), "rows_per_s": round(len(doc_texts) / max(sec, 1e-9), 1)})
    if verbose:
        print(f"  [{key}] {len(doc_texts):,} passages -> {vecs.shape[1]}d in {sec:.0f}s "
              f"({rec['rows_per_s']:.0f}/s incl. cache reads)", flush=True)
    return vecs, rec


# ------------------------------------------------------------------------------- scoring ----

def load_student(cfg, ar, data, want_dependencies):
    """The nano student the arm record describes, deserialised from the hashed BYTES (`data`),
    never from a second read of the file, with the dependency identity re-derived and held to the
    manifest's. `trainer10.save` writes {"model": state_dict, ...}."""
    dep = dependency_identity(cfg, ar["recipe"]["student"])
    if dep.get("sha256") != (want_dependencies or {}).get("sha256"):
        refuse(f"{ar['arm']}: the tokenizer/backbone configuration loaded now ({dep['sha256'][:12]}) "
               f"is not the one the manifest recorded ({str((want_dependencies or {}).get('sha256'))[:12]}); "
               f"a changed dependency changes the scoring under an unchanged checkpoint")
    if cfg.load_student is not None:
        return cfg.load_student(ar, data, cfg.device), dep
    import torch
    import nano10 as N
    r = ar["recipe"]
    model = N.Nano10(r["student"], n_layers=int(r["n_layers"]), head=r["head"])
    blob = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
    sd = blob.get("model", blob) if isinstance(blob, dict) else blob
    model.load_state_dict(sd)
    if not model.under_cap():
        refuse(f"{ar['arm']}: {model.n_params():,} parameters, over the registered 35M cap")
    if ar.get("params") is not None and int(ar["params"]) != model.n_params():
        refuse(f"{ar['arm']}: the record says {ar['params']:,} parameters, the rebuilt student "
               f"has {model.n_params():,}; the recipe in the record does not describe it")
    model.eval()
    return model.to(cfg.device), dep


def success_at_k(scores, rels, k):
    top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return 1.0 if any(pid in rels for pid, _s in top) else 0.0


def score_slice(cfg, student, sl, dv):
    """-> {"ndcg10": {qid: v}, "success5": {qid: v}} for one student on one slice, exact search
    over the whole remediated collection. Query prefix "" (prompt policy (b), what the student
    trained on; `run_arm.dev6` scores the same way)."""
    from evalkit import per_query_ndcg, topk_ids_scores
    q_texts, q_ids = list(sl["q_texts"]), list(sl["q_ids"])
    qv = student.encode_queries(q_texts)
    if qv.shape != (len(q_ids), dv.shape[1]):
        refuse(f"{sl['key']}: query vectors {qv.shape} vs document vectors {dv.shape}")
    ns_ids = [QNS + q for q in q_ids]
    run = topk_ids_scores(qv, dv, sl["doc_ids"], k=cfg.topk, chunk=cfg.chunk, device=cfg.device,
                          qids=ns_ids)
    qrels = {QNS + q: rels for q, rels in sl["qrels"].items()}
    ndcg = per_query_ndcg(run, qrels)
    if set(ndcg) != set(ns_ids):
        # no identifier leaves the protected tree, not even in a refusal
        refuse(f"{sl['key']}: {len(set(ns_ids) - set(ndcg))} of {len(ns_ids)} queries were not "
               f"scored; pytrec_eval omits a qid without qrels (CODEMAP pitfall 4)")
    return {"ndcg10": {q: float(ndcg[QNS + q]) for q in q_ids},
            "success5": {q: success_at_k(run[QNS + q], sl["qrels"][q], 5) for q in q_ids}}


def slice_means(per_slice):
    """{slice: {qid: v}} -> ({slice: mean}, macro at equal weight over slices, never pooled)."""
    means = {k: float(np.mean(list(v.values()))) for k, v in per_slice.items()}
    return means, float(np.mean(list(means.values())))


def paired_bootstrap(cand, comp, B, seed, quantile=0.975, method=QUANTILE_METHOD):
    """The registered veto interval: per-query (candidate - comparator) deltas on identical qids,
    resampled WITHIN each slice, the macro of the slice means per draw, and the one-sided upper
    bound as the R17 quantile. Slices enter the RNG stream in sorted order."""
    keys = sorted(cand)
    if set(comp) != set(keys):
        raise ValueError(f"slices differ: {sorted(cand)} vs {sorted(comp)}")
    diffs, plan_h = {}, hashlib.sha256(f"B={B};seed={seed}".encode())
    rng = np.random.default_rng(int(seed))
    draws = np.zeros(int(B), dtype=np.float64)
    for k in keys:
        if set(cand[k]) != set(comp[k]):
            raise ValueError(f"{k}: candidate and comparator scored different qids")
        qids = sorted(cand[k])
        if not qids:
            raise ValueError(f"{k}: no queries")
        d = np.asarray([cand[k][q] - comp[k][q] for q in qids], dtype=np.float64)
        idx = rng.integers(0, d.size, size=(int(B), d.size), dtype=np.int64)
        plan_h.update(k.encode())
        plan_h.update(idx.tobytes())
        draws += d[idx].mean(axis=1)
        diffs[k] = d
    draws /= len(keys)
    point = float(np.mean([float(d.mean()) for d in diffs.values()]))
    return {"delta_macro_raw": point,
            "upper_q975_raw": float(np.quantile(draws, quantile, method=method)),
            "quantile": quantile, "quantile_method": method, "B": int(B), "seed": int(seed),
            "paired_within_slice": True,
            "per_slice_delta_raw": {k: float(d.mean()) for k, d in diffs.items()},
            "n_by_slice": {k: int(d.size) for k, d in diffs.items()},
            "plan_sha256": plan_h.hexdigest(),
            "draws_sha256": hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest(),
            "_gate_note": "upper_q975_raw and delta_macro_raw are the only fields the veto reads"}


def decide(plan, boot, margin):
    """`m10/LOTTE_LOCK.md`: vetoed iff the selected macro is worse than the comparator's by more
    than the margin AND the one-sided 97.5% upper bound on (selected - comparator) is below
    -margin. bs32: skipped and forfeited. `build13.check_gate` recomputes this from the record."""
    if plan["comparator"] is None:
        return "skipped", None
    fires = boot["delta_macro_raw"] < -margin and boot["upper_q975_raw"] < -margin
    return ("veto" if fires else "no_veto"), bool(fires)


# ----------------------------------------------------------------------------------- run ----

def _attempt(cfg, line):
    d = Path(cfg.gate_work_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "attempts.jsonl", "a") as fh:
        fh.write(json.dumps(line, default=str) + "\n")
    with open(d / "attempts.jsonl") as fh:
        return sum(1 for _ in fh)


def _environment(device):
    import torch
    env = {"host": platform.node(), "python": platform.python_version(),
           "torch": torch.__version__, "cuda": torch.version.cuda, "device": device}
    if device == "cuda" and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def acquire_lock(cfg):
    """A non-blocking exclusive `flock` under the gate's work directory, held (the fd stays open)
    for the whole attempt: a second process — a plain run or a `--recover` while the first is still
    reading — cannot open the surface concurrently (Sol 2026-09-10, finding 1)."""
    d = Path(cfg.gate_work_dir)
    d.mkdir(parents=True, exist_ok=True)
    fd = os.open(d / "lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        refuse(f"another gate process holds {d / 'lock'}; one process reads at a time, and a "
               f"recovery may start only after the process that started the read has exited")
    return fd


def _create_receipt(cfg, plan):
    """The receipt: O_EXCL, then the file AND its directory fsynced, so a crash after the first
    protected open cannot lose the entry that makes a fresh run refuse."""
    p = Path(cfg.gate_work_dir) / "receipt.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {"_what": "LoTTE read #1 receipt: the read STARTED with these inputs; a plain re-run "
                     "refuses while this exists, --recover requires the identical identity",
            "identity": identity_of(plan), "started_at": utcnow(), "git_head": R.git_head(),
            "host": platform.node()}
    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as fh:
        json.dump(body, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_dir(p.parent)
    return body


def _load_receipt(cfg, plan):
    p = Path(cfg.gate_work_dir) / "receipt.json"
    body = json.loads(p.read_text())
    want, got = identity_of(plan), body.get("identity") or {}
    diff = sorted(k for k in IDENTITY_FIELDS if want.get(k) != got.get(k))
    if diff:
        refuse(f"--recover: the started read's identity differs from today's on {diff}; a recovery "
               f"completes THE SAME read or nothing. Restore those inputs (checkout the receipt's "
               f"code identity, the same records, device and verdict) or stop.")
    return body


def _slice_path(cfg, key):
    topic, split = key.split("/")
    return Path(cfg.gate_work_dir) / f"slice-{topic}-{split}.json"


def _journal(cfg, key, sha):
    with open(Path(cfg.gate_work_dir) / "slices.jsonl", "a") as fh:
        fh.write(json.dumps({"key": key, "sha256": sha}) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _journaled(cfg):
    p = Path(cfg.gate_work_dir) / "slices.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["key"], set()).add(r["sha256"])
    return out


def _valid_scores(scores, q_ids):
    return set(scores) == set(q_ids) and all(
        isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0 for v in scores.values())


def _persisted_slice(cfg, key, plan, roles, journal):
    """A persisted slice is accepted for recovery only if its digest was journaled when it was
    written, its identity, key and roles are this read's, its surface hashes are the pin's, and
    its per-query rows are well-formed (finding 8). Anything else refuses: a slice file is
    evidence, not a suggestion."""
    p = _slice_path(cfg, key)
    if not p.exists():
        return None
    if sha256_file(p) not in journal.get(key, set()):
        refuse(f"{p} is not the file this read journaled when it wrote it; a slice file that was "
               f"edited or replaced cannot complete the read")
    s = json.loads(p.read_text())
    pinned = plan["pin"]["slices"][key]
    surf = s.get("surface") or {}
    ok = (s.get("identity") == identity_of(plan) and s.get("key") == key
          and set(s.get("per_role") or {}) == set(roles)
          and surf.get("hashes") == pinned["hashes"]
          and surf.get("n_queries") == pinned["n_queries"])
    if ok:
        q_ids = set(surf_q for surf_q in (s["per_role"][roles[0]]["ndcg10"]))
        ok = len(q_ids) == pinned["n_queries"] and all(
            _valid_scores(s["per_role"][r][m], q_ids) for r in roles for m in ("ndcg10", "success5"))
    if not ok:
        refuse(f"{p} was written under a different identity, role set or slice, or its rows are "
               f"malformed; it cannot complete this read. Inspect it before deciding anything.")
    return s


def run(cfg=None, *, preflight_only=False, recover=False, verbose=True):
    """Read #1, end to end. -> the gate record (or the preflight plan under `preflight_only`)."""
    cfg = cfg or Config()
    plan = preflight(cfg, recover=recover, verbose=verbose)
    if preflight_only:
        return plan
    if cfg.claim_guard:
        claim_lotte()
    lock_fd = acquire_lock(cfg)
    try:
        return _run_locked(cfg, plan, recover=recover, verbose=verbose)
    finally:
        os.close(lock_fd)


def _run_locked(cfg, plan, *, recover, verbose):
    t_start = time.time()
    read_at = utcnow()
    receipt = _load_receipt(cfg, plan) if recover else _create_receipt(cfg, plan)
    n_attempts = _attempt(cfg, {"started_at": read_at, "recover": recover, "branch": plan["branch"],
                                "git_head": R.git_head(), "candidate": plan["candidate_sha256"]})
    roles = ["candidate"] + (["comparator"] if plan["comparator"] is not None else [])
    students, deps = {}, {}
    for role in roles:
        students[role], deps[role] = load_student(
            cfg, plan[role], checkpoint_bytes(cfg, plan[role]),
            (plan["manifest"].get(role) or {}).get("dependencies"))
    reg = plan["registration"]
    per = {role: {"ndcg10": {}, "success5": {}} for role in roles}
    surface, caches, recovered = {}, {}, []
    journal = _journaled(cfg) if recover else {}
    for key in plan["slice_order"]:
        done = _persisted_slice(cfg, key, plan, roles, journal) if recover else None
        if done is not None:
            for role in roles:
                per[role]["ndcg10"][key] = done["per_role"][role]["ndcg10"]
                per[role]["success5"][key] = done["per_role"][role]["success5"]
            surface[key], caches[key] = done["surface"], done["cache"]
            recovered.append(key)
            if verbose:
                print(f"  [{key}] recovered from {_slice_path(cfg, key).name}; not re-read", flush=True)
            continue
        sl = read_slice(cfg, key, reg["surface"]["slices"][key], plan["pin"]["slices"][key])
        dv, cache = encode_docs(cfg, key, sl["doc_texts"], verbose=verbose)
        scored = {}
        for role, st in students.items():
            s = score_slice(cfg, st, sl, dv)
            scored[role] = s
            per[role]["ndcg10"][key] = s["ndcg10"]
            per[role]["success5"][key] = s["success5"]
        del dv
        surface[key] = {k: sl[k] for k in ("n_docs", "n_queries", "n_qrels_pairs", "hashes",
                                            "read_relpath")}
        caches[key] = cache
        sp = write_atomic(_slice_path(cfg, key), {
            "_what": "LoTTE read #1: one slice's outputs, persisted as it completed so a crashed read "
                     "can be completed without re-reading this slice; per-query rows stay here",
            "identity": identity_of(plan), "key": key, "surface": surface[key], "cache": cache,
            "per_role": scored, "scored_at": utcnow()})
        _journal(cfg, key, sha256_file(sp))
        if verbose:
            row = " ".join(f"{role} nDCG@10 {np.mean(list(per[role]['ndcg10'][key].values())):.4f}"
                           for role in roles)
            print(f"  [{key}] {row}", flush=True)
    rows = {}
    for role in roles:
        n_means, n_macro = slice_means(per[role]["ndcg10"])
        s_means, s_macro = slice_means(per[role]["success5"])
        rows[role] = {"arm": plan[role]["arm"], "checkpoint_sha256": plan[role]["sha256"],
                      "dependencies": deps[role],
                      "ndcg10": {"per_slice": n_means, "macro": n_macro},
                      "success5": {"per_slice": s_means, "macro": s_macro}}
    veto = plan["veto"]
    boot = None
    if "comparator" in students:
        boot = paired_bootstrap(per["candidate"]["ndcg10"], per["comparator"]["ndcg10"],
                                veto["bootstrap"]["B"], veto["bootstrap"]["seed"])
    decision, fired = decide(plan, boot, veto["margin"])
    code_now = code_identity()
    if code_now != plan["code_identity"]:
        refuse(f"the code identity changed during the read ({plan['code_identity'][:12]} -> "
               f"{code_now[:12]}); no record is written. Restore the code and --recover.")
    pq_path = Path(cfg.gate_work_dir) / f"perquery-{plan['branch']}.json"
    write_atomic(pq_path, {"_what": "LoTTE read #1 per-query nDCG@10 and Success@5 by slice; "
                                    "derived from LoTTE, kept under work/lotte",
                           "branch": plan["branch"], "read_at": read_at, "per_role": per})
    record = {
        "_what": "LoTTE read #1 — the gate RECORD m13src/build13.check_gate requires, written by the "
                 "executor that performed the read (m13/LOTTE_GATE_REGISTRATION.json "
                 "gate_record_contract). No query text, label or per-query row is in this file.",
        "executed": True,
        "branch": plan["branch"],
        "decision": decision,
        "veto_fired": fired,
        "e1_batch": plan["e1_batch"],
        "e1_verdict_sha256": plan["e1_verdict_sha256"],
        "candidate_sha256": plan["candidate_sha256"],
        "comparator_sha256": plan["comparator_sha256"],
        "candidate": plan["candidate"],
        "comparator": plan["comparator"],
        "manifest_path": str(Path(cfg.manifest_path)),
        "manifest_sha256": plan["manifest_sha256"],
        "manifest_commit": plan["manifest_commit"],
        "pin_path": str(Path(cfg.pin_path)),
        "pin_sha256": plan["pin_sha256"],
        "pin_commit": plan["pin_commit"],
        "macro_ndcg10": {role: rows[role]["ndcg10"] for role in rows},
        "success_at_5": {role: rows[role]["success5"] for role in rows},
        "dependencies": {role: rows[role]["dependencies"] for role in rows},
        "delta_candidate_minus_comparator": boot,
        "bootstrap_upper_bound": None if boot is None else boot["upper_q975_raw"],
        "veto": {**veto, "metric": reg["metric"]["veto"], "fired": fired,
                 "consequence": ("the comparator's recipe (bs32) is what the 200M build trains"
                                 if fired else
                                 "none: the E1 selection stands" if decision == "no_veto" else
                                 "none: skipped and forfeited (identical recipe and action); the "
                                 "observational row above is a 5M-dose number and selects nothing")},
        "surface": {"name": reg["surface"]["name"], "n_slices": len(surface),
                    "total_queries": sum(s["n_queries"] for s in surface.values()),
                    "macro": reg["surface"]["macro"], "slices": surface,
                    "doc_caches": caches, "read_order": plan["slice_order"]},
        "teacher": plan["teacher"],
        "registration_path": str(Path(cfg.registration_path)),
        "registration_sha256": plan["registration_sha256"],
        "perquery": {"path": str(pq_path), "sha256": sha256_file(pq_path)},
        "receipt": {"started_at": receipt["started_at"], "git_head_at_start": receipt["git_head"],
                    "recovered": recover, "recovered_slices": recovered,
                    "attempts_including_this": n_attempts},
        "read_at": read_at,
        "seconds": round(time.time() - t_start, 1),
        "environment": _environment(cfg.device),
        "code_identity": code_now,
        "code_identity_files": list(CODE_IDENTITY_FILES),
        "git_head": R.git_head(),
        "firewall": reg.get("firewall"),
    }
    rec_p = Path(cfg.record_path)
    if rec_p.exists():
        refuse(f"{rec_p} appeared during the read; not overwriting it")
    write_atomic(rec_p, record)
    _attempt(cfg, {"finished_at": utcnow(), "decision": decision, "record": str(rec_p)})
    if verbose:
        print(f"LoTTE read #1 {plan['branch']}: {decision.upper()}"
              + (f" (delta {boot['delta_macro_raw']:+.4f}, upper {boot['upper_q975_raw']:+.4f}, "
                 f"margin {veto['margin']})" if boot else "")
              + f"; candidate macro nDCG@10 {rows['candidate']['ndcg10']['macro']:.4f}, "
                f"Success@5 {rows['candidate']['success5']['macro']:.4f}"
              + (f"; recovered {len(recovered)} slice(s)" if recovered else "")
              + f"; wrote {rec_p}. Commit and push it.", flush=True)
    return record


def build_argparser():
    ap = argparse.ArgumentParser(description="LoTTE read #1: the pre-build veto and observational "
                                             "row (m13/LOTTE_GATE_REGISTRATION.json, ruling R16)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--write-manifest", action="store_true",
                    help="write m13/LOTTE_GATE_MANIFEST.json from the published E arm records (the "
                         "lock's second manifest commit); opens no LoTTE path; you commit and push it")
    ap.add_argument("--preflight-only", action="store_true",
                    help="check the registration, E1 verdict, arm records, checkpoints, manifest "
                         "and pin; open no LoTTE path and write nothing")
    ap.add_argument("--recover", action="store_true",
                    help="complete a read that crashed after its receipt was written: identical "
                         "identity required, only slices without a journaled output are read")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None):
    a = build_argparser().parse_args(argv)
    cfg = Config(device=a.device)
    if a.write_manifest:
        write_manifest(cfg, verbose=not a.quiet)
        return 0
    run(cfg, preflight_only=a.preflight_only, recover=a.recover, verbose=not a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
