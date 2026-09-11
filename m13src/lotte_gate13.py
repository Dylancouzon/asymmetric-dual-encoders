"""M13 — LoTTE read #1 as a small script (ruling R16): the pre-build veto and the observational row.

    .venv/bin/python m13src/lotte_gate13.py --preflight-only      # opens no LoTTE path, writes nothing
    .venv/bin/python m13src/lotte_gate13.py [--device cuda]       # THE read; writes m13/LOTTE_GATE.json

What it executes is registered, not decided here: `m13/LOTTE_GATE_REGISTRATION.json` (ruling R8)
names the seven remediated slices and their counts, nDCG@10 as the veto metric with Success@5 beside
it, the veto constants (margin 0.004, paired bootstrap B = 10,000 seed 903, one-sided 97.5% upper
bound) and the identities; `m10/LOTTE_LOCK.md` fixes WHEN (after both 5M E arms, before the build)
and the two branches:

  * E1 selected **bs32** -> the selected recipe IS the anchor recipe: the veto is SKIPPED and
    forfeited, and the observational row is still read on `E-bs32`'s cycle-3 checkpoint.
  * E1 selected **bs128** -> the veto RUNS: candidate `E-bs128`, comparator `E-bs32`, both the A100
    cycle-3 checkpoints their published arm records name. A veto means the 200M build trains bs32
    (`build13.check_gate` reads `decision` and overrides the E1 batch).

The record it writes is the one `m13src/build13.check_gate` refuses without: `executed: true`, the
branch, the decision, both checkpoint shas, the E1 verdict sha it read, `read_at`, `code_identity`,
and the per-slice numbers. It carries NO query text, no label and no per-query row; per-query scores
stay under `work/lotte/gate13/` inside the protected tree, and so do the stella document vectors
(`work/lotte/enc/`), so nothing derived from LoTTE leaves `work/lotte`.

Access. This module claims its own `m8src/paths_guard` allowlist entry (`m13src.lotte_gate13`,
LEDGER 15 amendment 2026-09-10) and opens only `collection.tsv`, `questions.forum.tsv` and
`qas.forum.jsonl` under `work/lotte/remediated/<topic>/<split>/` — never the GooAQ-licensed search
split, never the raw archive. Preflight reads the registration, the E1 verdict, the two arm records
and the two checkpoints, and touches no LoTTE path. An existing gate record refuses: read #1 is one
access and there is no third read. A crash before the record exists leaves resumable encode shards
and an `attempts.jsonl` line; re-running completes the same read (no verdict or row was produced).

Two pitfalls this file exists to get right: LoTTE's qids and pids are BOTH small integers, and
`evalkit.run_from_arrays` drops a hit whose doc id equals the query id (the BEIR self-hit rule), so
queries are scored under a `q:` namespace; and `pytrec_eval` silently omits a run qid without qrels,
so the scored qid set is asserted against the slice's every time (M13 CODEMAP pitfall 4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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


@dataclass
class Config:
    """Every path and injected component the read touches. Production defaults are the real ones;
    the tests build one pointing at a synthetic tree with an injected document encoder."""

    repo: Path = REPO
    registration_path: Path = REPO / "m13" / "LOTTE_GATE_REGISTRATION.json"
    record_path: Path = REPO / "m13" / "LOTTE_GATE.json"
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
    # (arm record summary, device) -> object with encode_queries(texts). None -> nano10 from the recipe
    load_student: object = None
    claim_guard: bool = True                     # a synthetic tree has no protected path to claim
    # the document tower's identity, the same pin `access13.Config` carries
    teacher_model_id: str = "NovaSearch/stella_en_400M_v5"
    teacher_revision: str = "ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20"


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")


def sha_obj(obj):
    """`m8src/freeze_lotte.sha`: sha256 of the sorted-key JSON encoding, so the slice hashes here
    are comparable with a pin written by that module."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def code_identity():
    """sha256 of the bytes of the code that RUNS the read — always this source tree, never a
    fixture's `cfg.repo`."""
    h = hashlib.sha256()
    for name in CODE_IDENTITY_FILES:
        h.update((REPO / name).read_bytes())
    return h.hexdigest()


def write_atomic(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    return p


def _is_sha(x):
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


# ------------------------------------------------------------------------------- preflight ----

def registration(cfg):
    """-> (the registration, its sha). Refuses a file that is not the R8 registration."""
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
    for f, want in (("margin", 0.004),):
        if veto.get(f) != want:
            refuse(f"{p}: veto.{f} is {veto.get(f)!r}, not the locked {want}")
    for f, want in (("B", 10000), ("seed", 903), ("paired_within_slice", True)):
        if boot.get(f) != want:
            refuse(f"{p}: veto.bootstrap.{f} is {boot.get(f)!r}, not the locked {want}")
    contract = (reg.get("gate_record_contract") or {}).get("path")
    if contract and (Path(cfg.repo) / contract).resolve() != Path(cfg.record_path).resolve():
        refuse(f"{p}: the registration's record path is {contract!r}, not {cfg.record_path}")
    return reg, sha256_file(p)


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


def arm_record(cfg, arm, registry):
    """-> the summary of one E arm's published record, or refuses. The checkpoint the gate reads
    is the one `build13._registered_e_checkpoints` will hold it to: the record's cycle-3 sha."""
    p = Path(cfg.arm_records_dir) / f"m10_arm_{slug(arm)}.json"
    if not p.exists():
        refuse(f"no published arm record at {p}: read #1 happens after BOTH 5M E arms finish "
               f"(m10/LOTTE_LOCK.md), and the gate compares their cycle-3 checkpoints")
    rec = json.loads(p.read_text())
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
    for f in ("student", "n_layers", "head", "dose_examples", "batch"):
        if f not in recipe:
            refuse(f"{p}: recipe lacks {f!r}; the student cannot be rebuilt from the record")
    entry = (registry.get("arms") or {}).get(arm) or {}
    if int(recipe["dose_examples"]) != int(entry.get("dose_examples", SCREEN_DOSE)) or \
            int(recipe["dose_examples"]) != SCREEN_DOSE:
        refuse(f"{p}: dose {recipe['dose_examples']} is not the registered screen dose "
               f"{SCREEN_DOSE} (the candidate is 'the 5M A100 arm', m10/LOTTE_LOCK.md)")
    if int(recipe["batch"]) != int(entry.get("batch")):
        refuse(f"{p}: batch {recipe['batch']} is not the registry's {entry.get('batch')} for {arm}")
    return {"arm": arm, "record": str(p), "record_sha256": sha256_file(p), "checkpoint": ck,
            "sha256": sha, "recipe": {k: recipe[k] for k in ("student", "n_layers", "head",
                                                                "dose_examples", "batch")},
            "params": rec.get("params"), "device": rec.get("device"),
            "git_head": rec.get("git_head")}


def verify_checkpoint(cfg, ar):
    p = Path(cfg.repo) / ar["checkpoint"]
    if not p.exists():
        refuse(f"{ar['arm']}: checkpoint {p} is not on this machine; the gate reads the exact "
               f"bytes the arm record names")
    got = sha256_file(p)
    if got != ar["sha256"]:
        refuse(f"{ar['arm']}: {p} hashes {got[:12]}, the record says {ar['sha256'][:12]}; "
               f"refusing to read a checkpoint that is not the published one")
    return p


def preflight(cfg, verbose=True):
    """Everything that must be true before a LoTTE path is opened. Opens none."""
    rec_p = Path(cfg.record_path)
    if rec_p.exists():
        refuse(f"{rec_p} already exists. Read #1 is ONE access and there is no third read "
               f"(m10/LOTTE_LOCK.md); a re-run is a protocol change, not a retry. Move the record "
               f"aside deliberately if it is not the executed gate.")
    reg, reg_sha = registration(cfg)
    branch, batch, vsha = branch_of(cfg)
    registry = json.loads(Path(cfg.registry_path).read_text())
    arms = {a: arm_record(cfg, a, registry) for a in E_ARMS}
    cand = arms["E-bs128" if branch == "bs128" else "E-bs32"]
    comp = arms["E-bs32"] if branch == "bs128" else None
    verify_checkpoint(cfg, cand)
    if comp is not None:
        verify_checkpoint(cfg, comp)
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
    order = sorted(reg["surface"]["slices"], key=lambda k: reg["surface"]["slices"][k]["docs_after_remedy"])
    plan = {"branch": branch, "e1_batch": batch, "e1_verdict_sha256": vsha,
            "candidate": cand, "comparator": comp,
            "veto_runs": comp is not None,
            "registration_sha256": reg_sha, "registration": reg,
            "slice_order": order, "veto": veto, "device": cfg.device, "teacher": teacher,
            "record_path": str(rec_p)}
    if verbose:
        print(f"LoTTE read #1 preflight: branch {branch} (E1 batch {batch}); veto "
              f"{'RUNS' if comp else 'SKIPPED and forfeited (identical recipe and action)'}")
        print(f"  candidate  {cand['arm']}: {cand['checkpoint']} {cand['sha256'][:12]}")
        if comp:
            print(f"  comparator {comp['arm']}: {comp['checkpoint']} {comp['sha256'][:12]}")
        print(f"  slices (smallest first): {order}")
        print(f"  veto: margin {veto['margin']}, B {veto['bootstrap']['B']}, seed "
              f"{veto['bootstrap']['seed']}, {veto['bootstrap']['interval']}")
        print(f"  device {cfg.device}; teacher {teacher}; no LoTTE path opened", flush=True)
    return plan


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


def read_slice(cfg, key, expected):
    """-> the remediated slice, checked against the registration's counts, with the five hashes
    `m8src/freeze_lotte._hash_slice` would compute. Refuses on any mismatch: a slice that is not
    the registered one is not read further."""
    topic, split = key.split("/")
    d = Path(cfg.remediated_dir) / topic / split
    for name in SLICE_FILES:
        if not (d / name).exists():
            refuse(f"{key}: {d / name} is missing; refusing to skip a registered slice")
    doc_ids, doc_texts = _read_tsv(d / "collection.tsv")
    q_ids, q_texts = _read_tsv(d / "questions.forum.tsv")
    rows = []
    with open(d / "qas.forum.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    qrels = {str(r["qid"]): sorted(str(p) for p in r["answer_pids"]) for r in rows}
    n_pairs = sum(len(v) for v in qrels.values())
    problems = []
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
        refuse(f"{key} under {d} is not the registered slice: " + "; ".join(problems))
    q_by_id = dict(zip(q_ids, q_texts))
    hashes = {"doc_ids_sha256": sha_obj(doc_ids), "doc_texts_sha256": sha_obj(doc_texts),
              "query_ids_sha256": sha_obj(sorted(q_ids)),
              "query_texts_sha256": sha_obj([q_by_id[q] for q in sorted(q_ids)]),
              "qrels_sha256": sha_obj(qrels)}
    return {"key": key, "doc_ids": doc_ids, "doc_texts": doc_texts, "q_ids": q_ids,
            "q_texts": q_texts, "qrels": {q: {p: 1 for p in v} for q, v in qrels.items()},
            "n_docs": len(doc_ids), "n_queries": len(q_ids), "n_qrels_pairs": n_pairs,
            "hashes": hashes, "read_relpath": str(d)}


def encode_docs(cfg, key, doc_texts, verbose=True):
    """-> (document vectors, cache record). stella once per slice, shard-resumable, cached INSIDE
    the protected tree (`teacher.ENC` is rebound to `cfg.enc_root`), prefix "" like the shared
    index's own caches, stored fp16 like the DEV-6 caches."""
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
                               device=cfg.device)
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

def load_student(cfg, ar):
    """The nano student the arm record describes, loaded from the exact checkpoint bytes it
    names. `trainer10.save` writes {"model": state_dict, ...}."""
    if cfg.load_student is not None:
        return cfg.load_student(ar, cfg.device)
    import torch
    import nano10 as N
    r = ar["recipe"]
    model = N.Nano10(r["student"], n_layers=int(r["n_layers"]), head=r["head"])
    blob = torch.load(Path(cfg.repo) / ar["checkpoint"], map_location="cpu", weights_only=False)
    sd = blob.get("model", blob) if isinstance(blob, dict) else blob
    model.load_state_dict(sd)
    if not model.under_cap():
        refuse(f"{ar['arm']}: {model.n_params():,} parameters, over the registered 35M cap")
    if ar.get("params") is not None and int(ar["params"]) != model.n_params():
        refuse(f"{ar['arm']}: the record says {ar['params']:,} parameters, the rebuilt student "
               f"has {model.n_params():,}; the recipe in the record does not describe it")
    model.eval()
    return model.to(cfg.device)


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
        missing = sorted(set(ns_ids) - set(ndcg))[:5]
        refuse(f"{sl['key']}: {len(ns_ids) - len(ndcg)} queries were not scored (e.g. {missing}); "
               f"pytrec_eval omits a qid without qrels (CODEMAP pitfall 4)")
    return {"ndcg10": {q: float(ndcg[QNS + q]) for q in q_ids},
            "success5": {q: success_at_k(run[QNS + q], sl["qrels"][q], 5) for q in q_ids}}


def slice_means(per_slice):
    """{slice: {qid: v}} -> ({slice: mean}, macro at equal weight over slices, never pooled)."""
    means = {k: float(np.mean(list(v.values()))) for k, v in per_slice.items()}
    return means, float(np.mean(list(means.values())))


def paired_bootstrap(cand, comp, B, seed, quantile=0.975, method="inverted_cdf"):
    """The registered veto interval: per-query (candidate - comparator) deltas on identical qids,
    resampled WITHIN each slice, the macro of the slice means per draw, and the one-sided upper
    bound as the empirical quantile (`inverted_cdf`, the repo's registered quantile method,
    `m9src/final_stats.bootstrap`). Slices enter the RNG stream in sorted order."""
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
    -margin. bs32: skipped and forfeited."""
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


def run(cfg=None, *, preflight_only=False, verbose=True):
    """Read #1, end to end. -> the gate record (or the preflight plan under `preflight_only`)."""
    cfg = cfg or Config()
    plan = preflight(cfg, verbose=verbose)
    if preflight_only:
        return plan
    if cfg.claim_guard:
        claim_lotte()
    t_start = time.time()
    read_at = utcnow()
    n_attempts = _attempt(cfg, {"started_at": read_at, "branch": plan["branch"],
                                "git_head": R.git_head(), "candidate": plan["candidate"]["sha256"]})
    students = {"candidate": load_student(cfg, plan["candidate"])}
    if plan["comparator"] is not None:
        students["comparator"] = load_student(cfg, plan["comparator"])
    reg = plan["registration"]
    per = {role: {"ndcg10": {}, "success5": {}} for role in students}
    surface, caches = {}, {}
    for key in plan["slice_order"]:
        sl = read_slice(cfg, key, reg["surface"]["slices"][key])
        dv, cache = encode_docs(cfg, key, sl["doc_texts"], verbose=verbose)
        for role, st in students.items():
            s = score_slice(cfg, st, sl, dv)
            per[role]["ndcg10"][key] = s["ndcg10"]
            per[role]["success5"][key] = s["success5"]
        del dv
        surface[key] = {k: sl[k] for k in ("n_docs", "n_queries", "n_qrels_pairs", "hashes",
                                            "read_relpath")}
        caches[key] = cache
        if verbose:
            row = " ".join(f"{role} nDCG@10 {np.mean(list(per[role]['ndcg10'][key].values())):.4f}"
                           for role in students)
            print(f"  [{key}] {row}", flush=True)
    rows = {}
    for role in students:
        n_means, n_macro = slice_means(per[role]["ndcg10"])
        s_means, s_macro = slice_means(per[role]["success5"])
        rows[role] = {"arm": plan[role]["arm"], "checkpoint_sha256": plan[role]["sha256"],
                      "ndcg10": {"per_slice": n_means, "macro": n_macro},
                      "success5": {"per_slice": s_means, "macro": s_macro}}
    veto = plan["veto"]
    boot = None
    if "comparator" in students:
        boot = paired_bootstrap(per["candidate"]["ndcg10"], per["comparator"]["ndcg10"],
                                veto["bootstrap"]["B"], veto["bootstrap"]["seed"])
    decision, fired = decide(plan, boot, veto["margin"])
    # per-query rows stay inside the protected tree; the record carries their hash only
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
        "candidate_sha256": plan["candidate"]["sha256"],
        "comparator_sha256": plan["comparator"]["sha256"] if plan["comparator"] else None,
        "candidate": plan["candidate"],
        "comparator": plan["comparator"],
        "macro_ndcg10": {role: rows[role]["ndcg10"] for role in rows},
        "success_at_5": {role: rows[role]["success5"] for role in rows},
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
        "attempts_before_this_record": n_attempts,
        "read_at": read_at,
        "seconds": round(time.time() - t_start, 1),
        "environment": _environment(cfg.device),
        "code_identity": code_identity(),
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
                f"Success@5 {rows['candidate']['success5']['macro']:.4f}; wrote {rec_p}", flush=True)
    return record


def build_argparser():
    ap = argparse.ArgumentParser(description="LoTTE read #1: the pre-build veto and observational "
                                             "row (m13/LOTTE_GATE_REGISTRATION.json, ruling R16)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--preflight-only", action="store_true",
                    help="check the registration, E1 verdict, arm records and checkpoints; open "
                         "no LoTTE path and write nothing")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None):
    a = build_argparser().parse_args(argv)
    run(Config(device=a.device), preflight_only=a.preflight_only, verbose=not a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
