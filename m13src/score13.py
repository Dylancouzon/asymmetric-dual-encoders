"""M13 — the six-set scoring transaction (`m13/STAGE1_DESIGN.md` §2).

One shot. The ordering below is not stylistic: the spent receipt is pushed to origin BEFORE the
first protected byte is opened, so "was the access consumed?" has a durable external answer.

    lock + seal -> preflight (no protected payload)
    -> ledger FINAL-RUN-BEGIN commit + push
    -> the BEGIN-BOUND RUN MANIFEST, atomically, BEFORE the tag (review B2/B3)
    -> annotated `m10-six-spent` tag + push          <- the access is spent HERE
    -> payload-level checks (lengths, duplicates, qid-set equality)   [ruling R1: INSIDE]
    -> per dataset in `partitions.all6`:
         verify_and_load + query-text/qrel hashes -> stella document vectors (cached, NEVER
         encoded, shards verified present) -> student queries
         -> bge-small anchor -> exact search -> per-query nDCG@10
         -> ATOMIC per-dataset write of string-keyed scores + the run-time registry sha
         -> bridge (dataset-mean |delta| <= 0.003 hard gate; per-query movement reported;
            the anchor row is DISCARDED)
    -> evidence_for x4, each guarded independently -> decide -> headline      [nano only]
    -> reserved batch iff `reserved_batch_runs`
    -> FINAL-RUN-END + digest pushed

TWO re-entry modes, disjoint by the tag (`final_run_registry.infra_retry_admissible_iff`):

  * PRE-TAG `--infra-retry` — the tag is not on origin; nothing was spent.
  * `--recover` — decisions only, from persisted raw scores, NEVER re-reading the six.

There is NO post-tag continuation (ruling R14, 2026-09-10, which withdraws R11 unexecuted): once
the tag is on origin the access is consumed, and a rerun reports what was persisted and exits
nonzero. Reliability comes from rehearsing this exact code on the synthetic fixture, not from
recovery machinery. `m13src/rehearse13.py --run` is that rehearsal.

`--preflight-only` is OBSERVATIONAL in every state (review B1): it reports and exits before any
dispatch to the post-tag report, recovery or scoring, and opens no protected payload.

Datasets come from `partitions.all6`, NEVER `bench/core.py:DATASETS` (five unless `m7src/_paths` is
imported first — M13 CODEMAP pitfall 3).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import access13
from access13 import (END, commit_and_push, ledger_append, preflight, sha256_file,
                      seal_protected_paths, spend_access, spent_tag_exists, utcnow, write_atomic)

REPO = access13.REPO
ANCHOR_PREFIX = "Represent this sentence for searching relevant passages: "

EXIT_OK, EXIT_REFUSED, EXIT_NOT_DURABLE, EXIT_ACCESS_LOST = 0, 2, 3, 4
EXIT_INCOMPLETE_RESERVED = 5

# The code whose identity the run manifest binds: this executor plus the modules it reuses to load,
# encode, score and decide (review B2). A recovery that runs different code is not the same run.
CODE_IDENTITY_FILES = ("m13src/score13.py", "m13src/access13.py", "m7src/final_run.py",
                       "m7src/teacher.py", "m7src/evalkit.py", "m7src/hashing.py",
                       "m10src/final10.py", "m10src/nano10.py")

# The identity a per-dataset score row must carry and re-present to `--recover`.
IDENTITY_FIELDS = ("system", "freeze_sha256", "registry_sha256", "begin_commit", "code_sha256",
                   "comparator_sha256")


def load_conf(cfg):
    return json.loads(Path(cfg.registry_path).read_text())


def code_identity():
    """sha256 over the executor and every reused module, by relative path.

    Deliberately hashed from THIS checkout (`access13.REPO`), not from `cfg.repo`: the rehearsal
    points every data path at a fixture but still executes the source in this working tree, which
    is exactly the source whose identity a recovery must re-present.
    """
    per_file = {}
    for rel in CODE_IDENTITY_FILES:
        p = REPO / rel
        if not p.exists():
            raise SystemExit(f"REFUSED: {rel} is missing; the executed code cannot be identified.")
        per_file[rel] = sha256_file(p)
    return access13.sha_json(per_file)


def run_manifest(cfg, conf, freeze, begin_commit, candidate):
    """The BEGIN-bound run manifest. Written atomically BEFORE the tag is pushed."""
    return {
        "begin_commit": begin_commit,
        "system": candidate,
        "freeze_sha256": freeze["_sha256"],
        "checkpoint": freeze["checkpoint"],
        "registry_sha256": sha256_file(cfg.registry_path),
        "comparator_sha256": sha256_file(cfg.perquery_path),
        "code_sha256": code_identity(),
        "code_files": list(CODE_IDENTITY_FILES),
        "datasets": list(conf["partitions"]["all6"]),
        "teacher": {"model_id": cfg.teacher_model_id, "revision": cfg.teacher_revision,
                    "dtype": cfg.teacher_dtype},
        "spent_tag": cfg.spent_tag,
        "started": utcnow(),
        "_note": "written BEFORE the m10-six-spent tag; every per-dataset row re-presents the "
                 "identity fields below and `--recover` refuses a row that differs.",
    }


def identity_of(man):
    return {k: man.get(k) for k in IDENTITY_FIELDS}


def row_identity_problems(ds, row, man):
    """A persisted row belongs to THIS run or it is not usable (review B2)."""
    return [f"`{ds}`: {k} {str(row.get(k))[:12]} != the run manifest's {str(man.get(k))[:12]}"
            for k in IDENTITY_FIELDS if row.get(k) != man.get(k)]


def bind_final10(cfg):
    """`final10` resolves its constants through a module-level REGISTRY path, and `canonical()`
    refuses any conf that is not that file. Pointing it at the config's registry is how the
    rehearsal runs the production decision layer against a fixture registry."""
    import final10
    final10.REGISTRY = Path(cfg.registry_path)
    return final10


# ------------------------------------------------------------------ loading and encoding

def assert_transaction_phase(cfg, ds):
    """The phase gate in front of every `results/frozen_eval/<ds>.json` read (review A4).

    The run manifest exists and the spent tag exists in this repository. Both become true only
    inside `spend_access`, so no preflight, no `--preflight-only` report and no helper can reach a
    frozen payload before the access is spent.
    """
    if not Path(cfg.state_path).exists():
        raise SystemExit(f"REFUSED to open results/frozen_eval/{ds}.json: no run manifest at "
                         f"{cfg.state_path}. The six are readable only inside the transaction.")
    if not access13.sh(cfg, "git", "tag", "-l", cfg.spent_tag):
        raise SystemExit(f"REFUSED to open results/frozen_eval/{ds}.json: {cfg.spent_tag} does not "
                         f"exist. The receipt is created before the first protected read.")


def open_frozen_six(cfg, ds):
    """THE reader for `results/frozen_eval/<ds>.json`, and the only one in this module. Every
    other caller — `payload_checks`, `load_six` — goes through here or through
    `final_run.verify_and_load` with the same gate asserted first."""
    assert_transaction_phase(cfg, ds)
    return json.loads((Path(cfg.frozen_eval_dir) / f"{ds}.json").read_text())


def authenticate_queries(cfg, ds, q_ids, q_texts, qrels):
    """Hash the EXACT objects about to be scored against the manifest (review A1).

    `verify_and_load` authenticates the corpus only, so changed query text or changed labels under
    unchanged qids would have been encoded and scored silently. The convention is the producer's
    (`scripts/freeze_eval_assets.py`:22,44 — `sha(obj) = sha256(json.dumps(obj, sort_keys=True))`).
    A manifest field that is absent is a REFUSAL, never a skipped check.
    """
    man = json.loads(Path(cfg.manifest_path).read_text())["datasets"][ds]
    want = {"qids_sha256": access13.sha_json(sorted(q_ids)),
            "qtexts_sha256": access13.sha_json([t for _q, t in sorted(zip(q_ids, q_texts))]),
            "qrels_sha256": access13.sha_json(qrels)}
    for field, got in want.items():
        if field not in man:
            raise SystemExit(f"ABORTED on `{ds}`: eval_manifest.json has no `{field}`, so the "
                             f"queries actually scored cannot be authenticated. Regenerate the "
                             f"manifest with scripts/freeze_eval_assets.py; nothing is scored "
                             f"against an unauthenticated payload.")
        if man[field] != got:
            raise SystemExit(f"ABORTED on `{ds}`: {field} mismatch vs the frozen manifest "
                             f"({got[:12]} vs {str(man[field])[:12]}). The loaded queries or "
                             f"labels are not the frozen ones.")
    return want


def load_six(cfg, ds):
    """-> (doc_ids, doc_texts, q_ids, q_texts, qrels), qids sorted, labels from frozen_eval only.

    Production reuses `m7src/final_run.verify_and_load` verbatim. It fetches the corpus from
    HuggingFace, which a synthetic fixture cannot supply, so `cfg.corpus_reader` injects the corpus
    and the rest of the function — manifest hash fields, the frozen payload, sorted qids, the
    ledger line — is the same sequence of checks.
    """
    assert_transaction_phase(cfg, ds)
    if cfg.corpus_reader is None:
        import final_run
        final_run.MANIFEST = Path(cfg.manifest_path)
        final_run.FROZEN = Path(cfg.frozen_eval_dir)
        final_run.LEDGER = Path(cfg.ledger_path)
        doc_ids, doc_texts, q_ids, q_texts, qrels = final_run.verify_and_load(ds, "six")
    else:
        from hashing import sha_stream_list
        man = json.loads(Path(cfg.manifest_path).read_text())["datasets"][ds]
        doc_ids, doc_texts = cfg.corpus_reader(ds)
        for name, got in (("corpus_ids_sha256", sha_stream_list(doc_ids)),
                          ("corpus_text_sha256", sha_stream_list(doc_texts)),
                          ("n_docs", len(doc_ids))):
            if man[name] != got:
                raise SystemExit(f"ABORTED: {ds}.{name} mismatch vs the frozen manifest")
        froz = open_frozen_six(cfg, ds)
        q_ids = sorted(froz["queries"])
        q_texts, qrels = [froz["queries"][q] for q in q_ids], froz["qrels"]
        ledger_append(cfg, f"- {utcnow()} — **FINAL-RUN access** (six) `{ds}`: {len(doc_ids):,} "
                           f"docs / {len(q_ids):,} queries, corpus hashes verified against the "
                           f"frozen manifest, labels read from `{ds}.json`.")
    # The query text and the labels are authenticated on the objects that are about to be scored,
    # not on a second read of the file (review A1).
    authenticate_queries(cfg, ds, q_ids, q_texts, qrels)
    return doc_ids, doc_texts, q_ids, q_texts, qrels


def _refuse_to_encode(*a, **k):
    raise SystemExit(
        "REFUSED: the transaction asked to ENCODE document text. The six-set numbers come from the "
        "frozen document cache and nothing else; an encode here would silently produce vectors "
        "from a teacher this run never verified (review A2). Delete nothing — find the missing "
        "shard.")


def cache_record(cfg, ds, doc_texts):
    """-> the consumed cache's identity and shard hashes, refusing before any encode (review A2).

    `teacher.encode_cached` encodes whatever shard is absent, so "frozen-cache-only" has to be
    established BEFORE the call: derive the exact cache key, require the cache's own manifest to
    name every shard the corpus needs, and require each of those files to exist at the recorded
    byte size. `encode_cached(verify=True)` then re-hashes them.
    """
    import torch
    import teacher
    if (teacher.TEACHER, teacher.TEACHER_REV) != (cfg.teacher_model_id, cfg.teacher_revision):
        raise SystemExit(
            f"REFUSED: the active teacher is {teacher.TEACHER}@{teacher.TEACHER_REV[:12]}, not the "
            f"pinned {cfg.teacher_model_id}@{cfg.teacher_revision[:12]}. Set M7_ENCODER to the "
            f"document tower the index was built with; a different teacher is a different index.")
    if cfg.teacher_dtype != "fp32":
        raise SystemExit(f"REFUSED: the registered document encode dtype is fp32, not "
                         f"{cfg.teacher_dtype!r}.")
    name = cfg.doc_cache_fmt.format(ds=ds)
    key, _blob = teacher.cache_key(name, "", 512, teacher.TEACHER, teacher.TEACHER_REV,
                                   teacher.sha_texts(doc_texts), torch.float32)
    d = Path(cfg.enc_root) / key
    man = d / "shards.json"
    if not man.exists():
        raise SystemExit(f"REFUSED: no shard manifest at {man}. The document vectors for `{ds}` "
                         f"were never hash-recorded, and this transaction never encodes.")
    shards = (json.loads(man.read_text()).get("shards") or {})
    n_needed = (len(doc_texts) + teacher.SHARD - 1) // teacher.SHARD
    problems = []
    for s in range(n_needed):
        sid = f"{s:05d}"
        rec, p = shards.get(sid), d / f"shard_{sid}.npy"
        if rec is None:
            problems.append(f"shard {sid} is not recorded in shards.json")
        elif not p.exists():
            problems.append(f"shard {sid} is recorded but missing on disk")
        elif rec.get("bytes") is not None and p.stat().st_size != rec["bytes"]:
            problems.append(f"shard {sid} is {p.stat().st_size} bytes, recorded as {rec['bytes']}")
    if problems:
        raise SystemExit(f"REFUSED before any encode: the `{ds}` document cache {d.name} is "
                         f"incomplete:\n  - " + "\n  - ".join(problems))
    return {"cache_key": key, "dir": d.name, "n_shards": n_needed, "n_rows": len(doc_texts),
            "teacher": {"model_id": teacher.TEACHER, "revision": teacher.TEACHER_REV,
                        "dtype": cfg.teacher_dtype},
            "shard_sha256": {f"{s:05d}": shards[f"{s:05d}"].get("sha256")
                             for s in range(n_needed)}}


def doc_vectors(cfg, ds, doc_texts):
    """-> (vectors, consumed-cache record). `verify=True`: never re-encode silently — a cache whose
    bytes changed under us is a fact to surface, not to paper over. And `teacher.encode` is
    replaced for the duration of the call so a cache gap raises instead of encoding (review A2:
    `encode_cached` has no no-encode flag)."""
    if cfg.doc_vector_loader is not None:
        return cfg.doc_vector_loader(cfg, ds, doc_texts), {"injected_loader": True}
    import torch
    import teacher
    teacher.ENC = Path(cfg.enc_root)
    rec = cache_record(cfg, ds, doc_texts)
    original_encode = teacher.encode
    teacher.encode = _refuse_to_encode
    try:
        vecs = teacher.encode_cached(cfg.doc_cache_fmt.format(ds=ds), doc_texts, prefix="",
                                     dtype=torch.float32, verify=True)
    finally:
        teacher.encode = original_encode
    rec["combined_sha256"] = (teacher.PROVENANCE.get(cfg.doc_cache_fmt.format(ds=ds), {})
                              .get("combined_sha256"))
    return vecs, rec


class Nano10Student:
    """`m10src/nano10.Nano10` behind the `encode(texts) -> ndarray` adapter contract."""

    def __init__(self, freeze, device=None, repo=REPO):
        import torch
        from nano10 import Nano10
        self.model = Nano10(freeze["student_key"], n_layers=freeze.get("n_layers", 3),
                            head=freeze.get("head", "linear"),
                            out_dim=freeze.get("out_dim", 1024),
                            max_seq=freeze.get("max_seq", 512))
        blob = torch.load(Path(repo) / freeze["checkpoint"], map_location="cpu",
                          weights_only=False)
        sd = blob.get("model", blob) if isinstance(blob, dict) else blob
        self.model.load_state_dict(sd)
        if not self.model.under_cap():
            raise SystemExit(f"REFUSED: the frozen student has {self.model.n_params():,} "
                             f"parameters, over the registered 35M cap.")
        self.model.eval()
        if device:
            self.model.to(device)
        self.prefix = freeze.get("query_prefix", "")

    def encode(self, texts):
        return self.model.encode_queries(list(texts), prefix=self.prefix)


class M9Student:
    """`m9src/nano.Nano` — a different class with the same serving contract. Imported lazily:
    `m9base` installs `paths_guard` at import (M10 CODEMAP pitfall 14)."""

    def __init__(self, freeze, device=None, repo=REPO):
        import torch
        sys.path.insert(0, str(REPO / "m9src"))
        from nano import Nano
        self.model = Nano(freeze["student_key"], out_dim=freeze.get("out_dim", 1024))
        blob = torch.load(Path(repo) / freeze["checkpoint"], map_location="cpu",
                          weights_only=False)
        sd = blob.get("model", blob) if isinstance(blob, dict) else blob
        self.model.load_state_dict(sd)
        self.model.eval()
        if device:
            self.model.to(device)
        self.prefix = freeze.get("query_prefix", "")

    def encode(self, texts):
        return self.model.encode_queries(list(texts), prefix=self.prefix)


class BgeSmallAnchor:
    """bge-small-en-v1.5 served symmetrically — the bridge anchor, ported from
    `m9src/bridge_dryrun.Anchor` and made device-agnostic. Reproducibility, not quality."""

    REPO_ID = "BAAI/bge-small-en-v1.5"
    REV = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

    def __init__(self, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from _paths import DEVICE
        self.torch = torch
        self.device = device or DEVICE
        self.tok = AutoTokenizer.from_pretrained(self.REPO_ID, revision=self.REV)
        self.m = AutoModel.from_pretrained(self.REPO_ID, revision=self.REV).to(self.device).eval()

    def encode(self, texts, batch_size=256):
        torch = self.torch
        texts = list(texts)
        out = np.empty((len(texts), self.m.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                b = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(self.device)
                h = self.m(**b).last_hidden_state[:, 0]
                out[i:i + batch_size] = torch.nn.functional.normalize(
                    h.float(), dim=-1).cpu().numpy()
        return out


def exact_scores(cfg, qv, dv, doc_ids, q_ids, qrels):
    """Exact search + per-query nDCG@10, string-keyed."""
    from evalkit import per_query_ndcg, topk_ids_scores
    run = topk_ids_scores(qv, dv, doc_ids, k=cfg.topk, chunk=cfg.chunk, qids=list(q_ids))
    return {str(k): float(v) for k, v in per_query_ndcg(run, qrels).items()}


# ------------------------------------------------------------------ payload + qid checks

def comparator_rows(cfg, conf, datasets):
    """-> {system: {dataset: {qid: score}}} from the frozen comparator, qids as STRINGS."""
    pq = json.loads(Path(cfg.perquery_path).read_text())["datasets"]
    out = {}
    for ds in datasets:
        qids = [str(q) for q in pq[ds]["qids"]]
        for sysname, row in pq[ds]["systems"].items():
            out.setdefault(sysname, {})[ds] = {q: float(v) for q, v in zip(qids, row)}
    return out


def comparator_qids(cfg, datasets):
    pq = json.loads(Path(cfg.perquery_path).read_text())["datasets"]
    return {ds: [str(q) for q in pq[ds]["qids"]] for ds in datasets}


def payload_checks(cfg, conf, datasets):
    """The payload-level checks, INSIDE the transaction, first after the tag (ruling R1, settled).

    Lengths, duplicates and qid-set equality between the frozen payload and the comparator. There
    is no switch: preflight is manifest-only and the payload is reachable only through
    `open_frozen_six`, which asserts the phase (review B1/A4).
    """
    problems = []
    pq_qids = comparator_qids(cfg, datasets)
    for ds in datasets:
        froz = open_frozen_six(cfg, ds)
        q = list(froz["queries"])
        if len(set(q)) != len(q):
            problems.append(f"{ds}: the frozen payload has duplicate qids")
        if any(not isinstance(x, str) for x in q):
            problems.append(f"{ds}: the frozen payload has non-string qids")
        if set(q) != set(pq_qids[ds]):
            problems.append(f"{ds}: frozen payload qids != comparator qids "
                            f"({len(set(q) - set(pq_qids[ds]))} extra, "
                            f"{len(set(pq_qids[ds]) - set(q))} missing)")
        if len(q) != len(pq_qids[ds]):
            problems.append(f"{ds}: {len(q)} frozen queries vs {len(pq_qids[ds])} comparator qids")
    return problems


def assert_qid_set(ds, scored, frozen_qids, comp_qids):
    """pytrec_eval silently omits a run qid that has no qrel, so the scored set is asserted against
    BOTH the frozen list and the comparator's — as strings (M13 CODEMAP pitfall 4)."""
    scored_set, froz_set, comp_set = set(scored), {str(q) for q in frozen_qids}, set(comp_qids)
    if scored_set != froz_set:
        raise SystemExit(
            f"HARD FAIL on `{ds}`: the scored qid set is not the frozen list "
            f"({len(scored_set - froz_set)} extra, {len(froz_set - scored_set)} missing). "
            f"pytrec_eval drops a qid with no qrel; a missing query is a different evaluation.")
    if froz_set != comp_set:
        raise SystemExit(
            f"HARD FAIL on `{ds}`: the frozen qid set is not the comparator's "
            f"({len(froz_set - comp_set)} extra, {len(comp_set - froz_set)} missing). "
            f"Strict alignment would abort after the access was spent.")


def bridge_row(ds, anchor_scores, frozen_row, max_mean_delta):
    """The dataset-mean bridge. Hard-fails on qid-set inequality or |mean delta| > the registered
    bound; per-query movement is REPORTED and never fails (registry `bridge.report_never_fail`)."""
    if set(anchor_scores) != set(frozen_row):
        raise SystemExit(f"BRIDGE HARD FAIL on `{ds}`: the re-scored anchor's qid set differs from "
                         f"the frozen bge-small row. A missing, extra or reordered query is a "
                         f"different evaluation, not a numerical difference.")
    qids = sorted(anchor_scores)
    d = np.array([anchor_scores[q] - frozen_row[q] for q in qids], dtype=np.float64)
    row = {"n": int(d.size), "mean_delta": float(d.mean()),
           "mean_abs_delta": float(np.abs(d).mean()),
           "max_abs_delta": float(np.abs(d).max()),
           "changed_queries": int((d != 0).sum()),
           "bound": float(max_mean_delta), "passes": bool(abs(float(d.mean())) <= max_mean_delta)}
    if not row["passes"]:
        raise SystemExit(f"BRIDGE HARD FAIL on `{ds}`: dataset |mean delta| "
                         f"{abs(row['mean_delta']):.6f} > {max_mean_delta}. The access is consumed "
                         f"and no verdict is produced (registry `bridge.failure`).")
    return row


# ------------------------------------------------------------------ per-dataset scoring

def score_dataset(cfg, conf, ds, student, anchor, comp_qids, frozen_anchor_row, meta):
    doc_ids, doc_texts, q_ids, q_texts, qrels = load_six(cfg, ds)
    dv, cache = doc_vectors(cfg, ds, doc_texts)
    qv = student.encode(q_texts)
    scores = exact_scores(cfg, qv, dv, doc_ids, q_ids, qrels)
    assert_qid_set(ds, scores, q_ids, comp_qids)

    adv = anchor.encode(doc_texts)
    aqv = anchor.encode([ANCHOR_PREFIX + t for t in q_texts])
    anchor_scores = exact_scores(cfg, aqv, adv, doc_ids, q_ids, qrels)
    assert_qid_set(ds, anchor_scores, q_ids, comp_qids)
    try:
        bridge = bridge_row(ds, anchor_scores, frozen_anchor_row,
                            conf["bridge"]["dataset_mean_abs_delta_max"])
    except SystemExit as e:
        # A bridge failure is TERMINAL for this dataset and is persisted before anything else
        # happens (review B3): the failure, not merely the absence of a score, is what the
        # post-tag report and `--recover` read to say the evaluation is over.
        write_atomic(cfg.score_path(ds), json.dumps(
            {"dataset": ds, **identity_of(meta), "bridge_ok": False, "terminal": True,
             "error": str(e), "scores": None, "written": utcnow(),
             "_terminal": "registry `bridge.failure`: a HARD_FAIL_ON condition consumes the "
                          "access. This dataset is never re-scored and no verdict may be "
                          "produced from a partition containing it."}, indent=1))
        raise
    # The anchor row itself is validation-only and is DISCARDED here: only the movement summary is
    # persisted, and every conjunct uses the frozen rows (`bridge.anchor_row_is_discarded`).
    del anchor_scores, adv, aqv

    payload = {"dataset": ds, **identity_of(meta), "scores": scores, "n": len(scores),
               "bridge": bridge, "bridge_ok": True, "doc_cache": cache, "written": utcnow(),
               "_qids": "STRING keys; a JSON round-trip re-sorts int keys and changes the bound"}
    write_atomic(cfg.score_path(ds), json.dumps(payload, indent=1))
    print(f"[score13] {ds}: {len(scores)} queries, mean {np.mean(list(scores.values())):.4f}; "
          f"bridge mean_delta {bridge['mean_delta']:+.6f} max|d| {bridge['max_abs_delta']:.6f} "
          f"changed {bridge['changed_queries']}/{bridge['n']}", flush=True)
    return payload


def persisted(cfg, datasets):
    """-> {ds: payload} for every dataset already written."""
    out = {}
    for ds in datasets:
        p = cfg.score_path(ds)
        if p.exists():
            out[ds] = json.loads(p.read_text())
    return out


# ------------------------------------------------------------------ decisions

M9_REFUSAL = (
    "REFUSED: `decision_layer='m9'` is not wired until M9's own recipe decisions are fixed "
    "(review B13). Three concrete gaps, none of them a switch: (1) DATASETS — M9's registry has a "
    "top-level `datasets` list, not `partitions.all6`, and this executor iterates the latter "
    "everywhere; (2) BRIDGE — M9's `bridge._amended_2026_09_10` (ruling R3) is M9's own dated "
    "amendment and its constants must be read from M9's registry, not inherited from M10's; "
    "(3) CRASH POLICY — M9 has no `reserved._six_crash` and no R11 amendment, so post-tag "
    "the post-tag report's wording is M10's, not M9's. The student adapter "
    "(`score13.M9Student`) is kept and is the only part that is ready. Wire M9 as its own thin "
    "entry point, sharing these helpers, once those three are registered.")


def evidence_with_isolation(final10, conf, rows):
    """-> (evidence, errors, verdict). Each `evidence_for` call is guarded independently.

    `implementation.executor_deliverables`: "guard each `evidence_for` call independently: a raise
    on a conjunct the sequence never reached is recorded and omitted, not fatal". Reachability is
    not knowable before `decide` runs, so a failed conjunct starts ABSENT — which `decide` reports
    as NOT_TESTED once the sequence has stopped — and is promoted to the registry's `unscorable`
    sentinel only when `decide` says the sequence actually reaches it. Every error is persisted
    either way.
    """
    evidence, errors = {}, {}
    for cid in final10.sequence(conf):
        try:
            evidence[cid] = final10.evidence_for(cid, rows)
        except Exception as e:                                # noqa: BLE001 — recorded verbatim
            errors[cid] = f"{type(e).__name__}: {e}"
            print(f"[score13] evidence_for({cid}) FAILED: {errors[cid]}", flush=True)
    supplied = dict(evidence)
    for _ in range(len(errors) + 1):
        try:
            return evidence, errors, final10.decide(supplied)
        except ValueError as e:
            cid = next((c for c in errors if c not in supplied and c in str(e)), None)
            if cid is None:
                raise
            # The sequence reached a conjunct that could not be scored: the registry's handling is
            # UNSCORABLE — the sequence stops there and every verdict already established stands.
            supplied[cid] = {"unscorable": errors[cid]}
    raise SystemExit("REFUSED: the decision layer could not be resolved after isolating every "
                     f"evidence failure ({sorted(errors)}).")


def decision_record(cfg, conf, rows, candidate):
    """The registry selects the layer: nano -> `final10`; M9 refuses (review B13)."""
    if cfg.decision_layer == "m9":
        raise SystemExit(M9_REFUSAL)
    final10 = bind_final10(cfg)
    evidence, errors, verdict = evidence_with_isolation(final10, conf, rows)
    return {"layer": "nano", "candidate": candidate, "evidence": evidence,
            "evidence_errors": errors, "verdict": verdict,
            "reserved_batch_runs": bool(verdict["reserved_batch_runs"]),
            "outcome": ("REJECTED:" + ",".join(verdict["rejected"])) if verdict["rejected"]
                       else "no conjunct rejected",
            "headline": final10.headline(verdict)}


def build_rows(cfg, conf, scored, candidate):
    rows = comparator_rows(cfg, conf, list(scored))
    if candidate in rows:
        raise SystemExit(f"REFUSED: `{candidate}` already exists in the frozen comparator; the "
                         f"freshly scored candidate must not overwrite a frozen row.")
    rows[candidate] = {ds: p["scores"] for ds, p in scored.items()}
    return rows


# ------------------------------------------------------------------ the reserved batch

def reserved_batch(cfg, conf, rows_candidate):
    """The conditional reserved four — a clearly separated stage, per-system atomic resume.

    The encode step is a STUB by design: no FEVER or DBpedia document vectors exist under
    `work/enc` (names only; `m13/STAGE1_DESIGN.md` §4, ruling R10), so this refuses with a precise
    message rather than pretending the batch is priced or runnable. Note also that reading the
    reserved payloads requires an ALLOWLIST entry in `m8src/paths_guard.py` naming THIS module —
    `claim()` verifies the caller is physically the module it claims, so `m8src.final_run`'s entry
    cannot be borrowed. That is a LEDGER amendment, not an edit.
    """
    res = conf["reserved"]
    out_dir = Path(cfg.scores_dir) / "reserved"
    out_dir.mkdir(parents=True, exist_ok=True)
    done, todo = [], []
    for system in res["systems_included"]:
        (done if (out_dir / f"{system}.json").exists() else todo).append(system)
    if not todo:
        return {"status": "complete", "systems": done, "datasets": res["datasets"]}
    encoder = cfg.reserved_encoder
    if encoder is None:
        missing = [ds for ds in res["datasets"]
                   if not access13._doc_cache_dirs(cfg, ds.lower())]
        raise SystemExit(
            f"RESERVED BATCH NOT RUNNABLE: no document vectors for {missing}. The reserved four "
            f"need ~10M stella passage encodes that have never been made and are not in the M13 "
            f"allocation (ruling R10), and this module holds no `m8src/paths_guard` allowlist "
            f"entry, so it may not open a reserved payload at all. Systems already complete: "
            f"{done or 'none'}; still owed: {todo}. The six-set verdict above STANDS.")
    for system in todo:                      # per-system atomic outputs; resume at the first gap
        rec = encoder(cfg, conf, system, rows_candidate)
        write_atomic(out_dir / f"{system}.json", json.dumps(rec, indent=1))
        done.append(system)
    return {"status": "complete", "systems": done, "datasets": res["datasets"]}


# ------------------------------------------------------------------ the transaction

def _freeze_blob(cfg):
    fz = json.loads(Path(cfg.freeze_path).read_text())
    fz["_sha256"] = sha256_file(Path(cfg.repo) / fz["checkpoint"])
    return fz


def _student(cfg, freeze):
    if cfg.load_student is not None:
        return cfg.load_student(freeze)
    return (M9Student(freeze, repo=cfg.repo) if cfg.decision_layer == "m9"
            else Nano10Student(freeze, repo=cfg.repo))


def _anchor(cfg):
    return cfg.load_anchor() if cfg.load_anchor is not None else BgeSmallAnchor()


def bridge_problems(scored):
    """Decisions require every dataset's bridge to have SUCCEEDED — the persisted FLAG, not the
    mere presence of scores (review B3)."""
    return [f"`{ds}`: bridge_ok is {row.get('bridge_ok')!r}"
            + (f" ({str(row.get('error'))[:120]})" if row.get("error") else "")
            for ds, row in sorted(scored.items()) if row.get("bridge_ok") is not True]


def finalize(cfg, conf, scored, candidate, begin_commit, freeze_sha, mode):
    """Decisions from the persisted raw scores, then END + digest pushed."""
    failed = bridge_problems(scored)
    if failed:
        print("NO VERDICT: a registered bridge gate failed and the access is consumed "
              "(registry `bridge.failure`):\n  - " + "\n  - ".join(failed))
        return EXIT_REFUSED
    rows = build_rows(cfg, conf, scored, candidate)
    rec = decision_record(cfg, conf, rows, candidate)
    blob = {"mode": mode, "candidate": candidate, "begin_commit": begin_commit,
            "freeze_sha256": freeze_sha, "registry_sha256": sha256_file(cfg.registry_path),
            "datasets": sorted(scored), "spent_tag": cfg.spent_tag,
            "bridge": {ds: p["bridge"] for ds, p in scored.items()},
            "rows_note": "the RAW per-query scores live beside this file, one JSON per dataset; "
                         "--recover re-runs evidence_for on them",
            "decision_record": rec, "written": utcnow()}
    reserved, end_status = None, "COMPLETE"
    if rec["reserved_batch_runs"]:
        # The reserved batch is REQUIRED here (`reserved.trigger`). A run that cannot execute it
        # ends INCOMPLETE_RESERVED and exits nonzero: the six-set decisions stay durable, but the
        # run has not delivered everything the registry made conditional on them (review B13).
        try:
            reserved = reserved_batch(cfg, conf, rows[candidate])
        except SystemExit as e:
            reserved, end_status = {"status": "INCOMPLETE_RESERVED", "reason": str(e)}, \
                "INCOMPLETE_RESERVED"
    blob["reserved"] = reserved
    blob["end_status"] = end_status
    write_atomic(cfg.result_path, json.dumps(blob, indent=1))
    digest = sha256_file(cfg.result_path)
    ledger_append(cfg, f"- {utcnow()} — **{END}** ({mode}, {end_status}) result sha256 "
                       f"`{digest[:16]}` outcome `{rec['outcome']}`")
    ok, cmd, err = commit_and_push(
        cfg, [cfg.ledger_path, cfg.result_path, cfg.scores_dir],
        f"{cfg.ledger_prefix}: {END} ({mode}) {digest[:12]}")
    print(f"\n[score13] VERDICT outcome={rec['outcome']} layer={rec['layer']} "
          f"datasets={len(scored)} reserved_batch_runs={rec['reserved_batch_runs']} "
          f"end_status={end_status} result_sha256={digest[:16]} durable_on_origin={ok}")
    if not ok:
        print(f"FAILED: {cmd} ({err}). The decision is on disk but NOT durable on origin; it must "
              f"not be acted on until pushed.")
        return EXIT_NOT_DURABLE
    if end_status == "INCOMPLETE_RESERVED":
        print("INCOMPLETE_RESERVED: a conjunct rejected, so the reserved batch is REQUIRED and it "
              "could not run. The six-set decisions above are persisted and durable; this run is "
              "not finished.")
        return EXIT_INCOMPLETE_RESERVED
    return EXIT_OK


def candidate_system(cfg, conf):
    if cfg.decision_layer == "m9":
        raise SystemExit(M9_REFUSAL)
    return conf["conjuncts"][conf["sequence"]["order"][0]]["a"]


def score_all(cfg, conf, datasets, man, freeze, already):
    """`man` is the BEGIN-bound run manifest; every row it writes re-presents its identity."""
    candidate = man["system"]
    meta = identity_of(man)
    todo = [ds for ds in datasets if ds not in already]
    if todo:
        student, anchor = _student(cfg, freeze), _anchor(cfg)
        comp = comparator_qids(cfg, datasets)
        frozen_anchor = comparator_rows(cfg, conf, datasets)[conf["bridge"]["anchor"]]
        for ds in todo:
            already[ds] = score_dataset(cfg, conf, ds, student, anchor, comp[ds],
                                        frozen_anchor[ds], meta)
    return candidate, already


def preflight_report(cfg, conf, infra_retry=False):
    """OBSERVATIONAL in every state (review B1). Opens no protected payload, spends nothing,
    dispatches nowhere, and exits 0 whatever it finds — it is a readiness report, and a report
    that dispatched into a post-tag path was the hazard."""
    problems = preflight(cfg, conf, infra_retry)
    if problems:
        print("[score13] --preflight-only REPORT (nothing was opened, nothing was spent):\n  - "
              + "\n  - ".join(problems))
    else:
        print("[score13] --preflight-only REPORT: clean (no protected payload was opened).")
    return EXIT_OK


def run(cfg=None, infra_retry=False, preflight_only=False, recover=False):
    cfg = cfg or access13.production()
    conf = load_conf(cfg)
    if cfg.decision_layer == "m9":
        raise SystemExit(M9_REFUSAL)
    datasets = list(conf["partitions"]["all6"])
    seal_protected_paths(fatal=True)
    access13.acquire_lock(cfg)

    # BEFORE any dispatch: to the post-tag report, to recovery or to scoring.
    if preflight_only:
        return preflight_report(cfg, conf, infra_retry)

    if recover:
        return recover_run(cfg, conf, datasets)

    exists, where = spent_tag_exists(cfg, conf)
    scored = persisted(cfg, datasets)
    if exists and where == "origin":
        return post_tag_report(cfg, conf, datasets, scored)

    problems = preflight(cfg, conf, infra_retry)
    if problems:
        print("FINAL RUN REFUSED:\n  - " + "\n  - ".join(problems))
        return EXIT_REFUSED
    print("[score13] preflight clean (no protected payload was opened).")

    freeze = _freeze_blob(cfg)
    candidate = candidate_system(cfg, conf)
    man = {}

    def write_manifest_before_the_tag(begin):
        # The manifest is BEGIN-bound and lands BEFORE the receipt (review B2/B3), so every row
        # this run persists can be tied to one code, registry, comparator and checkpoint identity
        # — and `--recover` can refuse rows that belong to a different run.
        man.update(run_manifest(cfg, conf, freeze, begin, candidate))
        write_atomic(cfg.state_path, json.dumps(man, indent=1))

    begin = spend_access(cfg, freeze["_sha256"], before_tag=write_manifest_before_the_tag)
    if not Path(cfg.state_path).exists():
        raise SystemExit("REFUSED: the run manifest is missing after the tag was pushed.")

    problems = payload_checks(cfg, conf, datasets)
    if problems:
        raise SystemExit("PAYLOAD CHECKS FAILED after the access was spent:\n  - "
                         + "\n  - ".join(problems))
    candidate, scored = score_all(cfg, conf, datasets, man, freeze, scored)
    return finalize(cfg, conf, scored, candidate, begin, freeze["_sha256"], "full")


def _identity_problems(cfg, conf, state):
    """The run manifest still describes THIS checkout, this registry and this receipt."""
    problems = []
    live_code = code_identity()
    if live_code != state.get("code_sha256"):
        problems.append(f"the executed code {live_code[:12]} is not the manifest's "
                        f"{str(state.get('code_sha256'))[:12]}; a recovery re-runs the same code.")
    live_comp = sha256_file(cfg.perquery_path)
    if live_comp != state.get("comparator_sha256"):
        problems.append(f"the comparator sha {live_comp[:12]} is not the manifest's "
                        f"{str(state.get('comparator_sha256'))[:12]}.")
    tag_at = access13.remote_tag_commit(cfg)
    if tag_at != state.get("begin_commit"):
        problems.append(f"the {cfg.spent_tag} tag on origin points at {str(tag_at)[:12]}, not the "
                        f"FINAL-RUN-BEGIN commit {str(state.get('begin_commit'))[:12]}.")
    if sorted(state.get("datasets") or []) != sorted(conf["partitions"]["all6"]):
        problems.append(f"the manifest expects datasets {sorted(state.get('datasets') or [])}, the "
                        f"live registry {sorted(conf['partitions']['all6'])}.")
    return problems


def post_tag_report(cfg, conf, datasets, scored):
    """The tag is on origin: the access is SPENT and there is no continuation (ruling R14).

    The registry's original wording stands — a crash after the tag consumes the access. This path
    re-opens nothing, scores nothing and invents nothing; it reports exactly what was persisted and
    exits nonzero. `--recover` can still turn a COMPLETE set of persisted, authenticated rows into
    decisions without any protected read.
    """
    done = sorted(scored)
    missing = [ds for ds in datasets if ds not in scored]
    terminal = sorted(ds for ds, row in scored.items() if row.get("bridge_ok") is not True)
    print(f"ACCESS SPENT. {cfg.spent_tag} is on origin, so the six-set access is consumed and this "
          f"executor will not re-open any dataset (ruling R14, 2026-09-10: no post-tag "
          f"continuation).\n  persisted: {done or 'none'}\n  never scored: {missing or 'none'}"
          f"\n  terminal bridge failure: {terminal or 'none'}\n  run manifest: "
          f"{'present' if Path(cfg.state_path).exists() else 'MISSING'}")
    if missing or terminal:
        print("  The six-set evaluation is INCOMPLETE and cannot be completed. Disclose the loss "
              "with the persisted rows above; do not rerun under any flag.")
    else:
        print("  Every dataset is persisted: `--recover` recomputes the decisions from these rows "
              "with zero protected reads.")
    return EXIT_ACCESS_LOST


def recover_run(cfg, conf, datasets):
    """Decisions from an already-written result or the persisted scores. NEVER re-reads the six.

    The registry-drift comparison lives HERE, outside `decide()`: re-deriving evidence stamps the
    NEW registry hash, so a check inside would pass trivially (`implementation.executor_contract`).
    """
    exists, where = spent_tag_exists(cfg, conf)
    scored = persisted(cfg, datasets)
    if not scored:
        if exists and where == "origin":
            print(f"ACCESS SPENT, NO RESULT. {cfg.spent_tag} is on origin but no per-dataset score "
                  f"file exists: the run died between spending the access and writing any score. "
                  f"No decision can be established and none may be invented. This is a documented "
                  f"LOSS of the six-set access — disclose it and stop.")
            return EXIT_ACCESS_LOST
        print("--recover needs persisted per-dataset scores.")
        return EXIT_REFUSED
    if not (exists and where == "origin"):
        print(f"--recover REFUSED: {cfg.spent_tag} is not on origin ({where or 'absent'}). A result "
              f"without a durable spend receipt has unverified provenance and must not be turned "
              f"into a decision.")
        return EXIT_REFUSED
    missing = [ds for ds in datasets if ds not in scored]
    if missing:
        print(f"--recover REFUSED: {missing} have no persisted scores. Decisions are computed over "
              f"the whole registered partition, and there is no continuation that could add them "
              f"(ruling R14). The access is consumed; disclose the incomplete run.")
        return EXIT_REFUSED
    live = sha256_file(cfg.registry_path)
    drifted = sorted({p.get("registry_sha256") for p in scored.values()} - {live})
    if drifted:
        print(f"--recover REFUSED: the scores were produced under registry "
              f"{[str(d)[:12] for d in drifted]}, the live registry is {live[:12]}. The decision "
              f"constants moved between the run and the recovery.")
        return EXIT_REFUSED
    if not Path(cfg.state_path).exists():
        print(f"--recover REFUSED: {cfg.state_path} is missing, so the persisted rows cannot be "
              f"tied to the run that produced them (review B2).")
        return EXIT_REFUSED
    state = json.loads(Path(cfg.state_path).read_text())
    problems = _identity_problems(cfg, conf, state)
    if live != state.get("registry_sha256"):
        problems.append(f"the live registry sha {live[:12]} is not the run manifest's "
                        f"{str(state.get('registry_sha256'))[:12]}.")
    for ds, row in sorted(scored.items()):
        problems += row_identity_problems(ds, row, state)
    problems += [f"`{ds}`: bridge_ok is {row.get('bridge_ok')!r}"
                 for ds, row in sorted(scored.items()) if row.get("bridge_ok") is not True]
    if problems:
        print("--recover REFUSED:\n  - " + "\n  - ".join(problems)
              + "\n  Decisions are computed only from rows this run produced and whose bridge "
                "SUCCEEDED (reviews B2/B3).")
        return EXIT_REFUSED
    freeze = _freeze_blob(cfg)
    if freeze["_sha256"] != state.get("freeze_sha256"):
        print("--recover REFUSED: the result was produced under a different frozen checkpoint than "
              "the freeze file names.")
        return EXIT_REFUSED
    candidate = state["system"]
    cfg.result_path.unlink(missing_ok=True)
    return finalize(cfg, conf, scored, candidate, state.get("begin_commit"), freeze["_sha256"],
                    "recover")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--infra-retry", action="store_true",
                    help="admissible ONLY when the spent tag is absent from origin")
    ap.add_argument("--recover", action="store_true",
                    help="recompute decisions from persisted scores; never re-reads the six")
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args(argv)
    return run(access13.production(), infra_retry=a.infra_retry, preflight_only=a.preflight_only,
               recover=a.recover)


if __name__ == "__main__":
    sys.exit(main())
