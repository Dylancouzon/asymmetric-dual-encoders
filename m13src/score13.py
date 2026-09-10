"""M13 — the six-set scoring transaction (`m13/STAGE1_DESIGN.md` §2).

One shot. The ordering below is not stylistic: the spent receipt is pushed to origin BEFORE the
first protected byte is opened, so "was the access consumed?" has a durable external answer.

    lock + seal -> preflight (no protected payload)
    -> ledger FINAL-RUN-BEGIN commit + push
    -> annotated `m10-six-spent` tag + push          <- the access is spent HERE
    -> payload-level checks (lengths, duplicates, qid-set equality)   [ruling R1: INSIDE]
    -> per dataset in `partitions.all6`:
         verify_and_load -> stella document vectors (cached, verified) -> student queries
         -> bge-small anchor -> exact search -> per-query nDCG@10
         -> ATOMIC per-dataset write of string-keyed scores + the run-time registry sha
         -> bridge (dataset-mean |delta| <= 0.003 hard gate; per-query movement reported;
            the anchor row is DISCARDED)
    -> evidence_for x4 -> decide -> headline           [nano; M9 uses final_stats + final9.decide]
    -> reserved batch iff `reserved_batch_runs`
    -> FINAL-RUN-END + digest pushed

Three re-entry modes, disjoint by the tag and by whether the scores are complete
(`final_run_registry.infra_retry_admissible_iff`, `reserved._six_crash`):

  * PRE-TAG `--infra-retry` — the tag is not on origin; nothing was spent.
  * POST-TAG CONTINUATION (the default when the tag is on origin and scores are incomplete) —
    opens ONLY the datasets with no persisted scores, and only if HEAD equals the BEGIN commit and
    the live registry sha equals the persisted one.
  * `--recover` — decisions only, from persisted raw scores, NEVER re-reading the six.

A crash between the tag push and the FIRST persisted dataset is an outright loss of the access:
there is no persisted registry sha to authenticate a continuation against, and this executor will
say so and exit nonzero rather than re-open the six on its own authority.

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


def load_conf(cfg):
    return json.loads(Path(cfg.registry_path).read_text())


def bind_final10(cfg):
    """`final10` resolves its constants through a module-level REGISTRY path, and `canonical()`
    refuses any conf that is not that file. Pointing it at the config's registry is how the
    rehearsal runs the production decision layer against a fixture registry."""
    import final10
    final10.REGISTRY = Path(cfg.registry_path)
    return final10


# ------------------------------------------------------------------ loading and encoding

def load_six(cfg, ds):
    """-> (doc_ids, doc_texts, q_ids, q_texts, qrels), qids sorted, labels from frozen_eval only.

    Production reuses `m7src/final_run.verify_and_load` verbatim. It fetches the corpus from
    HuggingFace, which a synthetic fixture cannot supply, so `cfg.corpus_reader` injects the corpus
    and the rest of the function — manifest hash fields, the frozen payload, sorted qids, the
    ledger line — is the same sequence of checks.
    """
    if cfg.corpus_reader is None:
        import final_run
        final_run.MANIFEST = Path(cfg.manifest_path)
        final_run.FROZEN = Path(cfg.frozen_eval_dir)
        final_run.LEDGER = Path(cfg.ledger_path)
        return final_run.verify_and_load(ds, "six")

    from hashing import sha_stream_list
    man = json.loads(Path(cfg.manifest_path).read_text())["datasets"][ds]
    doc_ids, doc_texts = cfg.corpus_reader(ds)
    for name, got in (("corpus_ids_sha256", sha_stream_list(doc_ids)),
                      ("corpus_text_sha256", sha_stream_list(doc_texts)),
                      ("n_docs", len(doc_ids))):
        if man[name] != got:
            raise SystemExit(f"ABORTED: {ds}.{name} mismatch vs the frozen manifest")
    froz = json.loads((Path(cfg.frozen_eval_dir) / f"{ds}.json").read_text())
    q_ids = sorted(froz["queries"])
    ledger_append(cfg, f"- {utcnow()} — **FINAL-RUN access** (six) `{ds}`: {len(doc_ids):,} docs / "
                       f"{len(q_ids):,} queries, corpus hashes verified against the frozen "
                       f"manifest, labels read from `{ds}.json`.")
    return doc_ids, doc_texts, q_ids, [froz["queries"][q] for q in q_ids], froz["qrels"]


def doc_vectors(cfg, ds, doc_texts):
    """The frozen stella document vectors. `verify=True`: never re-encode silently — a cache whose
    bytes changed under us is a fact to surface, not to paper over."""
    if cfg.doc_vector_loader is not None:
        return cfg.doc_vector_loader(cfg, ds, doc_texts)
    import torch
    import teacher
    teacher.ENC = Path(cfg.enc_root)
    return teacher.encode_cached(cfg.doc_cache_fmt.format(ds=ds), doc_texts, prefix="",
                                 dtype=torch.float32, verify=True)


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
    """The payload-level checks, INSIDE the transaction (ruling R1's recommendation).

    Lengths, duplicates and qid-set equality between the frozen payload and the comparator. Flip
    `Config.payload_checks_inside` to False to run these in preflight instead — the one-line switch
    the ruling may require; note that doing so makes preflight open `results/frozen_eval/`.
    """
    problems = []
    pq_qids = comparator_qids(cfg, datasets)
    for ds in datasets:
        froz = json.loads((Path(cfg.frozen_eval_dir) / f"{ds}.json").read_text())
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
    dv = doc_vectors(cfg, ds, doc_texts)
    qv = student.encode(q_texts)
    scores = exact_scores(cfg, qv, dv, doc_ids, q_ids, qrels)
    assert_qid_set(ds, scores, q_ids, comp_qids)

    adv = anchor.encode(doc_texts)
    aqv = anchor.encode([ANCHOR_PREFIX + t for t in q_texts])
    anchor_scores = exact_scores(cfg, aqv, adv, doc_ids, q_ids, qrels)
    assert_qid_set(ds, anchor_scores, q_ids, comp_qids)
    bridge = bridge_row(ds, anchor_scores, frozen_anchor_row,
                        conf["bridge"]["dataset_mean_abs_delta_max"])
    # The anchor row itself is validation-only and is DISCARDED here: only the movement summary is
    # persisted, and every conjunct uses the frozen rows (`bridge.anchor_row_is_discarded`).
    del anchor_scores, adv, aqv

    payload = {"dataset": ds, "system": meta["system"], "scores": scores, "n": len(scores),
               "registry_sha256": meta["registry_sha256"], "begin_commit": meta["begin_commit"],
               "freeze_sha256": meta["freeze_sha256"], "bridge": bridge, "written": utcnow(),
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

def decision_record(cfg, conf, rows, candidate):
    """The registry selects the layer: nano -> `final10`; M9 -> `final_stats` + `final9.decide`."""
    if cfg.decision_layer == "m9":
        sys.path.insert(0, str(REPO / "m9src"))
        import final_stats
        import final9
        m9conf = final_stats.cfg()
        rec = final9.decide(rows, m9conf)
        return {"layer": "m9", "record": rec,
                "reserved_batch_runs": bool(rec["decision"]["C1_passes"]),
                "outcome": rec["decision"]["outcome"], "headline": None}
    final10 = bind_final10(cfg)
    evidence = {cid: final10.evidence_for(cid, rows) for cid in final10.sequence(conf)}
    verdict = final10.decide(evidence)
    return {"layer": "nano", "candidate": candidate, "evidence": evidence, "verdict": verdict,
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


def finalize(cfg, conf, scored, candidate, begin_commit, freeze_sha, mode):
    """Decisions from the persisted raw scores, then END + digest pushed."""
    rows = build_rows(cfg, conf, scored, candidate)
    rec = decision_record(cfg, conf, rows, candidate)
    blob = {"mode": mode, "candidate": candidate, "begin_commit": begin_commit,
            "freeze_sha256": freeze_sha, "registry_sha256": sha256_file(cfg.registry_path),
            "datasets": sorted(scored), "spent_tag": cfg.spent_tag,
            "bridge": {ds: p["bridge"] for ds, p in scored.items()},
            "rows_note": "the RAW per-query scores live beside this file, one JSON per dataset; "
                         "--recover re-runs evidence_for on them",
            "decision_record": rec, "written": utcnow()}
    reserved = None
    if rec["reserved_batch_runs"]:
        try:
            reserved = reserved_batch(cfg, conf, rows[candidate])
        except SystemExit as e:
            reserved = {"status": "NOT RUN", "reason": str(e)}
    blob["reserved"] = reserved
    write_atomic(cfg.result_path, json.dumps(blob, indent=1))
    digest = sha256_file(cfg.result_path)
    ledger_append(cfg, f"- {utcnow()} — **{END}** ({mode}) result sha256 `{digest[:16]}` "
                       f"outcome `{rec['outcome']}`")
    ok, cmd, err = commit_and_push(
        cfg, [cfg.ledger_path, cfg.result_path, cfg.scores_dir],
        f"{cfg.ledger_prefix}: {END} ({mode}) {digest[:12]}")
    print(f"\n[score13] VERDICT outcome={rec['outcome']} layer={rec['layer']} "
          f"datasets={len(scored)} reserved_batch_runs={rec['reserved_batch_runs']} "
          f"result_sha256={digest[:16]} durable_on_origin={ok}")
    if not ok:
        print(f"FAILED: {cmd} ({err}). The decision is on disk but NOT durable on origin; it must "
              f"not be acted on until pushed.")
        return EXIT_NOT_DURABLE
    return EXIT_OK


def score_all(cfg, conf, datasets, begin_commit, freeze, already):
    candidate = (conf["contrasts"]["C1"]["a"] if cfg.decision_layer == "m9"
                 else conf["conjuncts"][conf["sequence"]["order"][0]]["a"])
    meta = {"system": candidate, "registry_sha256": sha256_file(cfg.registry_path),
            "begin_commit": begin_commit, "freeze_sha256": freeze["_sha256"]}
    todo = [ds for ds in datasets if ds not in already]
    if todo:
        student, anchor = _student(cfg, freeze), _anchor(cfg)
        comp = comparator_qids(cfg, datasets)
        frozen_anchor = comparator_rows(cfg, conf, datasets)[conf["bridge"]["anchor"]]
        for ds in todo:
            already[ds] = score_dataset(cfg, conf, ds, student, anchor, comp[ds],
                                        frozen_anchor[ds], meta)
    return candidate, already


def run(cfg=None, infra_retry=False, preflight_only=False, recover=False):
    cfg = cfg or access13.production()
    conf = load_conf(cfg)
    datasets = list(conf["partitions"]["all6"])
    seal_protected_paths(fatal=True)
    access13.acquire_lock(cfg)

    if recover:
        return recover_run(cfg, conf, datasets)

    exists, where = spent_tag_exists(cfg, conf)
    scored = persisted(cfg, datasets)
    if exists and where == "origin":
        return continue_run(cfg, conf, datasets, scored)

    problems = preflight(cfg, conf, infra_retry)
    if not cfg.payload_checks_inside:
        problems += payload_checks(cfg, conf, datasets)      # the R1 switch, one line
    if problems:
        print("FINAL RUN REFUSED:\n  - " + "\n  - ".join(problems))
        return EXIT_REFUSED
    print("[score13] preflight clean (no protected payload was opened).")
    if preflight_only:
        return EXIT_OK

    freeze = _freeze_blob(cfg)
    begin_commit = spend_access(cfg, freeze["_sha256"])
    write_atomic(cfg.state_path, json.dumps(
        {"begin_commit": begin_commit, "registry_sha256": sha256_file(cfg.registry_path),
         "freeze_sha256": freeze["_sha256"], "datasets": datasets, "started": utcnow()}, indent=1))

    if cfg.payload_checks_inside:
        problems = payload_checks(cfg, conf, datasets)
        if problems:
            raise SystemExit("PAYLOAD CHECKS FAILED after the access was spent:\n  - "
                             + "\n  - ".join(problems))
    candidate, scored = score_all(cfg, conf, datasets, begin_commit, freeze, scored)
    return finalize(cfg, conf, scored, candidate, begin_commit, freeze["_sha256"], "full")


def _continuation_problems(cfg, conf, state):
    problems = []
    head = access13.sh(cfg, "git", "rev-parse", "HEAD")
    if head != state.get("begin_commit"):
        problems.append(f"HEAD {head[:12]} is not the FINAL-RUN-BEGIN commit "
                        f"{str(state.get('begin_commit'))[:12]}; a continuation is the same code.")
    live = sha256_file(cfg.registry_path)
    if live != state.get("registry_sha256"):
        problems.append(f"the live registry sha {live[:12]} is not the one persisted with the "
                        f"scores {str(state.get('registry_sha256'))[:12]}.")
    allowed = tuple(cfg.allowed_drift) + (
        str(Path(cfg.ledger_path).relative_to(cfg.repo)),
        str(Path(cfg.scores_dir).relative_to(cfg.repo)),
        str(Path(cfg.result_path).relative_to(cfg.repo)))
    stray = [ln[3:] for ln in access13.sh_raw(cfg, "git", "status", "--porcelain").splitlines()
             if ln and not ln[3:].startswith(allowed)]
    if stray:
        problems.append(f"undeclared output drift since the BEGIN commit: {sorted(stray)[:8]}")
    return problems


def continue_run(cfg, conf, datasets, scored):
    """POST-TAG CONTINUATION (`reserved._six_crash`): open ONLY the unscored datasets."""
    if not scored:
        print(f"ACCESS SPENT, NO SCORES. {cfg.spent_tag} is on origin but no per-dataset score "
              f"file exists: the run died between spending the access and the first durable "
              f"write. There is no persisted registry sha to authenticate a continuation against, "
              f"so no decision can be established and none may be invented. This is a documented "
              f"LOSS of the six-set access — disclose it and stop.")
        return EXIT_ACCESS_LOST
    if not cfg.state_path.exists():
        print(f"REFUSED: {cfg.state_path} is missing; a continuation cannot be authenticated.")
        return EXIT_REFUSED
    state = json.loads(cfg.state_path.read_text())
    problems = _continuation_problems(cfg, conf, state)
    for ds, p in scored.items():
        if p.get("registry_sha256") != state.get("registry_sha256"):
            problems.append(f"`{ds}` was scored under registry {str(p.get('registry_sha256'))[:12]}")
    if problems:
        print("CONTINUATION REFUSED:\n  - " + "\n  - ".join(problems))
        return EXIT_REFUSED
    remaining = [ds for ds in datasets if ds not in scored]
    if not remaining:
        print("[score13] every dataset is already scored; finalizing from the persisted scores.")
    else:
        print(f"[score13] post-tag continuation: opening ONLY {remaining} "
              f"({sorted(scored)} already persisted and will NOT be re-opened).")
    freeze = _freeze_blob(cfg)
    if freeze["_sha256"] != state.get("freeze_sha256"):
        print("CONTINUATION REFUSED: the frozen checkpoint is not the one the run began with.")
        return EXIT_REFUSED
    candidate, scored = score_all(cfg, conf, datasets, state["begin_commit"], freeze, scored)
    return finalize(cfg, conf, scored, candidate, state["begin_commit"], freeze["_sha256"],
                    "continuation")


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
              f"the whole registered partition; run the continuation first.")
        return EXIT_REFUSED
    live = sha256_file(cfg.registry_path)
    drifted = sorted({p.get("registry_sha256") for p in scored.values()} - {live})
    if drifted:
        print(f"--recover REFUSED: the scores were produced under registry "
              f"{[str(d)[:12] for d in drifted]}, the live registry is {live[:12]}. The decision "
              f"constants moved between the run and the recovery.")
        return EXIT_REFUSED
    freeze = _freeze_blob(cfg)
    state = json.loads(cfg.state_path.read_text()) if cfg.state_path.exists() else {}
    if state.get("freeze_sha256") not in (None, freeze["_sha256"]):
        print("--recover REFUSED: the result was produced under a different frozen checkpoint than "
              "the freeze file names.")
        return EXIT_REFUSED
    candidate = (conf["contrasts"]["C1"]["a"] if cfg.decision_layer == "m9"
                 else conf["conjuncts"][conf["sequence"]["order"][0]]["a"])
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
