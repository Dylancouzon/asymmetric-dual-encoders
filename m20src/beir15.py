#!/usr/bin/env python3
"""M20's broad BEIR-15 descriptive validation. Eight systems, fifteen public BEIR datasets.

Registered pre-observation in `m20/REGISTRATION.md` and `m20/beir15_registry.json` under owner
ruling R22.  **Descriptive, `alpha = 0`.**  It is not a gate, it produces no estimand, and it never
enters or reinterprets M13's registered six-set inference.

**It must never open a reserved payload.**  Two independent reasons it cannot:

  * The reserved four are not scored here at all.  Their rows are COPIED from the completed atomic
    outputs of the tagged transaction, which is the only place they are ever read.
  * This process installs `m8src/paths_guard.py` holding only the corpus-only entry, so a reserved
    query or qrel file, HF cache directory or loader call raises at open time rather than being
    discovered afterwards in a log.

Per-(dataset, system) outputs are atomic: a crash resumes at the first missing pair.  BEIR-15 is
not protected access, so a resume here costs nothing but time.

  PYTHONPATH=m20src:m13src:m12src:m8src:m7src:bench M7_ENCODER=stella-400M-v5 \\
      .venv/bin/python -u m20src/beir15.py --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in ("bench", "m7src", "m8src", "m10src", "m11/release", "m12src", "m13src", "m20src"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

import pre_encode as P            # claims the corpus-only guard entry and installs the guard
import roster as R                # noqa: E402

SCORES = REPO / "results" / "m20_beir15_scores"
RESULT = REPO / "results" / "m20_beir15_run.json"
RESERVED_OUTPUTS = REPO / "results" / "m10_final_scores" / "reserved"
RESERVED_CORPORA = set(P.RESERVED_DATASETS)
RUN_STAGE = "beir15"


def registry():
    return R.registration()


def dataset_rows():
    """The fifteen registered BEIR rows, each with the corpora that make it up."""
    reg = registry()
    rows = []
    for row in reg["datasets"]:
        if row["key"] == "cqadupstack":
            members = [f"cqadup-{forum}" for forum in reg["cqadupstack"]["forums"]]
            rows.append({**row, "members": members,
                         "aggregation": reg["cqadupstack"]["aggregation"]})
        else:
            rows.append({**row, "members": [row["key"]],
                         "aggregation": "mean of the per-query nDCG@10 values"})
    return rows


def load_public(corpus_name):
    """Corpus, queries and test qrels for one PUBLIC corpus, at its pinned revision.

    Refuses a reserved corpus by name before anything is opened.  The guard would refuse the
    queries and qrels anyway; this makes the intent explicit at the call site.
    """
    if corpus_name in RESERVED_CORPORA:
        raise ValueError(f"{corpus_name} is reserved; its rows are copied from the tagged "
                         f"transaction and are never re-read here")
    from datasets import load_dataset

    source, revision = P.corpus_sources()[corpus_name]
    corpus, _source, _revision = P.load_corpus(corpus_name)
    P.authenticate_corpus(corpus_name, corpus, revision)
    doc_ids = [str(value) for value in corpus["_id"]]
    doc_texts = [P._doc_text(row) for row in corpus]
    queries = load_dataset(source, "queries", revision=revision)["queries"]
    split = "dev" if corpus_name == "msmarco" else "test"
    if corpus_name.startswith("cqadup-"):
        # CQADupStack keeps its labels in the forum's own repository, already pinned above.
        qrels_source, qrels_revision = source, revision
        rows = load_dataset(source, "default", revision=revision, split=split)
    else:
        # BEIR publishes labels in a SEPARATE repository with its own revision. Leaving it
        # unpinned would let a republished label set move a reported number silently.
        row = next(r for r in registry()["datasets"] if r["key"] == corpus_name)
        qrels_source, qrels_revision = row["qrels_source"], row["qrels_revision"]
        rows = load_dataset(qrels_source, revision=qrels_revision, split=split)
    qrels = {}
    for row in rows:
        qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    # BEIR convention: the queries file spans every split, so keep only queries with qrels here.
    q_ids, q_texts = [], []
    for qid, text in zip(queries["_id"], queries["text"]):
        if str(qid) in qrels:
            q_ids.append(str(qid))
            q_texts.append(text)
    qrels = {qid: qrels[qid] for qid in q_ids}
    if not q_ids:
        raise RuntimeError(f"{corpus_name}: no queries carry {split} qrels")
    return {"doc_ids": doc_ids, "doc_texts": doc_texts, "q_ids": q_ids, "q_texts": q_texts,
            "qrels": qrels, "source": source, "revision": revision, "split": split,
            "qrels_source": qrels_source, "qrels_revision": qrels_revision}


def shards_for(system, corpus_name):
    """Document vectors for one system, with every shard re-hashed against its manifest.

    verify=True is not optional: the shard bytes are the document identity behind every number in
    this table, and re-hashing them costs about a minute per full pass against a run measured in
    days."""
    import reserved_support as RS

    return RS.cache_for(system, corpus_name, repo=REPO, verify=True)


def score_path(corpus_name, system):
    return SCORES / corpus_name / f"{R.slug(system)}.json"


def _write(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(blob, indent=2) + "\n")
    tmp.replace(path)


def run_corpus(corpus_name, encoders, device="cuda", chunk=50_000):
    """Score every system on one public corpus, resuming at the first missing (corpus, system)."""
    from evalkit import topk_ids_scores

    pending = [s for s in R.SYSTEMS if not score_path(corpus_name, s).exists()]
    if not pending:
        print(f"[beir15] {corpus_name}: already complete", flush=True)
        return
    started = time.time()
    payload = load_public(corpus_name)
    q_ids, qrels = payload["q_ids"], payload["qrels"]
    print(f"[beir15] {corpus_name}: {len(payload['doc_ids']):,} docs, {len(q_ids):,} queries, "
          f"pending {pending}", flush=True)
    run_root = REPO / "work" / "m20-runs"

    def finish(system, scores, extra):
        for qid in q_ids:
            scores.setdefault(qid, 0.0)
        values = np.asarray([scores[q] for q in q_ids], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise RuntimeError(f"{corpus_name}/{system}: nDCG outside [0, 1]")
        _write(score_path(corpus_name, system), {
            "status": "COMPLETE", "dataset": corpus_name, "system": system,
            "registry_sha256": R.sha_file(REPO / "m20" / "beir15_registry.json"),
            "source": payload["source"], "revision": payload["revision"],
            "qrels_source": payload["qrels_source"], "qrels_revision": payload["qrels_revision"],
            "split": payload["split"], "n_docs": len(payload["doc_ids"]),
            "n_queries": len(q_ids), "mean_ndcg10": float(values.mean()),
            "scores": {q: scores[q] for q in q_ids}, **extra})
        print(f"[beir15] {corpus_name}/{system}: {values.mean():.6f}", flush=True)

    def producer_run_ready(system):
        """A completed producer's persisted run, if it is still exactly what it recorded.

        Re-running a producer whose row is already scored would rewrite the very file the scored
        row is supposed to authenticate. Reuse it instead, and recompute only when it is gone.
        """
        path = R.run_path(run_root, RUN_STAGE, system, corpus_name)
        score_file = score_path(corpus_name, system)
        if system in pending or not score_file.exists() or not path.exists():
            return None
        recorded = json.loads(score_file.read_text()).get("run_sha256")
        if not recorded or R.sha_file(path) != recorded:
            raise RuntimeError(f"{corpus_name}/{system}: persisted run does not match the hash its "
                               f"scored row recorded; delete the row to rescore it deliberately")
        return recorded

    derived_pending = [s for s in R.DERIVED_SYSTEMS if s in pending]
    needed_producers = {key for s in derived_pending for key in R.DERIVED_SYSTEMS[s]}

    # BM25 first: it is the only system that needs the document TEXT, which is the largest object
    # in this function, and freeing it before the dense passes keeps the peak down.
    if "bm25" in pending or ("bm25" in needed_producers and producer_run_ready("bm25") is None):
        import fusion

        run = R.bm25_run(payload["doc_ids"], payload["doc_texts"], q_ids, payload["q_texts"])
        path = R.run_path(run_root, RUN_STAGE, "bm25", corpus_name)
        digest = R.save_run(path, R.truncate(run), q_ids)
        if "bm25" in pending:
            finish("bm25", R.per_query_ndcg10(run, qrels),
                   {"config": fusion.BM25_CONFIG, "depth": fusion.DEPTH,
                    "package_versions": R.bm25_versions(),
                    "empty_runs": int(sum(1 for q in q_ids if not run.get(q))),
                    "run_path": str(path.relative_to(REPO)), "run_sha256": digest})
        else:
            recorded = json.loads(score_path(corpus_name, "bm25").read_text()).get("run_sha256")
            if recorded and digest != recorded:
                raise RuntimeError(f"{corpus_name}/bm25: rebuilt run does not reproduce the hash "
                                   f"its completed row recorded")
        del run
    payload["doc_texts"] = None

    for system in R.DENSE_SYSTEMS:
        if system not in pending:
            if system not in needed_producers or producer_run_ready(system) is not None:
                continue
        doc_ids, doc_vectors = shards_for(system, corpus_name)
        if doc_ids != payload["doc_ids"]:
            raise RuntimeError(f"{corpus_name}/{system}: shard document order differs from the "
                               f"corpus just loaded")
        qvectors = encoders[system].encode(payload["q_texts"])
        if qvectors.shape[1] != doc_vectors.shape[1]:
            raise RuntimeError(f"{corpus_name}/{system}: query/doc dimensions differ")
        run = topk_ids_scores(qvectors, doc_vectors, doc_ids, k=R.DENSE_DEPTH, chunk=chunk,
                              device=device, qids=q_ids)
        extra = {"document_cache_manifest": str(doc_vectors.manifest_path.relative_to(REPO)),
                 "query_encoder": encoders[system].identity}
        if system in R.RUN_PRODUCERS:
            path = R.run_path(run_root, RUN_STAGE, system, corpus_name)
            extra["run_path"] = str(path.relative_to(REPO))
            extra["run_sha256"] = R.save_run(path, R.truncate(run), q_ids)
        if system in pending:
            finish(system, R.per_query_ndcg10(run, qrels), extra)
        elif extra.get("run_sha256"):
            recorded = json.loads(score_path(corpus_name, system).read_text()).get("run_sha256")
            if recorded and extra["run_sha256"] != recorded:
                raise RuntimeError(f"{corpus_name}/{system}: rebuilt run does not reproduce the "
                                   f"hash its completed row recorded")
        del run, doc_vectors

    for system in R.DERIVED_SYSTEMS:
        if system not in pending:
            continue
        for needed in R.DERIVED_SYSTEMS[system]:
            if not score_path(corpus_name, needed).exists():
                raise RuntimeError(f"{corpus_name}/{system}: input {needed} is not scored")
        dense_key, lexical_key = R.DERIVED_SYSTEMS[system]
        runs, inputs = {}, {}
        for key in (dense_key, lexical_key):
            path = R.run_path(run_root, RUN_STAGE, key, corpus_name)
            # Bind to the hash the INPUT's own score row recorded, not to whatever is on disk now.
            # On a resume where the input was scored by an earlier process, that is the only thing
            # standing between a corrupted or stale run file and a silently wrong fused number.
            recorded = json.loads(score_path(corpus_name, key).read_text()).get("run_sha256")
            if not recorded:
                raise RuntimeError(f"{corpus_name}/{key}: scored row carries no run hash; "
                                   f"{system} cannot be derived from it")
            runs[key], _ = R.load_run(path, recorded)
            inputs[key] = {"run_path": str(path.relative_to(REPO)), "run_sha256": recorded}
        fused = R.dbsf_at_depth(runs[dense_key], runs[lexical_key])
        finish(system, R.per_query_ndcg10(fused, qrels),
               {"inputs": inputs,
                "operator": "m12src/qfusion.py:dbsf over both prefetches truncated to 100 first"})
    print(f"[beir15] {corpus_name}: done in {time.time() - started:.0f}s", flush=True)


def reserved_scores(corpus_name, system):
    """One reserved row, copied from the tagged transaction's completed atomic output."""
    path = RESERVED_OUTPUTS / f"{R.slug(system)}.json"
    record = json.loads(path.read_text())
    if record.get("status") != "COMPLETE":
        raise RuntimeError(f"{path}: reserved output is not complete")
    row = record["datasets"][corpus_name]
    return {"status": "COMPLETE", "dataset": corpus_name, "system": system,
            "origin": "copied from the tagged reserved transaction; not re-scored here",
            "reserved_output": str(path.relative_to(REPO)),
            "reserved_output_sha256": R.sha_file(path),
            "n_queries": row["n_queries"], "mean_ndcg10": row["mean_ndcg10"],
            "scores": row["scores"]}


def assemble(out_path=RESULT):
    """Build the descriptive table from every persisted per-(dataset, system) row."""
    rows = dataset_rows()
    registry_sha = R.sha_file(REPO / "m20" / "beir15_registry.json")
    per_corpus, missing = {}, []
    for row in rows:
        for member in row["members"]:
            per_corpus[member] = {}
            for system in R.SYSTEMS:
                if member in RESERVED_CORPORA:
                    try:
                        per_corpus[member][system] = reserved_scores(member, system)
                    except (FileNotFoundError, KeyError):
                        missing.append(f"{member}/{system} (reserved)")
                    continue
                path = score_path(member, system)
                if not path.exists():
                    missing.append(f"{member}/{system}")
                    continue
                blob = json.loads(path.read_text())
                if blob.get("status") != "COMPLETE" or blob.get("system") != system \
                        or blob.get("dataset") != member:
                    raise RuntimeError(f"{path}: incomplete or mislabelled scored row")
                if blob.get("registry_sha256") != registry_sha:
                    raise RuntimeError(f"{path}: scored against a different registration")
                per_corpus[member][system] = blob
    # Every system on one corpus must have scored the same query set, or the per-dataset means
    # are not comparable and the fused rows are not derived from their own inputs' queries.
    for member, systems in per_corpus.items():
        sets = {system: sorted(blob["scores"]) for system, blob in systems.items()
                if "scores" in blob}
        if sets and len({tuple(v) for v in sets.values()}) != 1:
            raise RuntimeError(f"{member}: systems scored different query sets")
    table = {}
    for row in rows:
        table[row["key"]] = {"contact": row["contact"], "aggregation": row["aggregation"],
                             "licence_role": row.get("licence_role"),
                             "members": row["members"], "systems": {}, "per_member": {}}
        for system in R.SYSTEMS:
            means = [per_corpus[m][system]["mean_ndcg10"] for m in row["members"]
                     if system in per_corpus.get(m, {})]
            if len(means) != len(row["members"]):
                continue
            table[row["key"]]["systems"][system] = float(np.mean(means))
            table[row["key"]]["per_member"][system] = {
                m: per_corpus[m][system]["mean_ndcg10"] for m in row["members"]}
    record = {
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "scope": "Broad descriptive BEIR-15 validation. alpha = 0. No gate, no threshold, no "
                 "superiority or equivalence claim. It does not enter or reinterpret M13's "
                 "registered six-set inference.",
        "authority": registry()["authority"],
        "registry_sha256": R.sha_file(REPO / "m20" / "beir15_registry.json"),
        "systems": list(R.SYSTEMS),
        "datasets": [row["key"] for row in rows],
        "reserved_rows_copied_from": str(RESERVED_OUTPUTS.relative_to(REPO)),
        "licence_disclosures": {row["key"]: row["licence_note"] for row in rows
                                if row.get("licence_note")},
        "table": table,
        "missing": missing,
        "per_query_rows": str(SCORES.relative_to(REPO)),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write(Path(out_path), record)
    print(json.dumps({"status": record["status"], "missing": len(missing)}))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=50_000)
    parser.add_argument("--corpora", nargs="*", default=None)
    parser.add_argument("--assemble-only", action="store_true")
    args = parser.parse_args(argv)

    R.assert_registered_identities()
    if args.assemble_only:
        record = assemble()
        return 0 if record["status"] == "COMPLETE" else 1

    corpora = args.corpora or list(P.beir15_encode_datasets())
    bad = [name for name in corpora if name in RESERVED_CORPORA]
    if bad:
        raise SystemExit(f"refusing reserved corpora {bad}: their rows come from the tagged "
                         f"transaction")
    # The five query towers together are about 2.2 GB of fp32 weights. Constructing them once and
    # reusing them across corpora avoids 22 reloads; on a small card, pass --corpora to run in
    # separate processes instead (the tower-rate benchmark shows what resident models cost there).
    encoders = {system: R.QueryEncoder(system, cfg=_nano_cfg(), device=args.device)
                for system in R.DENSE_SYSTEMS}
    try:
        for corpus_name in corpora:
            run_corpus(corpus_name, encoders, device=args.device, chunk=args.chunk)
    finally:
        for encoder in encoders.values():
            encoder.release()
    record = assemble()
    return 0 if record["status"] == "COMPLETE" else 1


def _nano_cfg():
    import access13 as A

    return A.production()


if __name__ == "__main__":
    raise SystemExit(main())
