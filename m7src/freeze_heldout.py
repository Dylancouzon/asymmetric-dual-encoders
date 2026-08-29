"""Pin the two held-out dev components into the dev manifest (Codex review #3 BLOCKER 2).

The four text-backed dev components were hash-pinned before any candidate result existed. The two
held-out slices were not: `freeze_m7_assets.py` iterates `devsuite.COMPONENTS` only, and
`dev_eval.dev_components()` appended `heldout-*` according to whether their JSON happened to exist.
Both files are DERIVED (`heldout._build()` regenerates them from the current training mix and doc
pool), so a changed mix, a changed pool order, or a deleted file would silently change the
selection statistic while `freeze.py` still verified the same manifest hash.

This closes that hole, and it is necessarily a LATE repair: the two components were
deterministically defined but not cryptographically pinned before the lever selections that used
them. That disclosure lives in m7/LEDGER.md and must appear in the report.

What gets pinned, per the review:
  - ordered qids, query texts, qrels (positive pool-row indices), and long-query membership;
  - the raw bytes of both held-out JSONs;
  - the ordered corpus identity: the pool's per-store id hashes, spans, counts and encoder;
  - the vector-cache identity: the pool vector file's size and content hash (the corpus IS those
    vectors -- the held-out slices carry no document text);
  - an explicit SIX-name component list, which `dev_eval.dev_components()` now aborts on rather
    than silently shrinking.

Run: freeze_heldout.py [--no-pool-hash]   (the pool hash reads 12.6 GB; skip only for a dry run)
"""
import hashlib
import json
import sys
import time

import devsuite
import encoders
import heldout
import pool as poolmod
from _paths import REPO, WORK
from hashing import sha, sha_stream_list

MANIFEST = REPO / "results" / "m7_dev_manifest.json"
FROZEN = REPO / "results" / "frozen_eval"
COMPONENTS = list(devsuite.COMPONENTS) + list(heldout.COMPONENTS)


def sha_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def heldout_entry(name):
    p = heldout.HELD / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"{p} missing -- build it with run_stage0c.sh before pinning")
    b = json.loads(p.read_text())
    q_ids, q_texts = b["q_ids"], b["q_texts"]
    return {
        "n_docs": b["n_docs"], "n_queries": len(q_ids),
        "corpus": "full-pool",
        # ORDERED, not sorted: the evaluator consumes q_ids in file order, and a reordering that
        # left the sorted hash intact would still reorder the query matrix.
        "qids_ordered_sha256": sha_stream_list(q_ids),
        "qids_sha256": sha(sorted(q_ids)),          # same convention as the text-backed entries
        "qtexts_ordered_sha256": sha_stream_list(q_texts),
        "qrels_sha256": sha(b["qrels"]),
        "n_tokens_sha256": sha_stream_list(str(n) for n in b["n_tokens"]),
        "by_source": b["by_source"],
        "json_sha256": sha_file(p),
        "construction": (
            "held-out training slice: sha256(qid) mod 50 == 0 at QUERY granularity (m7src/"
            "trainmix.heldout), corpus = the entire frozen doc pool, qrels = pool row indices"
            + ("; restricted to queries of >= 64 WordPiece tokens. SUBSET of heldout-train: the "
               "same 55 qids with identical corpus and qrels, hence identical per-query nDCG -- "
               "the dependence the macro must account for (boot.signflip_dep/paired_dep)."
               if name == "heldout-longq" else ".")),
    }


def assert_structure():
    """Prove the properties the manifest would otherwise merely record (Codex review #3b B1).

    Recording `n_shared_qids` next to a claim of nesting is not the same as checking the nesting,
    and a component pair that silently stopped nesting would make the dependence-preserving
    statistics wrong in the direction of over-confidence -- the exact failure being repaired."""
    tr = json.loads((heldout.HELD / "heldout-train.json").read_text())
    lq = json.loads((heldout.HELD / "heldout-longq.json").read_text())
    for name, b in (("heldout-train", tr), ("heldout-longq", lq)):
        n = len(b["q_ids"])
        if len(set(b["q_ids"])) != n:
            raise SystemExit(f"{name}: duplicate qids")
        if not (len(b["q_texts"]) == len(b["n_tokens"]) == n):
            raise SystemExit(f"{name}: q_ids/q_texts/n_tokens lengths disagree")
        if set(b["qrels"]) != set(b["q_ids"]):
            raise SystemExit(f"{name}: qrels keys do not match q_ids")
        if not all(b["qrels"][q] for q in b["q_ids"]):
            raise SystemExit(f"{name}: some query has no positive")
    if not set(lq["q_ids"]) <= set(tr["q_ids"]):
        raise SystemExit("heldout-longq is not a subset of heldout-train; the nesting the "
                         "dependence-preserving statistics assume does not hold")
    if lq["n_docs"] != tr["n_docs"]:
        raise SystemExit("the two held-out components address different corpora")
    t_text = dict(zip(tr["q_ids"], tr["q_texts"]))
    t_tok = dict(zip(tr["q_ids"], tr["n_tokens"]))
    for q, txt, ntk in zip(lq["q_ids"], lq["q_texts"], lq["n_tokens"]):
        if t_text[q] != txt or t_tok[q] != ntk or tr["qrels"][q] != lq["qrels"][q]:
            raise SystemExit(f"shared qid {q} differs between the two held-out components")
    if any(n < heldout.LONG_TOKENS for n in lq["n_tokens"]):
        raise SystemExit(f"heldout-longq contains a query under {heldout.LONG_TOKENS} tokens")
    n_long = sum(1 for n in tr["n_tokens"] if n >= heldout.LONG_TOKENS)
    if n_long != len(lq["q_ids"]):
        raise SystemExit(f"heldout-longq has {len(lq['q_ids'])} queries but heldout-train has "
                         f"{n_long} at >= {heldout.LONG_TOKENS} tokens")
    print(f"  structure OK: {len(tr['q_ids']):,} held-out queries, {len(lq['q_ids'])} of them long "
          f"and byte-identical in both components", flush=True)
    return sorted(lq["q_ids"])


def main(pool_hash=True):
    t0 = time.time()
    spec = encoders.active()
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for c in devsuite.COMPONENTS:
        if c not in man:
            raise SystemExit(f"dev manifest is missing text-backed component {c}; run devsuite.py")

    if not pool_hash:
        raise SystemExit(
            "refusing to write an authoritative pin without the pool-vector hash: the corpus of "
            "both held-out components IS that file, so a pin that does not bind its bytes pins "
            "nothing that matters (Codex review #3b BLOCKER 1). Use --dry-run to inspect.")
    nested = assert_structure()
    _, pool_vecs, pmeta = poolmod.build()
    vec_p = poolmod.VEC_DIR / "vecs.f16"
    pool_id = {"n": pmeta["n"], "dim": pmeta["dim"], "encoder": pmeta["encoder"],
               "encoder_repo": pmeta.get("encoder_repo"),
               "encoder_revision": pmeta.get("encoder_revision"),
               "stores": pmeta["stores"], "spans": pmeta["spans"], "counts": pmeta["counts"],
               "store_id_sha256": pmeta["id_sha256"],
               "vectors_path": str(vec_p.relative_to(WORK)),
               "vectors_bytes": vec_p.stat().st_size}
    if pool_hash:
        print(f"  hashing {vec_p.stat().st_size/1e9:.2f} GB of pool vectors ...", flush=True)
        pool_id["vectors_sha256"] = sha_file(vec_p)
        print(f"  pool vectors sha256 {pool_id['vectors_sha256'][:16]}... "
              f"({time.time()-t0:.0f}s)", flush=True)

    for name in heldout.COMPONENTS:
        e = heldout_entry(name)
        man[name] = e
        b = json.loads((heldout.HELD / f"{name}.json").read_text())
        (FROZEN / f"dev-{name}.json").write_text(json.dumps(
            {"queries": dict(zip(b["q_ids"], b["q_texts"])), "qrels": b["qrels"],
             "q_ids_ordered": b["q_ids"], "n_tokens": b["n_tokens"]}))
        print(f"dev {name:16s} {e['n_docs']:>9,} docs {e['n_queries']:>6,} queries  "
              f"json {e['json_sha256'][:16]}", flush=True)

    man["_pinned"] = {
        "components": COMPONENTS,
        "n_components": len(COMPONENTS),
        "macro": "equal weight per component (instructions-m7.md)",
        "abort_on_missing": "dev_eval.dev_components() raises if any of these is unavailable or "
                            "hash-mismatched; it must never silently shrink the suite",
        "pool": pool_id,
        "active_encoder": {"name": spec.name, "repo": spec.repo, "revision": spec.revision,
                           "dim": spec.dim, "pooling": spec.pooling},
        "nested_components": {"heldout-longq": {
            "subset_of": "heldout-train", "n_shared_qids": len(nested),
            "shared_qids_sha256": sha(nested),
            "verified": "subset membership, identical query text / n_tokens / qrels on every "
                        "shared qid, same corpus, and exactly the >= 64-token queries "
                        "(freeze_heldout.assert_structure)"}},
        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "disclosure": "LATE PIN: the two held-out components were deterministically defined but "
                      "not cryptographically pinned before the lever selections that used them "
                      "(Codex review #3 BLOCKER 2). The four text-backed components were pinned "
                      "before any candidate result. Report must say so.",
    }
    MANIFEST.write_text(json.dumps(man, indent=1))
    print(f"pinned {len(COMPONENTS)} dev components into {MANIFEST.name} "
          f"({len(nested)} qids shared between the two held-out slices), {time.time()-t0:.0f}s")


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        assert_structure()
        for n in heldout.COMPONENTS:
            print(json.dumps({n: heldout_entry(n)}, indent=1))
    else:
        main(pool_hash="--no-pool-hash" not in sys.argv)
