# M20 pre-flight smoke — the real BEIR-15 path, end to end

Run 2026-09-16 on the local RTX 3080 box, before any cloud rental and before any protected read.
No reserved payload was opened; `m20src/beir15.py` runs with the corpus-only guard installed and
refuses a reserved corpus by name.

## What was exercised

The complete stage-C path on one real BEIR-15 row, SciFact (5,183 documents, 300 test queries):

1. `m8src/pre_encode.py --batch beir15 --corpora scifact` for all three document towers, at their
   pinned revisions, writing hash-recorded fp16 shards and pinning the corpus hashes in
   `results/m20_corpus_pins.json`.
2. `m20src/beir15.py --corpora scifact` scoring **all eight registered systems**, including the
   two DBSF rows derived from persisted top-100 runs, and assembling the descriptive table.

## Result: every independently published number reproduced

| system | M20 stage C | prior published value | source | delta |
|---|---:|---:|---|---:|
| `nano-dense` | 0.721097496 | 0.721097 | M13 six-set, `m14/HANDOFF.md` | +5e-07 |
| `zero-dense` | 0.610130735 | 0.610130735 | `m12/six_dbsf.json` `dense` | +1e-16 |
| `stella-query` | 0.779561883 | — | no prior published value | — |
| `bge-small-en-v1.5` | 0.712706925 | 0.712706 | six-set nano − 0.008391 | +9e-07 |
| `leaf-ir-asym` | 0.699013247 | 0.699013 | six-set nano − 0.022084 | +2e-07 |
| `bm25` | 0.679093836 | 0.679093836 | `m12/six_dbsf.json` `bm25` | 0 |
| `zero+bm25 dbsf@100` | 0.717300632 | 0.717300632 | `m12/six_dbsf.json` `dbsf@100` | 0 |
| `nano+bm25 dbsf@100` | 0.743325403 | — | no prior published value | — |

The three deltas near 1e-06 are the rounding of the six-decimal published table, not a
disagreement: the two full-precision comparisons available (M12's `dense` and `dbsf@100`) and BM25
match to machine precision.

Two rows have no prior published value because they are new in M20: the symmetric Stella teacher
and Nano fused with BM25.

## Separately: the registered DBSF reproduction prerequisite

`m20src/dbsf_reproduction.py` on SciFact and NFCorpus, receipt
`results/m20_dbsf_reproduction.json`. All three deltas exactly **0.000e+00**:

- extending the executor moved M12's `dbsf@100` by 0;
- substituting the released `constella-zero` int8 bundle for the frozen M7 table moved it by 0;
- retrieving dense at depth 100 rather than depth 1000 then truncating moved it by 0.

## What this does not establish

It is one small corpus on a consumer GPU. It does not measure the A100 encode rate, the memory
behaviour of BM25 on a 5-9M document corpus, the download or archive paths, or anything about the
reserved transaction, whose payload stays sealed. Those are exercised on the pod, in order, with
the projection gate watching the clock.

## Cleanup

The artifacts this smoke produced under `results/` and `work/m13-reserved-enc/*/scifact/` were
deleted afterwards, so the cloud session starts from an empty state and the whole table is
produced on one device. Mixing a 3080 row into an A100 table is exactly the silent inconsistency
this project avoids. The numbers above are the receipt.
