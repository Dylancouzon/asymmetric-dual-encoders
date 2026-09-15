# M13-to-M14 conditional reserved execution pin

Pinned before any reserved query/qrel access. The M10 six-set result triggered the registered
conditional batch. Owner ruling R19 assigns the A100 execution to M14: M13 delivers this tested,
runnable implementation but must not resume the A100, create `m8-reserved-spent`, or open a
reserved query/qrel payload. This ownership change does not alter a six-set number, conjunct,
threshold, direction, partition or release rule.

**Terminal integration note:** `origin/main` carries a later owner instruction requiring M14 to
add dense constella-zero to this descriptive transaction. This document pins the tested original
three-system base. Before protected access, M14 must push the dated registry/ledger amendment,
extend and test the executor/controller for Zero, and make Zero reuse Nano's Stella document
vectors. Do not run the base controller unchanged.

## Exact scope

- Datasets, in reporting order: FEVER, dbpedia-entity, cqadup-android, cqadup-english.
- M13 base systems: `nano-dense`, `bge-small-en-v1.5`, `leaf-ir-asym`. M14's pre-access amendment
  adds dense `constella-zero` as the fourth system.
- Nano documents: `NovaSearch/stella_en_400M_v5` at
  `ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`; queries: the frozen M13 checkpoint.
- BGE documents and queries: `BAAI/bge-small-en-v1.5` at
  `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
- LEAF documents: `Snowflake/snowflake-arctic-embed-m-v1.5` at
  `e58a8f756156a1293d763f17e3aae643474e9b8a`; queries: `MongoDB/mdbr-leaf-ir` at
  `4262131b32c3182bd06e67e92ae69d7bd66e0c5c`.
- Document inference is fp32 on CUDA and normalized output is stored fp16: the M8 confirmatory
  contract and M4 convention the frozen comparators used. BGE and LEAF queries are fp32 on CUDA;
  Nano queries use its existing fp32 CPU scoring path because that model deliberately enables
  bf16 autocast on CUDA. TF32 matmul is disabled. Search promotes fp16 document blocks to fp32 and
  is exact over the whole corpus. Top 100 is sufficient for nDCG@10 and matches the final-six
  evaluator.

## Two separated contact classes

`m8src/pre_encode.py` claims the already-registered corpus-only allowlist entry. It may load only
the public `corpus` configuration, authenticates document count/IDs/text against
`results/eval_manifest.json`, and writes 50,000-row, hash-recorded shards. It cannot open a query,
qrel, frozen payload, or produce a ranking. A crash resumes only at a missing shard.

`m13src/score13.py --reserved-only` claims its registered capability at entry, then re-hashes every
document shard and checks at least 120 GB free without opening a protected payload. It commits and
pushes `RESERVED-RUN-BEGIN`, pushes the inherited durable receipt `m8-reserved-spent`, and only
then performs the first protected read. Each system scores all four datasets and lands as one
atomic JSON file. A crash after the tag resumes at the first missing system under the identical
tagged code/registry/pre-encode identity; a complete system is accepted only after its transaction,
payload, score and cache identities revalidate. No partial aggregate report is emitted.

## Descriptive report

The report reads the registry verbatim: B=10,000, seed 902, `inverted_cdf`, shared per-dataset
resample indices across R1 and R2. R1 is nano minus BGE; R2 is nano minus LEAF. The primary NDO-3
weights are 0.50 dbpedia-entity, 0.25 android, 0.25 english. It also reports equal-weight,
query-pooled, per-dataset, and weight-renormalized leave-one-out cuts. FEVER is explicitly a
double-contaminated sensitivity row. Every result is zero alpha, descriptive only, with no gate.

On completion the prior `INCOMPLETE_RESERVED` six-set record is preserved by its recorded SHA and
Git history, `results/m13_reserved_run.json` becomes the completion receipt, and
`results/m10_final_run.json.end_status` becomes `COMPLETE` without changing its six-set decision.

## M14 execution and safety

`scripts/m13_reserved_cloud.py` uses the retained 500 GB A100 pod, refuses a changed price or
uncovered wallet, runs the corpus pre-encode and protected scorer serially, pulls the durable
result, and sends STOP in `finally`. Before the spent tag, it loads and exercises all three query
towers on a fixed non-benchmark sentence, so model availability and shape are established without
contacting any reserved payload. That preflight and the protected scorer run with Hub/Transformers
offline; the constructed Nano dependency must reproduce the tokenizer/backbone-config hash frozen
in the pushed LoTTE manifest, closing the legacy unpinned `Nano10` constructor's network route. It
never terminates a pod or volume. The conservative cap remains 55.2 hours at at most
$1.6636111111/hour including storage.

M14 must use the pushed terminal M13 implementation and complete this batch before publication.
Until then, preserve `results/m10_final_run.json.end_status` as `INCOMPLETE_RESERVED`; do not
invent a result or relabel the deferred execution as complete.

Before this pin: guarded real-corpus metadata preflight passed all four exact document counts;
five reserved-support tests passed (dependency identity, cross-shard reads, known deltas,
mismatched-query refusal and saved-output authentication); Python compilation passed. All 36 M8
guard checks pass, including corpus/query/qrel separation and both historical `work/dev` aliases
when their duplicate protected payloads are intentionally absent from this worktree.
