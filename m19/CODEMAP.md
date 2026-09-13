# M19 code and artifact map

| Path | Purpose |
|---|---|
| `m19/registry.json` | prospective constants, gates, versions and execution state |
| `m19/inheritance-lock.json` | generated binding of admitted M18 manifests and local inherited bytes |
| `m19/STATUS.md`, `PLANNING.md`, `LEDGER.md`, `FINDINGS.md` | compact state and durable record |
| `m19src/common.py` | M19 paths, strict read/write admission, hashes and atomic writes |
| `m19src/inherit.py` | verify and publish the M18/Zero-v1 inheritance lock |
| `m19src/term_inventory.py` | consumed-byte-bound AddedToken support scan, source qualification and resumable v2 roster lock |
| `m19src/zero.py` | deterministic V0/T0 row algebra, compact int8 table and no-eager-float loader |
| `m19src/build_candidate.py` | one-shot local Stella teacher receipt and actual V0/T0 bundle build |
| `m19src/benchmark_serving.py` | fixed-sequence v1/T0 encoder and actual-index hybrid latency gates |
| `m19src/pool_pilot.py` | frozen ten-query seven-route pool construction and cost-only projection |
| `m19src/retrieval.py` | exact passage scoring, exclusion-before-truncation, artifact collapse and DBSF@100 |
| `m19src/metrics.py` | binary top-ten metrics, fixed-roster term macro and eligibility helpers |
| `m19src/judgments.py` | seven-route pool, blinded evidence packets, audit sampling/agreement and binary qrels freeze |
| `m19src/queries.py` | query shape/count/family/leakage validation and resumable hash-bound split sealing |
| `m19src/confirmation.py` | immutable, hash-bound one-shot confirmation transaction |
| `m19src/rehearse.py` | one-command synthetic pool/judgment/metric transaction and interrupted resume |
| `m19/term-roster-lock-v3.json`, `results/m19_term_inventory_v3.json` | final 12-term roster audit/evidence; v1/v2 are preserved and superseded |
| `m19/query-split-seal-v1.json` | text-free identities/counts for the sealed 60/30 development/confirmation split |
| `m19/pool-pilot-lock-v1.json` | pre-retrieval deterministic ten-query pool-cost selection |
| `m19/teacher-snapshot-lock-v1.json` | complete local Stella revision manifest including trusted code and tokenizer/config files |
| `results/m19_serving_benchmark_10000_v2.json`, `results/m19_serving_receipt_t0_v2.json` | superseding exact-route timing evidence and authenticated T0 serving gates; unversioned v1 is preserved but superseded |
| `results/m19_pool_pilot.json` | cost-only pilot counts, projections and hashes of the ignored blinded pool/packet |
| `m19src/test_common.py`, `test_inherit.py` | synthetic path-guard and lock tests |
| `m19/reviews/` | bounded prompts, reviewer identities, access logs, findings and dispositions |
| `work/m19/` | gitignored raw/derived/run state owned by M19 |
| `results/m19_*` | tracked compact evidence; never raw private query text |

M19 starts from M18 commit `bfa7257fe369a41ce048e150f861703a662a08f0`. The first fork is
`m18src/common.py`: M19 replaces its historical substring denylist with a narrow admitted-input
boundary, adds M18 confirmation refusal, and limits writes to M19-owned paths.
