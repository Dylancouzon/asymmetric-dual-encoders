# Final results matrix (nDCG@10, brute-force exact search)

| system | scifact | nfcorpus | fiqa | arguana | scidocs | trec-covid | avg-5 | avg-6 |
|---|---|---|---|---|---|---|---|---|
| arctic-embed-m-v1.5 | 0.7159 | 0.3624 | 0.4241 | 0.5953 | 0.2149 | 0.8461 | 0.4625 | 0.5264 |
| leaf-ir-asym | 0.6990 | 0.3608 | 0.4165 | 0.5833 | 0.2034 | 0.8301 | 0.4526 | 0.5155 |
| mdbr-leaf-ir | 0.7056 | 0.3653 | 0.3917 | 0.5938 | 0.1977 | 0.8197 | 0.4508 | 0.5123 |
| bge-small-en-v1.5 | 0.7127 | 0.3430 | 0.4035 | 0.6034 | 0.2052 | 0.7575 | 0.4536 | 0.5042 |
| arctic-embed-s | 0.6991 | 0.3258 | 0.3968 | 0.5684 | 0.1942 | 0.8116 | 0.4369 | 0.4993 |
| granite-small-r2 | 0.7582 | 0.3709 | 0.4086 | 0.5442 | 0.2405 | 0.6455 | 0.4645 | 0.4947 |
| opensearch-doc-v3-gte | 0.7275 | 0.3602 | 0.4072 | 0.5241 | 0.1686 | 0.7331 | 0.4375 | 0.4868 |
| gte-small | 0.7270 | 0.3477 | 0.3931 | 0.5542 | 0.2139 | 0.6664 | 0.4472 | 0.4837 |
| lightretriever-qwen2.5-1.5b-hybrid | 0.6734 | 0.3012 | 0.4275 | 0.5351 | 0.1752 | 0.7197 | 0.4225 | 0.4720 |
| arctic-embed-xs | 0.6451 | 0.3085 | 0.3452 | 0.5206 | 0.1835 | 0.7943 | 0.4006 | 0.4662 |
| lightretriever-qwen2.5-1.5b-hybrid-websearch | 0.6703 | 0.2970 | 0.4270 | 0.5069 | 0.1710 | 0.6844 | 0.4144 | 0.4594 |
| lightretriever-qwen2.5-1.5b-dense-int8table | 0.6629 | 0.2861 | 0.4153 | 0.5189 | 0.1752 | 0.6933 | 0.4117 | 0.4586 |
| lightretriever-qwen2.5-1.5b-dense | 0.6627 | 0.2857 | 0.4149 | 0.5181 | 0.1756 | 0.6926 | 0.4114 | 0.4583 |
| e5-small-v2 | 0.6885 | 0.3235 | 0.3748 | 0.4180 | 0.1772 | 0.7426 | 0.3964 | 0.4541 |
| lightretriever-qwen2.5-1.5b-dense-websearch | 0.6480 | 0.2788 | 0.4099 | 0.4601 | 0.1656 | 0.6295 | 0.3925 | 0.4320 |
| all-MiniLM-L6-v2 | 0.6451 | 0.3159 | 0.3687 | 0.5017 | 0.2165 | 0.4725 | 0.4096 | 0.4201 |
| bm25 | 0.6791 | 0.3180 | 0.2532 | 0.4878 | 0.1565 | 0.6099 | 0.3789 | 0.4174 |
| lightretriever-qwen2.5-1.5b-sparse | 0.6306 | 0.2891 | 0.3479 | 0.3061 | 0.1459 | 0.6198 | 0.3439 | 0.3899 |
| potion-retrieval-32M | 0.6331 | 0.3073 | 0.1876 | 0.4488 | 0.1369 | 0.4469 | 0.3427 | 0.3601 |
| static-retrieval-mrl-en-v1 | 0.5935 | 0.3005 | 0.1999 | 0.4439 | 0.1322 | 0.4460 | 0.3340 | 0.3527 |
| potion-base-8M | 0.5064 | 0.2427 | 0.1662 | 0.4196 | 0.1239 | 0.4573 | 0.2917 | 0.3193 |

Systems without trec-covid runs (projection ablation, 5-ds only):
- potion-base-8M-proj-to-arctic-m: {'arguana': 0.4192, 'fiqa': 0.1721, 'nfcorpus': 0.2733, 'scidocs': 0.1149, 'scifact': 0.5205}
- potion-retrieval-32M-proj-to-arctic-m: {'arguana': 0.449, 'fiqa': 0.2207, 'nfcorpus': 0.3034, 'scidocs': 0.1169, 'scifact': 0.5379}

## Query-side costs (CPU, batch 1)

| system | latency ms | load s | artifact |
|---|---|---|---|
| all-MiniLM-L6-v2 | 2.48 | 1.23 | 45.4 MB |
| arctic-embed-s | 4.65 | 1.53 | 66.4 MB |
| bge-small-en-v1.5 | 4.61 | 1.31 | 66.7 MB |
| granite-small-r2 | 4.87 | 1.63 | 95.3 MB |
| gte-small | 4.6 | 1.37 | 66.7 MB |
| lightretriever-lookup | 0.023 | 0.7 | 465.9 MB |
| lightretriever-lookup-int8 | — | — | 233.6 MB |
| mdbr-leaf-ir | 2.52 | 1.71 | 45.7 MB |
| opensearch-query-side | 0.018 | 0.42 | 0.9 MB |
| potion-base-8M | 0.22 | 0.47 | 15.1 MB |
| potion-retrieval-32M | 0.23 | 0.59 | 64.6 MB |
| static-retrieval-mrl-en-v1 | 0.16 | 0.91 | 62.5 MB |

## Significance (paired bootstrap, 6-ds macro, B=10k)

- lr-dense-websearch vs bge-small-en-v1.5: d=-0.0722 CI95=[-0.0878,-0.0566] p=0.0
- lr-dense-websearch vs granite-small-r2: d=-0.0625 CI95=[-0.0790,-0.0469] p=0.0
- lr-dense-websearch vs potion-retrieval-32M: d=+0.0719 CI95=[+0.0522,+0.0920] p=0.0
- lr-dense-websearch vs static-retrieval-mrl-en-v1: d=+0.0793 CI95=[+0.0602,+0.0990] p=0.0
- lr-dense-websearch vs all-MiniLM-L6-v2: d=+0.0119 CI95=[-0.0051,+0.0292] p=0.1816
- lr-dense-websearch vs leaf-ir-asym: d=-0.0835 CI95=[-0.0978,-0.0693] p=0.0
- lr-dense-pertask vs lr-dense-websearch: d=+0.0263 CI95=[+0.0202,+0.0326] p=0.0
- leaf-ir-asym vs mdbr-leaf-ir: d=+0.0032 CI95=[-0.0020,+0.0083] p=0.2252
- leaf-ir-asym vs arctic-embed-m-v1.5: d=-0.0109 CI95=[-0.0154,-0.0067] p=0.0
## Provenance note (2026-08-25, M7 comparator freeze)

`results/perquery.json` freezes per-query nDCG@10 vectors for nine comparator systems (M7 tiers + M8 references), regenerated from the fp16 artifact caches. Four cells differ from this table by at most 3e-4 — potion-retrieval-32M arguana (0.4489 vs 0.4488) and trec-covid (0.4466 vs 0.4469), mdbr-leaf-ir arguana (0.5939, 9e-5), arctic-embed-m-v1.5 scidocs (0.2148, 1.3e-4): the table recorded encode-time scores computed from pre-save fp32 vectors, while the frozen vectors come from the fp16 files at rest — near-tie reorderings far inside the ≤0.003 harness validation standard. The other 50 cells agree to <5e-5, and every system's avg-6 matches this table to four decimals. For every M7/M8 pairing, perquery.json is authoritative; `scripts/validate_perquery.py` enforces cell-level agreement with these four documented exceptions, and `results/eval_manifest.json` + `results/frozen_eval/` pin the dataset content the pairing assumes.
