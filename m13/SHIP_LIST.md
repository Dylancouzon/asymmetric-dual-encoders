# M13 ship list — what the cloud instance needs (sizes measured 2026-09-10, names only)

Everything below is gitignored and must be copied; the clone carries code, registries and results.
Same absolute path on the instance: `/home/dylan/asymetric-dual-encoders` (the corpus manifest and
the target-cache record carry absolute paths that enter the resume fingerprint). **Never copy**
`work/dev/cqadup-android.json`, `work/dev/cqadup-english.json`, `work/m9reserve`, `work/lotte`,
`results/frozen_eval/untouched-*` or any reserved qrels cache.

| Group | Path | Size |
|---|---|---|
| training: query corpora | `work/m10paq/paq_build.jsonl` | 65M |
| training: query corpora | `work/m10harvest/harvest_train.jsonl` | 181M |
| training: query corpora | `work/m10harvest/harvest_forms12.jsonl` | 240K |
| training: query corpora | `work/m10gen/generated_queries.jsonl` | 301M |
| training: query corpora | `work/m9_screen_queries.json` | 16M |
| training: query corpora | `work/m9_screen_rows.npy` | 1.9M |
| training: query corpora | `work/m8_trainq_texts.json` | 21M |
| training: query corpora | `work/m9long/corpora/nqopen` | 4.6M |
| training: query corpora | `work/m9long/corpora/triviaqa` | 12M |
| training: targets | `work/enc/trainq-337981-fp16-0e82fdcfbf2f` | 991M |
| training: targets | `work/enc/trainq-337981-fp16-7423fa42cd3e` | 1.3G |
| training: targets | `work/enc/trainq-337981-fp16-e82235ce62b4` | 991M |
| training: targets | `work/enc9/m9long-nqopen` | 168M |
| training: targets | `work/enc9/m9long-triviaqa` | 264M |
| training: targets | `work/m10targets/query-stella_en_400M_v5-8e9a608b5383` | 12G |
| training: documents | `work/pool/stella-400M-v5` | 12G |
| training: documents | `work/pool/ids-esci-prod.json` | 11M |
| training: documents | `work/pool/ids-fever-pos.json` | 268K |
| training: documents | `work/pool/ids-hotpotqa-corpus.json` | 59M |
| training: documents | `work/pool/ids-mrtydi-docs.json` | 1.3M |
| training: documents | `work/pool/ids-squad-ctx.json` | 268K |
| training: documents | `work/decontam` | 51M |
| training: packed tokens and masks | `work/m10tok` | 7.5G |
| training: packed tokens and masks | `work/m10cov/rescreen10` | 184K |
| training: packed tokens and masks | `work/m10arxiv/arxiv_drawn.json` | 3.6M |
| COV reads | `work/m10cov/probe` | 686M |
| COV reads | `work/enc9/m10cov-BRIGHT-d-stella-400M-v5-09a709ba1005` | 1.6G |
| COV reads | `work/enc9/m10cov-BRIGHT-q-stella-400M-v5-9f41fafa613c` | 2.5M |
| COV reads | `work/enc9/m10cov-LEDGER-d-stella-400M-v5-b4e164c82cfb` | 187M |
| COV reads | `work/enc9/m10cov-LEDGER-q-stella-400M-v5-b2ac48cc6bb9` | 40M |
| COV reads | `work/enc9/m10cov-LegalBenchConsumerContractsQA-d-stella-400M-v5-9eba3c192f04` | 632K |
| COV reads | `work/enc9/m10cov-LegalBenchConsumerContractsQA-q-stella-400M-v5-ba33c0f6467b` | 1.6M |
| COV reads | `work/enc9/m10cov-LegalBenchCorporateLobbying-d-stella-400M-v5-1557641d6093` | 1.3M |
| COV reads | `work/enc9/m10cov-LegalBenchCorporateLobbying-q-stella-400M-v5-a06503f4e437` | 1.4M |
| COV reads | `work/enc9/m10cov-MedicalQARetrieval-d-stella-400M-v5-10e9f29435bd` | 8.1M |
| COV reads | `work/enc9/m10cov-MedicalQARetrieval-q-stella-400M-v5-518feac66a2c` | 8.1M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-hotpotqa-docs-fp16-133dfacf940a` | 15G |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-hotpotqa-docs-fp16-88798cdf67ec` | 20G |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-nq-250k-docs-fp16-75da16dd9f8c` | 977M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-nq-250k-docs-fp16-fd69d3594d7a` | 733M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-physics-docs-d2q5-fp16-09a2753b5e2e` | 75M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-physics-docs-fp16-218bfaa8683b` | 75M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-physics-docs-fp16-3f3f7d9affc0` | 57M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-physics-docs-fp16-4895283350a3` | 75M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-physics-docs-fp16-4bff59342088` | 57M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-physics-docs-fp16-5b6bc55c6482` | 75M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-physics-docs-fp16-620ea408fc23` | 75M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-physics-docs-fp16-88f29df31f25` | 75M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-physics-docs-fp16-934082df0b5c` | 75M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-physics-docs-fp16-c0afc4653719` | 57M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-physics-docs-fp16-d1666aba58ac` | 75M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-physics-docs-fp16-f0c6ae9fa551` | 57M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-programmers-docs-d2q5-fp16-aafbcfd5b3b1` | 63M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-programmers-docs-fp16-304d33ef7e39` | 48M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-programmers-docs-fp16-4f9ce05de7f6` | 63M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-programmers-docs-fp16-60d327b7b288` | 63M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-programmers-docs-fp16-6e899595ac74` | 63M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-programmers-docs-fp16-8cbffe146e65` | 63M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-programmers-docs-fp16-a01508c1352a` | 48M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-programmers-docs-fp16-c5ec186bbda9` | 63M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-programmers-docs-fp16-d601344992f7` | 48M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-programmers-docs-fp16-e0919c7cd603` | 63M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-programmers-docs-fp16-e4d19030f069` | 48M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/enc/dev-cqadup-programmers-docs-fp16-ee0797107067` | 63M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/dev/hotpotqa.json` | 1.6G |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/dev/nq-250k.json` | 125M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/dev/cqadup-physics.json` | 31M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/dev/cqadup-programmers.json` | 34M |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/dev/heldout-longq.json` | 28K |
| DEV-6 (or run DEV-6 on the box from the returned checkpoint) | `work/dev/heldout-train.json` | 1.3M |
| final scoring (six-set document vectors, frozen) | `work/enc/final-six-arguana-docs-fp32-7a83b94ec5ed` | 17M |
| final scoring (six-set document vectors, frozen) | `work/enc/final-six-fiqa-docs-fp32-b2dae6f12097` | 226M |
| final scoring (six-set document vectors, frozen) | `work/enc/final-six-nfcorpus-docs-fp32-28f8d3da3d17` | 7.2M |
| final scoring (six-set document vectors, frozen) | `work/enc/final-six-scidocs-docs-fp32-c33e15defe49` | 51M |
| final scoring (six-set document vectors, frozen) | `work/enc/final-six-scifact-docs-fp32-74a3bdba533c` | 11M |
| final scoring (six-set document vectors, frozen) | `work/enc/final-six-trec-covid-docs-fp32-79dc23f59b0b` | 670M |
| weights (HF cache) | `/home/dylan/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5` | 129M |
| weights (HF cache) | `/home/dylan/.cache/huggingface/hub/models--NovaSearch--stella_en_400M_v5` | 1.7G |

Totals by group are the sum of the rows; the training groups are what both E arms need, COV and DEV-6
are what `run_arm.py` reads at cycle ends and at the final checkpoint, the six-set caches are for the
final scoring transaction only if it runs on the instance.

```bash
# from the box, after the instance has the clone at the same path (dry run first: add -n)
rsync -a --info=progress2 --relative \
  --exclude="work/dev/cqadup-android.json" --exclude="work/dev/cqadup-english.json" \
  --exclude="work/m9reserve" --exclude="work/lotte" \
  ./work/m10paq/paq_build.jsonl ./work/m10harvest/harvest_train.jsonl ./work/m10harvest/harvest_forms12.jsonl \
  ./work/m10gen/generated_queries.jsonl ./work/m9_screen_queries.json ./work/m9_screen_rows.npy ./work/m8_trainq_texts.json \
  ./work/m9long/corpora ./work/enc/trainq-337981-fp16-* ./work/enc9/m9long-nqopen ./work/enc9/m9long-triviaqa \
  ./work/m10targets ./work/pool ./work/decontam ./work/m10tok ./work/m10cov/rescreen10 ./work/m10cov/probe ./work/m10arxiv/arxiv_drawn.json \
  ./work/enc9/m10cov-* ./work/enc/dev-hotpotqa-docs-* ./work/enc/dev-nq-250k-docs-* ./work/enc/dev-cqadup-physics-docs-* ./work/enc/dev-cqadup-programmers-docs-* \
  ./work/dev/hotpotqa.json ./work/dev/nq-250k.json ./work/dev/cqadup-physics.json ./work/dev/cqadup-programmers.json ./work/dev/heldout-*.json \
  ./work/enc/final-six-*-docs-fp32-* \
  <user>@<instance>:/home/dylan/asymetric-dual-encoders/
```

After the copy, on the instance: `sha256sum results/perquery.json` must read
`6b18e3dd74fd308b087d4e652c0999c9d6a2e729c78edb73f000f9b7a25fda7e`; then `./run_checks.sh`, then
`.venv/bin/python m10src/arm_smoke.py --device cuda` and `m10src/run_arm.py --plan` before any arm.

## Notes on the sizes

- Two `dev-hotpotqa-docs` caches exist (20 GB and 15 GB) and `work/m10tok` holds older token caches
  beside the current ones: only the identities the current code derives are needed. Resolve them on
  the box before copying (`teacher.cache_key` for the DEV-6 caches; the `work/m10tok/<id>/meta.json`
  the assemble manifest names) and copy those alone.
- **Recommended split:** ship the training groups plus COV (about 35 GB) for the two E arms, and run
  DEV-6 on the box afterwards from the returned `cycle3.pt` (415 MB per arm) instead of shipping
  35 GB of DEV-6 caches. `run_arm.py` reads DEV-6 at the end of an arm on the same machine, so this
  needs a small `dev6-from-checkpoint` entry point; otherwise ship the DEV-6 group.
- The six-set document vectors (about 1 GB) and the stella weights are needed on the instance only
  if the final scoring transaction and the LoTTE encode run there.
