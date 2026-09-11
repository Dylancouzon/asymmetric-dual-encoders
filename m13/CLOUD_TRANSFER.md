# Initial M13 cloud transfer — 2026-09-11

`cloud_transfer_files.txt` is an explicit file allowlist, relative to filesystem root `/`.
It covers initial E training, COV, BAAI weights, the tiny-random test configuration, and
Stella weights for the optional training-document encode benchmark. It does not authorize
protected evaluation. No transfers were performed while preparing this list.

All 266 listed files exist locally. Regular-file payload totals **46,295,021,686 bytes
(46.30 GB / 43.12 GiB)**, excluding symlink payload duplication. No missing selected artifact
was found. These are stat sizes, not transfer or content-integrity verification.

| Group | Files | Bytes |
|---|---:|---:|
| Ship-list training and COV document vectors | 140 | 37,670,289,465 |
| Additional loader dependencies | 19 | 2,955,464,943 |
| HF model caches | 69 | 1,880,672,407 |
| COV dataset Arrow caches | 38 | 3,788,594,871 |

The code clone must be installed separately at `/home/dylan/asymetric-dual-encoders`.
The paths enter corpus/resume fingerprints; do not run from the local M13 worktree path on cloud.
Run rsync from the source box, with the committed allowlist path adjusted if needed:

```bash
rsync -an --info=progress2 \
  --files-from=/home/dylan/asymetric-dual-encoders/work/m13cloud/m13/cloud_transfer_files.txt \
  / root@HOST:/
```

Inspect that dry run, then remove only `n` from `-an`. Configure SSH host/port/key separately.
No `--delete`, wildcards or directory-recursive flags are needed: each file is enumerated.
`-a` preserves HF snapshot symlinks; all their blob referents are also in the allowlist.
Do not use `-L`, which duplicates the model bytes. Never transfer a credential directory or
all of `work/`, HF datasets, `work/decontam`, or `work/train`.

## Dependencies missing from the historical ship list

- `m9src/data.py:labelled_query_pool` reconstructs source labels via `m7src/train.py:build_arrays`.
  This requires the five exact `work/train/sources/{hotpotqa-train,fever-train,squad-train,esci-us,mrtydi-en}.json`
  files and `work/decontam/kept.json`. The FEVER file is the admitted **training lineage** file;
  the M9 loader subsequently excludes FEVER queries. No reserved FEVER evaluation is included.
- `train.banned_rows` verifies its mask against **`work/pool/meta.json`**, in addition to
  `work/pool/stella-400M-v5/meta.json` required by the live vector pool.
- `longrun.extra_texts` eagerly loads every source in `results/m9_extended_screen.json`, including
  `work/pseudoq/pseudoq-2000000-0.json` (152,832,623 bytes), even though E consumes only nqopen
  and triviaqa. It also requires `work/train/querytext/{nqopen,triviaqa}.json` and all three
  `work/decontam/m9_kept_*.json` files named in the allowlist. The tokenized
  `work/m9long/corpora` directories do not replace these inputs.
- A document token-cache miss needs four exact `work/train/stores/{hotpotqa-corpus,esci-prod,mrtydi-docs,squad-ctx}.json`
  files (2,703,439,119 bytes). Including them permits valid smoke draws/cache regeneration.
  No FEVER document store is needed for the screened document draw.
- `run_arm.CovEval` calls `cov_probe.units`, which loads pinned datasets, and `cov_eval10`
  separately loads BRIGHT document slices. The HF dataset cache directories are explicit in
  the allowlist: `mteb___medical_qa`, `mteb___legalbench_corporate_lobbying`,
  `mteb___legalbench_consumer_contracts_qa`, `artefactory___ledger-long-context-kpi-qa`, and
  `xlangai___bright`. BRIGHT includes only the six admitted slices, for examples/documents,
  at revision `3066d29c9651a576c8aba4832d249807b181ecae`.
  Dataset loaders may still contact HF to resolve metadata; validate loader/cache use on cloud
  before launching E. This list contains Arrow caches, not the entire HF download cache.

## Identity and scope

The selected `trainq-337981-fp16-7423fa42cd3e` metadata names Stella revision
`ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`, 1024-dimensional Dense projection and the registered
query prefix. Other teacher target variants are omitted. The training token directories are
an explicitly enumerated existing-cache superset (7,999,701,441 bytes): live assembly must derive
and validate its key. The candidate cut-query cache is `dfc8d12618876554` (bge-small, 512 tokens,
2,651,572 rows). Do not force a cache identity or change tokenizer settings to obtain a hit.

Model caches include `models--BAAI--bge-small-en-v1.5`,
`models--hf-internal-testing--tiny-random-BertModel` (config only; rehearsal seeds its own weights),
and `models--NovaSearch--stella_en_400M_v5`. Stella is optional for E's cached targets but supports
the planned encode benchmark. The full test suite's other pretrained-model requirements still
need the environment review; existence of this list does not certify all tests offline.

Excluded: all DEV-6 inputs, final-six vectors, LoTTE, reserved surfaces/qrels,
`work/m9reserve`, `results/frozen_eval/untouched-*`, the protected fingerprint index
`work/decontam/m8_protected_query_index.npz`, COV probe vectors and teacher query vectors.
Use `--dev6 defer` for both E arms and return checkpoints for the registered box-side DEV-6 fill.

After transfer, verify the frozen comparator digest from the clone, run the prescribed checks,
validate real assembly and COV dataset/cache availability, and smoke both registered shapes and
resume before E execution. Do not interpret transfer success as experiment readiness.
