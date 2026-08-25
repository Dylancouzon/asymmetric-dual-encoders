# M7 protocol ledger

Append-only. Records the partition ledger, the freeze record, every six-set and
untouched-final access, and crash re-runs.

## Bring-up

- 2026-08-25 — Machine confirmed: RTX 3080 10 GB VRAM, 25 GB RAM, 16 cores, 946 GB free ext4, nvcc 12.6.

## Bring-up (continued)

- 2026-08-25 — Env: Python 3.12.14 venv, torch 2.8.0+cu126 (CUDA available, RTX 3080),
  transformers 4.57.6, datasets 5.0.1, pytrec-eval-terrier 0.5.10. Lock: `m7/requirements.lock.txt`.
- 2026-08-25 — `scripts/validate_perquery.py`: OK, 54 cells (4 allowlisted per FINAL_MATRIX.md).
- 2026-08-25 — `scripts/verify_manifest.py` (new): all six datasets re-downloaded from HF and
  matched to `results/eval_manifest.json` on n_docs/n_queries/corpus_ids/corpus_text/qids/qrels,
  and `results/frozen_eval/` matched to the fresh download. Frozen comparator pairing is valid.
- 2026-08-25 — **SIX-SET ACCESS, class (a) harness validation** (`m7src/validate_harness.py`,
  `results/m7_harness_validation.json`): bge-small ArguAna 0.6038 (want 0.6034, +0.0004);
  bge-small SciFact 0.7127 (0.0000); bm25 FiQA 0.2532 (-0.0000). All within the 0.003 standard.
  No new-model number was scored against six-set qrels in this access.

## Partition ledger (2026-08-25)

Doc-encode dtype decision, logged with evidence: teacher fp16 vs fp32 agrees to cosine
1.000000 on 10K FiQA docs (`results/m7_throughput.json`) and to |Δ nDCG@10| ≤ 3e-4 on both
CQADupStack dev components (`m7src/dtype_check.py` output). **fp16 for dev and training
corpora (2.4x throughput), fp32 for the six-set and untouched-final final run.**

### TRAIN
Approved sources only (`research/m7-data-licensing.md`): HotpotQA train qrels (BEIR),
FEVER train qrels (BEIR), SQuAD train, Amazon ESCI (US locale), MIRACL-en, Mr. TyDi-en,
NQ-open + TriviaQA question text (objective B only — TriviaQA evidence docs keep their own
copyright), plus self-generated synthetic queries over approved seeds if needed.
Excluded by decision: MS MARCO, Quora, S2ORC, PubMed, StackExchange (new dumps), GooAQ,
ELI5, WikiAnswers, sentence-transformers/embedding-training-data as a blanket source.

### DEV (pinned, hashes in `results/m7_dev_manifest.json` — frozen before any candidate result)
| component | docs | queries |
|---|---|---|
| nq-250k (all qrels-positive + rng(0) distractors to 250K) | 250,000 | 3,452 |
| hotpotqa (full BEIR corpus) | 5,233,329 | 7,405 |
| cqadup-programmers | 32,176 | 876 |
| cqadup-physics | 38,316 | 1,039 |
| heldout-train / heldout-longq (built by trainmix.py) | see m7_dev_manifest.json | |

Banned from dev: Touché (args.me is ArguAna's source family), Quora (no license).

### KNOWN-TEST (development-informed)
The six: scifact, nfcorpus, fiqa, arguana, scidocs, trec-covid. Content pinned by
`results/eval_manifest.json` + `results/frozen_eval/`, re-verified on this machine.

### UNTOUCHED-FINAL
- BEIR **FEVER** — admissible (CC BY-SA, fever.ai/download/fever/license.html, verbatim).
- BEIR **DBpedia-entity** — admissible with caveat (test collection MIT,
  github.com/iai-group/DBpedia-Entity/blob/master/LICENSE; underlying DBpedia abstracts
  CC BY-SA 3.0 + GFDL, dbpedia.org/about).
- BEIR **Climate-FEVER** — **DROPPED. Fails the affirmative-license standard.** A Sonnet
  primary-source sweep (2026-08-25) found no license statement at climatefever.ai, in
  arXiv:2012.00614 including appendices, or in github.com/tdiggelm/climate-fever-dataset
  (no LICENSE file, README silent). Only HF mirrors assert CC-BY-SA-4.0 — a wrapper tag,
  which this project does not accept as evidence (the same rule that excluded Quora).

### Dev/eval source-level license evidence (recorded at kickoff, per the eval-use standard)
- NQ — CC BY-SA 3.0, first-party but **no longer on the live README**: declared by Google's
  maintainers in merged PR #11 (2019-06-10, commit c307fa7030) and silently dropped by an
  Aug-2019 commit. Cite the commit, not the live page. The repo's LICENSE file is Apache-2.0
  and covers code only.
- HotpotQA — CC BY-SA 4.0, dataset and underlying Wikipedia corpus, hotpotqa.github.io.
- CQADupStack — CC BY-SA 3.0, stated verbatim in the ADCS 2015 paper ("released in line with
  the original licence of the StackExchange dump"), i.e. the 2014 dump, predating Stack
  Exchange's 2024 no-LLM-training clickwrap. Eval-only use here. The official download host
  (nlp.cis.unimelb.edu.au) was unreachable during the sweep; the paper text is the anchor.
  Note the HF wrapper tags contradict each other (BeIR cc-by-sa-4.0 vs mteb apache-2.0) —
  exactly why wrapper tags are not evidence.
- ESCI — Apache 2.0 at repo root (the repo is the dataset). Caveat: unanswered issue #21
  (opened 2024-11-12) asks whether Apache-2.0 covers the data; Amazon has not replied.
- MIRACL, Mr. TyDi — Apache 2.0, LICENSE files confirmed, no caveats.
- FEVER — CC BY-SA per fever.ai's own license page (per-article Wikipedia terms, 3.0 fallback).
- **BEIR itself is not a license authority**: its Apache-2.0 covers packaging/code only and
  its README disclaims per-dataset licensing.
