# Astra review — M20 stage C msmarco split fix (2026-09-23)

Model `gpt-6-astra`, reasoning effort xhigh, `--sandbox read-only`. Verdict **GO**, no essential
correctness defect, scoring continues, no score rows discarded.

Brief: `work/m20/logs/astra_split_fix_brief.md`.
Full transcript: `work/m20/logs/astra_split_fix_review.log` — **that log contains a reserved-qrels
preview returned by a web search (see the incident below). It is under the gitignored `work/`
tree and is untracked. Do not copy its contents into the repository and do not re-read it
casually.**

## Access incident — recorded, not buried

The reviewer self-reported: a web search about the PUBLIC cqadupstack gaming forum unexpectedly
returned a preview of the RESERVED cqadup-english forum's qrels, and those values landed in its
tool output. It states it did not open that dataset and did not use the values; no file was
written, nothing was evaluated or encoded, and no GPU work ran.

Assessment:

- **Containment.** The values exist only in `work/m20/logs/astra_split_fix_review.log`. `work/` is
  gitignored (`.gitignore:190`) and the file is untracked, so nothing reached the repository. This
  record was extracted from the final report only and was scanned for qrels rows and identifiers
  before being written: none present.
- **Effect on the reserved four.** None that changes their status. They were spent once on
  2026-09-19 and are already **known-test** — not usable to decide anything in this project — so
  there is no further decision this could contaminate.
- **Effect on these findings.** None. The review's subject is msmarco's split; no finding depends
  on cqadup-english qrels, and the reviewer explicitly declined to certify the reserved forums'
  historical split precisely because that evidence was outside its allowlist.
- **The brief was not at fault for omitting the rule** — it forbade loading, opening or inspecting
  reserved queries and qrels "including via `datasets.load_dataset`". What it did not anticipate
  is a **web search** surfacing protected content without the dataset ever being opened.
  **Lesson for future briefs: forbid web searches naming any reserved dataset or forum, not just
  local loads.**

## Access audit, otherwise

No recursive search over `results/` or `work/`; no read of `results/frozen_eval/untouched-*`,
reserved qrels caches or `work/m9reserve`. Isolated execution used synthetic loader stubs rather
than touching reserved data.

## What it established beyond my own evidence

The `dev` -> `validation` mapping at the mechanism level, not by counts: the upstream repository at
the pinned revision `253fbf8…` contains `dev.tsv`, and Hugging Face normalizes `dev` filenames to
the `validation` split name. It noted that my 7,437-pair / 6,980-query match corroborates this but
would not on its own prove label identity — a fair correction to how I framed the evidence.

It also verified all 48 completed rows: query sets agree across systems, stored means reproduce
within `1.2e-16`, and fusion input hashes match their producer rows.

Stated verification limits: no fresh numerical recomputation of the six corpora, no label-by-label
comparison against the original BEIR archive, and it cannot certify the reserved forums' historical
split selection from the permitted files.

---

No essential correctness defect found in `eececdc`. The fix selects MS MARCO’s registered dev split and preserves the 48 completed rows.

1. **MS MARCO: `validation` is correct.** The upstream repository’s head matches pinned revision `253fbf8…` and contains `dev.tsv`, `test.tsv`, and `train.tsv`. Hugging Face explicitly normalizes `dev` filenames to `validation`. This establishes the mapping independently of matching counts. Your 7,437-pair / 6,980-query evidence corroborates it; counts alone would not prove label identity. I did not compare every label against the original BEIR archive. [Repository files](https://huggingface.co/datasets/BeIR/msmarco-qrels/tree/main), [HF split-name rules](https://huggingface.co/docs/datasets/repository_structure#split-name-keywords).

2. **The six completed corpora are unaffected.** I executed the isolated split helpers and checked all 48 rows:

   | Corpus | Registered → loaded | Queries per system |
   |---|---|---:|
   | scifact | test → test | 300 |
   | nfcorpus | test → test | 323 |
   | scidocs | test → test | 1,000 |
   | trec-covid | test → test | 50 |
   | fiqa | test → test | 648 |
   | arguana | test → test | 1,406 |

   All rows match the committed files, current registry hash, and registered source/revision pins. Query sets agree across systems; stored means reproduce within `1.2e-16`; fusion input hashes match their producer rows. SciFact matches the smoke references, and NFCorpus’s Zero DBSF value matches the reproduction receipt to rounding. The fix changes no inputs or calculations for these six, and resume skips them before loading data. **Fresh numerical recomputation was not performed**, as requested.

3. **Recording `"dev"` is appropriate provenance.** It names the registered benchmark split and the underlying `dev.tsv`. Recording an additional loader split could improve convenience, but its absence does not invalidate these artifacts: the pinned repository and committed mapping identify what is loaded. Rewriting existing rows or changing the registry is unnecessary.

4. **The CQADupStack branch is correct for the ten public forums.** Their published default qrels splits are `test`, consistent with BEIR’s convention. The loader uses each forum’s registered repository/revision and `default` configuration. Reserved names are rejected before loading, so this branch cannot change their historical scores. [BEIR dataset table](https://github.com/beir-cellar/beir#-available-datasets), [example forum configuration](https://huggingface.co/datasets/mteb/cqadupstack-gaming/blob/main/README.md).

   **Historical verification limit:** the permitted registry omits the forum split, and the tagged reserved executor was outside the original allowlist. I cannot independently certify its historical split selection from the files inspected.

5. **No additional split-selection defect found in the inspected path.** Isolated execution with synthetic loader stubs checked all 22 public corpora: only MS MARCO changes its loader split; registered metadata and revision arguments remain intact. This does not certify uninspected metric helpers or constitute a real multi-corpus smoke.

6. **The duplicated registry lookup is not an essential defect.** Both lookups select the same dataset key. With the registered file unchanged, there is no demonstrated inconsistency. Consolidating them would be optional cleanup.

7. **The commit’s substantive scoring claims check out**, subject to the verification limits above. The recorded failure occurs before MS MARCO scoring begins.

**P2 — reviewer access incident:** my public gaming-forum web search unexpectedly returned a reserved English-forum qrels preview. Those values entered the tool output; I did not open that dataset or use them. This violated the intended access boundary and must be included in your review-acceptance audit. No files were written, no evaluation or encoding ran, and no GPU work was launched.

**GO — let scoring continue. No score rows need to be discarded because of this fix.**
