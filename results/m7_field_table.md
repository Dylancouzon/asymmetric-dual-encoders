# M7 objective-by-dataset field table

Written before the first training run, as `instructions-m7.md` requires. Counts are read
from the built mix and `results/m7_decontam.json`, never hand-copied. Rights are
source-level (see `m7/LEDGER.md` for the primary URLs and caveats).

| source | objectives | queries (total) | held-out dev | TRAIN | after decontam | positives | provided hard negs |
|---|---|---|---|---|---|---|---|
| hotpotqa-train | A and B | 85,000 | 1,717 | 83,283 | 82,155 | 170,000 | 0 |
| fever-train | A and B | 109,810 | 2,151 | 107,659 | 107,492 | 140,085 | 0 |
| squad-train | A and B | 87,599 | 1,790 | 85,809 | 85,752 | 87,599 | 0 |
| esci-us | A and B | 74,888 | 1,598 | 73,290 | 73,272 | 988,062 | 122,273 |
| mrtydi-en | A and B | 3,547 | 69 | 3,478 | 3,474 | 3,547 | 104,854 |
| nqopen | B only | 87,925 | 1,813 | 86,112 | 85,899 | 0 | 0 |
| triviaqa | B only | 138,384 | 2,733 | 135,651 | 135,496 | 0 | 0 |

## Fields, rights, and positive construction

### hotpotqa-train
- **fields**: BeIR/hotpotqa queries.text; BeIR/hotpotqa-qrels train split (score>0); BeIR/hotpotqa corpus title+text
- **rights**: CC BY-SA 4.0, dataset and underlying Wikipedia corpus, hotpotqa.github.io
- **positive**: every train-qrels document with score>0 (the supporting paragraphs)
- **objectives**: A and B
- **doc store**: `hotpotqa-corpus`

### fever-train
- **fields**: BeIR/fever queries.text (claims); BeIR/fever-qrels train split; corpus title+text
- **rights**: CC BY-SA, fever.ai/download/fever/license.html (per-article Wikipedia terms, 3.0 fallback)
- **positive**: every train-qrels evidence document with score>0
- **objectives**: A and B. Kept or dropped by the phase-5 dev ablation: keeping it makes BEIR FEVER an in-domain untouched-final row, which the report labels.
- **doc store**: `fever-pos`

### squad-train
- **fields**: rajpurkar/squad train: question, context
- **rights**: CC BY-SA 4.0, canonical HF repo (rajpurkar/squad)
- **positive**: the question's own context paragraph; contexts deduplicated into the store
- **objectives**: A and B
- **doc store**: `squad-ctx`

### esci-us
- **fields**: tasksource/esci, product_locale=='us': query, query_id, product_id, esci_label, product_text
- **rights**: Apache 2.0 at amazon-science/esci-data repo root (the repo is the dataset). Caveat: unanswered issue #21 asks whether Apache-2.0 covers the data.
- **positive**: esci_label=='Exact'. esci_label=='Irrelevant' is kept as a provided hard negative -- a graded judgment, not a mined guess.
- **objectives**: A and B
- **doc store**: `esci-prod`

### mrtydi-en
- **fields**: castorini/mr-tydi english train (parquet mirror): query, positive_passages, negative_passages
- **rights**: Apache 2.0, castorini/mr.tydi LICENSE
- **positive**: positive_passages (title + text); negative_passages kept as provided hard negatives
- **objectives**: A and B
- **doc store**: `mrtydi-docs`

### nqopen
- **fields**: google-research-datasets/nq_open train: question only
- **rights**: CC BY-SA 3.0, declared by Google's maintainers in merged PR #11 (2019-06-10, commit c307fa7030); dropped from the live README in Aug 2019, so cite the commit. The repo LICENSE (Apache-2.0) covers code only.
- **positive**: none -- query text only
- **objectives**: B only
- **doc store**: `-`

### triviaqa
- **fields**: mandarjoshi/trivia_qa rc.nocontext train: question only
- **rights**: Apache 2.0 for the QA pairs (mandarjoshi/triviaqa). Evidence documents keep their own copyright, so they are never used -- this is why TriviaQA is B-only.
- **positive**: none -- query text only
- **objectives**: B only
- **doc store**: `-`
