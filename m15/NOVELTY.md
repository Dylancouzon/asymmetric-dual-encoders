# M15 novelty and related work

Sub-agent pass 2026-09-17: internal novelty files re-read, then an external check against the
literature as of today. Findings below are the sub-agent's; the verdict line at the end is mine.

## The headline, stated plainly

**Index-preserving query-encoder swap is prior art, under two established names.** The paper must
not claim it as a new capability.

- MongoDB **LEAF** (arXiv 2509.12539, ACL 2026) ships this as a first-class released mode and calls
  it *asymmetric inference* or *mixed-checkpoint inference*: encode documents with the teacher,
  queries with the distilled student, one index. Released `-asym` checkpoints exist.
  https://huggingface.co/MongoDB/mdbr-leaf-ir-asym
- **pyNIFE** (Stephan Tulkens, 2025-11-03) trains a dense per-token lookup table by cosine
  distillation against a frozen off-the-shelf teacher and reuses the teacher's index unchanged.
  This is the same construction as `zero` and predates it. `research/m7-novelty.md` already
  withdrew that novelty claim on 2026-09-03; the external check confirms the withdrawal still holds.
  https://github.com/stephantul/pynife
- **Backward-compatible training** (Shen et al., CVPR 2020) and **forward-compatible training**
  (Apple, arXiv 2112.02805) name the general goal of changing the model without re-embedding the
  corpus. The setting there is version migration rather than a standing choice between compute tiers.

No academic work uses the word "hot-swap" for this. The word appears in developer writing about the
re-indexing pain point, so it carries no citation weight and no literature gap.

## Related work map

| Work | What it does | How our setup differs |
|---|---|---|
| LightRetriever (arXiv 2505.12260) | Token lookup table trained end to end, no query-side forward pass | Its document tower is co-trained, so it cannot move onto an index someone else built |
| pyNIFE | Per-token lookup table distilled against a frozen off-the-shelf teacher | Same construction as `zero`, published first. NanoBEIR numbers only, one query-side option |
| LEAF (arXiv 2509.12539) | 23M transformer student aligned to a frozen teacher, shipped for mixed-checkpoint inference | One student per teacher. No lookup-table tier beside it, no frontier between tiers |
| Query Encoder Distillation via Embedding Alignment (arXiv 2306.11550) | Frozen document encoder, transformer student, MSE alignment | Older, smaller scale, transformer student only |
| BCT / FCT | Change the model without re-embedding the gallery | Version migration, not a standing multi-tier deployment choice |
| Model2Vec, potion-retrieval-32M | Static lookup-table embeddings | Used symmetrically for both sides, so it never tests the swap |
| CARE, Li-LSR, KAHM, ERA | Reviewed and ruled out in `research/m7-novelty.md` | Co-trained towers, scalar rows, prototype mixtures, or a full query transformer |

## What is defensible

The open ground is the **measurement**, not the mechanism. Every published asymmetric pair trains
its cheap query tower against the document encoder it will retire alongside. None of them asks
which of several already-existing cheap query encoders should be plugged into a document index that
was not built for that purpose, and none measures a controlled two-tier quality-against-compute
frontier on one held-fixed index under a protocol registered before the numbers existed.

That is an evaluation-design contribution. The paper claims the frontier and the deployable
implementation, and cites pyNIFE and LEAF as the prior art it stands on.

## The three attacks and what answers each

1. *"pyNIFE published this construction already."* Answer with what pyNIFE does not have: full BEIR
   breadth rather than NanoBEIR's 50-query samples, a different and larger teacher
   (`stella_en_400M_v5`, 1024-d), the second tier, and the frontier between the two.
2. *"LEAF already ships mixed-checkpoint inference."* Answer by stating the claim as measurement
   under a registered protocol, and by citing LEAF as prior art in the same paragraph.
3. *"Swapping the query encoder is just the bi-encoder inference contract; this is an engineering
   demo."* Answer with the frontier: where each tier is usable and where it is not, in numbers,
   plus the places the swap is not free (prompt strings, tokenization, dtype, fusion depth).

## Bibliography

- LightRetriever, arXiv 2505.12260 (v5, 2026-01-30). https://arxiv.org/abs/2505.12260
- pyNIFE, Stephan Tulkens, 2025-11-03. https://github.com/stephantul/pynife
- LEAF, arXiv 2509.12539, ACL 2026. https://arxiv.org/html/2509.12539v2 ; https://huggingface.co/MongoDB/mdbr-leaf-ir-asym
- MongoDB LEAF engineering blog. https://www.mongodb.com/company/blog/engineering/leaf-distillation-state-of-the-art-text-embedding-models
- Query Encoder Distillation via Embedding Alignment, arXiv 2306.11550. https://arxiv.org/abs/2306.11550
- Shen et al., Towards Backward-Compatible Representation Learning, CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/papers/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.pdf
- Forward Compatible Training, Apple, arXiv 2112.02805. https://arxiv.org/pdf/2112.02805
- Model2Vec. https://github.com/MinishLab/model2vec

CARE and ERA are carried from `research/m7-novelty.md` without an independent re-fetch on
2026-09-17. Re-verify before citing either in the paper.
