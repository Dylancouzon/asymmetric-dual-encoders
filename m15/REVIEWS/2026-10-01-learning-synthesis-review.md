# Learning synthesis: adversarial review

2026-10-01. Reviewed `m15/LEARNINGS.md` against all three learning audits and targeted named source passages. **Disposition: usable synthesis; no blocking factual or scientific overclaim identified.** One consequential omission and one editorial boundary deserve attention before using the catalog to select the manuscript.

## 1. Add the tested shortcut: post-hoc linear alignment of an existing cheap encoder

The catalog covers compatibility defects and how to build students, but omits the early experiment that tests a natural alternative: fit a small projection from an already available static model into the retained document space. This is a distinct production decision, not another failed training arm.

The early audit and `research/m1-m6-findings.md` report potion-32M→arctic at **0.3280**, below potion's own-index **0.3427**, and potion-8M→arctic at **0.3036**, even after selecting regularization directly on test retrieval. The loader mismatch was repaired before these reported outcomes.

**Catalog addition:** A dimension-matching adapter does not by itself establish a useful index-compatible retriever. Evaluate its retrieval before assuming it avoids student training or document migration. These particular static-encoder/linear-map experiments failed to establish a competitive replacement despite optimistic parameter selection.

**Boundary:** Do not inherit the source's “airtight” rejection of post-hoc alignment as a class. Test-selected regularization is an optimistic local diagnostic, not untouched confirmation or an impossibility result. Nonlinear and jointly trained alternatives remain open. Verify the original receipt and metric/protocol before adding these numbers to the paper. A main-text sentence could explain the option without adding another historical table.

## 2. Keep two production entry points separate in the editorial selection

The catalog's opening question is conditional on retaining an existing document index, whereas its strongest proposed construction result, L02, compares alternative teacher/index pairs. L02's individual entry handles this correctly; the final editorial table needs the same separation when translated into prose.

- **Already have the index:** choose an available aligned query encoder or train a student for that fixed space; test its rankings and approximate-search requirements.
- **Still choosing the index/teacher:** compare candidate cheap encoders in each teacher's own space before committing. The ridge screen supports this decision only for its tested recipe; it does not select the final trained Nano/Zero recipe.

Without that branch, “choose by student ranking” risks sounding like an actionable 26-model selection tool for a deployed Stella index. With it, the study connects deployment and construction coherently. This is a framing qualification, not a request for more experiments.

## Checks passed

- The catalog genuinely extends beyond M15: early surrogate/fragmentation failures, M9–M10 preparation and representation work, M12 fusion depth, M17–M19 failed or inconclusive specialization, and later serving/cost failures all remain available. Main-text selection is selective rather than erasing them.
- Native quantization interaction, graph-free query-precision control, fixed-artifact timing, and general architecture claims are separated. The weak primary replication and absolute miss headroom remain visible.
- Fusion is proposed as an alternative to test, with extra costs and negative corpus examples. The depth reversal is labeled descriptive, the policy override is disclosed, and score-magnitude causality is withheld.
- Oracle selection is not sold as a router, a development-selected blend is not sold as a deployable improvement, and operationally bounded vocabulary changes are not sold as proven relevance gains.
- Construction cost excludes preparation/research. The catalog does not convert M9's failed model into a capacity diagnosis, the rebuild into a causal ablation, or the voided teacher arm into evidence for trained-student teacher selection.
- Caller/device parity and judgment resolution are concrete production checks supported by observed failures. They add more practical value than additional marginal model comparisons. Keep one strong example of each in the manuscript, rather than copying the full catalog into a checklist.

## Editorial judgment

The proposed selection is useful if the manuscript keeps a few decisions in focus: choose a compatible quality budget; measure search and fusion at the intended candidate budget; approve the actual served route and its evidence. Teacher selection is a clearly marked construction branch. Weak routing and blending results can be one compact paragraph. Allocator chronology, solver details, every failed objective, and archive state need not become main-text material merely because they were audited.

No new experiment is required to use the present catalog honestly. Stronger claims about a native query-precision remedy, a useful production router, or successful relevance patching require their respective missing measurements; the catalog already says so.

## Access record

Read `m15/LEARNINGS.md`, `m15/REVIEWS/2026-10-01-learning-audit-early.md`, and `m15/REVIEWS/2026-10-01-learning-audit-late.md`; reused the middle audit written and checked in the immediately preceding task. Targeted source passages were read in `research/m1-m6-findings.md`, `m20/FINDINGS.md`, and `m20/STATUS.md`. Filename-only inventory identified the three audit files. No result receipt, raw query/qrel, protected evaluation, or sealed confirmation was opened. Only this review was written; no manuscript change, experiment, commit, or push.
