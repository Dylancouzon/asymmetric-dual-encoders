# Target-Reader Review of M15 Paper v11

2026-10-01. Essential-only read against `instructions-m15.md`: clear research goal, citable IR knowledge, practical value to search engineers, checkable claims, and a credible Qdrant result. This is a reader review, not a numerical or bibliography audit. I did not edit the paper.

## Verdict

The goal is now clear: determine when a cheaper query path is worth building and serving while document vectors stay fixed. The three decisions (assess the candidate vector space, account for preparation and fitting, retune and measure search) give the paper a reason to exist. The shorter main body reads as a research study rather than a collection of benchmark results. Moving routing, shuffling, prefixes, and absorbability to appendices was the right cut.

Two observations are worth citing: the gte-large/Stella reversal under the fixed closed-form recipe (0.2455 versus 0.3974 table nDCG@10 despite the reverse teacher ranking), and the larger ANN loss for the cheap table (16.3% versus 7.0% for Nano at the same unquantized `ef=16` on the 1M diagnostic). The second result turns an encoding-only speed comparison into a search-engine result. The explicit tier-relative latency target and the $95 training-loop boundary make these observations more trustworthy. Qdrant appears as the measured engine and a source of tunable retrieval behavior, without a marketing claim.

## Essential Remaining Risk

The abstract still presents the +0.1519 table-score difference as what “selecting by the development screen instead changes” after asking about an **existing** index. In that situation the teacher cannot be selected, and the 26-tower result is a comparison of closed-form probes on 26 *different* teacher indexes. The six-set contrast is retrospective; it does not show that the probe selects the separately trained Zero or Nano recipe. Sections 1, 3, 4, and 6 all disclose these distinctions, but a reader can form the broader interpretation before reaching them. That is the likeliest reviewer objection to the central claim, not a minor wording issue.

Fix the abstract's decision boundary directly: “For a fixed index, we test whether a cheap student works in its vector space. When choosing an index that must later support cheap queries, we compare closed-form table probes across candidate teachers.” Then describe the +0.1519 as an observed contrast between the development-selected Stella probe and the probe from the strongest teacher *in this roster*. Keep “screen the intended cheap representation” as the rule. Training the served Zero recipe on a contrasting tower remains the one experiment that would justify a stronger rule about selecting teachers for the final system; without it, the present bounded claim is still publishable as an empirical case study.

## Community-Value Ceiling

The paper now gives an engineer a credible sequence to repeat, but it deliberately does not ship a standalone bring-your-own-teacher tool or pretrained document-vector package. That limits *adoption* of the method even if the findings get cited. This is not a publication blocker and does not justify adding another experimental section. If the public release can include one small, complete worked invocation with input manifests and expected outputs for the closed-form probe and exact-to-ANN check, it would make the process in Section 3.3 usable outside this repository's historical setup. If that artifact cannot be shipped, keep the present honest wording rather than implying that the 4-7 minute probe is a complete end-to-end reproduction cost.

No other essential reader issue found. The existing limitations cover the model/data mismatch, post hoc 16-tower extension, one-index serving study, and tier-relative quality targets.

## Paths Read

Root: `/Users/dylanc/Documents/GitHub/asymetric-dual-encoders`. For this focused pass, `m15/PAPER.md` v11 and the previously read `instructions-m15.md`, `CLAUDE.md`, `m15/HANDOFF.md`, `m15/LOG.md` (last two entries), `m15/NOVELTY.md`, `m15/EVIDENCE.md`, and `m15/REVIEWS/2026-10-01-owner-reader.md`. No protected results, reserved qrels, `work/m9reserve`, or raw M18 confirmation were opened.
