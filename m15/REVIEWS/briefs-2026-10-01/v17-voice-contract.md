# Brief: does v17 honour the voice contract, and is it the paper the frame promises?

Repository: /Users/dylanc/Documents/GitHub/asymetric-dual-encoders, branch m15-whitepaper, read-only. Read-only shell commands (cat, sed -n, head) are allowed. Change no files; run nothing else; never read results/frozen_eval/untouched-*, reserved qrels caches, work/m9reserve, raw M18 or sealed M19 confirmation; no repo-wide search across results/ or work/.

## Context

m15/PAPER.md is draft v17. It was rewritten after the owner judged v16 "an article, not scientifically exciting" and after two reviews (m15/REVIEWS/2026-10-01-fable-register-v16.md and 2026-10-01-astra-register-v16.md). The frame is now the frozen index: what a cheap query student costs over an index that cannot be rebuilt, and what the index decides about that cost. The agreed voice contract is the last section of m15/REVISION_PLAN_V17.md ("Voice contract"). Three result blocks are marked [pending] (E22, E23, E24) and will be filled from experiments now queued; judge the draft as if those blocks were present.

The owner's standing warning: earlier revisions oscillated between a boring benchmark reading and a blog post. This draft must be neither. It must read as a research paper an IR reviewer would cite, and be interesting.

## Read

1. m15/REVISION_PLAN_V17.md, the "Voice contract" and "Reframing adopted" sections
2. m15/PAPER.md in full
3. results/m15_e21_width_model.json, top-level "rosters" > "pooled" > "head" > "all6" > "models" and "spearman", and "registered_fit_predicts_exploratory" (to check the numbers in Section 4)
4. m15/EVIDENCE.md cards C16 and C17 (Sections 3 and 5 numbers)

## Answer, under 700 words, in this order

1. **Contract check.** Go through the eight contract rules. For each, pass or fail, with one quoted sentence as evidence when it fails.
2. **Register.** Does it read as a research paper now? Quote the three sentences that most read as a blog or as a benchmark report, if any remain.
3. **Argument.** Is the thread one story (frozen index; what you lose and what predicts it; what you pay at search; the instance)? Where does it break?
4. **Numbers.** Any number in Sections 4 to 6 that does not match the sources listed, or any claim stronger than its table.
5. **Cuts.** The main text is about 4,300 words including tables. Name what to cut to land near 3,800 without losing a claim.
6. **The one change** that would most raise the chance an IR researcher cites this.

Be concrete and do not soften. Do not propose new experiments; three are already queued.
