# Brief: the complete v17, numbers and voice

Repository: /Users/dylanc/Documents/GitHub/asymetric-dual-encoders, branch m15-whitepaper, read-only. Read-only shell commands (cat, sed -n, head) are allowed. Change no files; run nothing else; never read results/frozen_eval/untouched-*, reserved qrels caches, work/m9reserve, raw M18 or sealed M19 confirmation; no repo-wide search across results/ or work/.

## Context

m15/PAPER.md is v17 with every experiment in: E21 (width model), E22 (gap predictors), E23 (corpus-size subsample), E24 (prospective test, predictions hashed before scoring), E25 (LightRetriever's own lookup path). Title chosen by the owner: "Your Index Is Fine, Your Query Encoder Is Not: Query Distillation over Frozen Indexes". The voice contract is the last section of m15/REVISION_PLAN_V17.md. The owner's inclusion philosophy: keep everything valuable that meets the criteria; reviewers cut together, unless coherence suffers. Main-text prose is about 5,100 words, above the contract's 4,000.

## Read

1. m15/REVISION_PLAN_V17.md ("Voice contract", "Reframing adopted")
2. m15/PAPER.md in full
3. m15/EVIDENCE.md cards C16 to C22 (the numbers behind Sections 4 and 5)
4. results/m15_e24_prospective.json top-level "stats" and "predictions_committed_sha256"; results/m15_e22_gap_predictors.json "univariate" and "held_out_models"; results/m15_e23_fiqa25k.json "summary"; results/m15_e25_lightretriever.json "workloads" > contrasts and exact. Do not read "rows".

## Answer, under 800 words, in this order

1. **Numbers.** Every number in Sections 4, 5, and 6 and Appendix B that does not match its source, or any claim stronger than its table. Be exact: section, quoted sentence, source value.
2. **Contract.** The eight rules: pass or fail, one quoted sentence per failure.
3. **Coherence.** Does the complete paper read as one argument? Where does a new section (5.3 Table F, 5.4, 5.5, 4.3 prospective) break the thread or repeat another?
4. **Cuts.** Name specific paragraphs, sentences, or table rows to remove to reach about 4,000 words of prose without losing a claim or a required disclosure. Rank them by words saved per claim lost.
5. **The sentence you would put first in the abstract** if the goal is that IR researchers share it.
6. **Sign-off.** Would you sign this as a coauthor? If not, the one thing that stops you.

Do not soften.
