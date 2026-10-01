# Brief: v18 after the cuts, narration round 1 (Astra)

Repository: /Users/dylanc/Documents/GitHub/asymetric-dual-encoders, branch m15-whitepaper, read-only. Read-only shell commands (cat, sed -n, head, grep on named files) are allowed. Change no files; never read results/frozen_eval/untouched-*, reserved qrels caches, work/m9reserve, raw M18 or sealed M19 confirmation; no repo-wide search across results/ or work/. Read JSON results by named top-level keys only; never print "rows".

## Context

m15/PAPER.md is v18. Since your complete-v17 review: your ten corrections are applied; the Fable reviewer's corrections are applied (m15/REVIEWS/2026-10-02-fable-complete-v17.md); its sign-off blocker was real and is fixed by E24 amendment 1 (m15/MEASUREMENTS.md): the head row of the prospective test is now n=8 at +0.86 [+0.33, +1.00], the table row n=7 unchanged. The cuts both of you proposed are applied where you agreed (Section 6 merged, intro numbers paragraph replaced, 4.4 prose replaced, 5.5 folded), plus Fable's (5.4 folded into 5.2, Table F to four rows with the full table in Appendix C, typing/pause/submit sentence removed, abstract leads with the width finding). Not taken: your 180-word abstract cut and the relocation of the 4.2 definitions, on the grounds that the abstract is what gets cited and a reader checking Table B needs the definitions in place. Main-text prose is about 4,100 words. "We" is kept as standard IR usage.

The owner's goals, in his words: a research paper, not a benchmark report and not a blog; it builds credibility with seasoned search engineers and IR researchers; findings people cite and argue about; interesting, useful, applicable, and something people want to share, while still looking like a serious paper. The claim he wants defended: query-side distillation over a frozen index is more efficient end to end and drops into an existing pipeline; never quality superiority over joint training. You hold narration authority; the Fable reviewer holds technical authority; both are equal and the owner trusts the two of you to settle the final text. We go back and forth until both sign.

## Read

1. m15/REVISION_PLAN_V17.md, "Voice contract"
2. m15/PAPER.md in full
3. m15/REVIEWS/2026-10-02-fable-complete-v17.md
4. m15/EVIDENCE.md cards C16 to C22, as needed for numbers

## Answer, under 900 words, in this order

1. **Numbers in the edited passages.** Abstract, Section 1, 4.4, the last paragraph of 5.2, 5.3, 5.5, Section 6: any number or claim that does not match its table or source. Quote the sentence and give the source value.
2. **Narration.** Is this the best version for the goals above? Give at most five concrete changes, each as the current sentence and your replacement text, ranked by how much each raises the chance an IR researcher reads to the end and shares the paper. Blog register is not acceptable.
3. **Coherence.** Any place the cuts broke the thread or left a dangling reference.
4. **Disagreements.** Any cut decision above you would reverse, with the reason.
5. **Sign-off.** Yes or no as coauthor; if no, the one thing.

Essential-only: correctness that changes a claim, coherence, and the five narration changes. No style nitpicks. Do not soften.
