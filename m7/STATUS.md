# M7 status

**Stage:** teacher decided and validated; phase-2 answered; arctic-embed-l encodes started.
Next session is compute. Detail: `m7/RESULTS.md` (runs), `m7/LEDGER.md` (protocol + the Codex
gate's open list), `m7/EXPLORED.md` (closed and reopened avenues), `results/m7_*.json` (numbers).

## Run these next, in this order

1. **Teacher learnability probe — do this BEFORE the 6.17M-doc pool encode.** The teacher was
   picked on *symmetric ceiling*; what determines the shipped system is how well a bag-of-tokens
   table approximates that teacher (Codex M-probe), and that is the tie-break for the tension below.
   Design, so it need not be re-derived: it needs **no documents and no qrels** — targets are the
   candidate's own **query** vectors, so encode the TRAIN query texts (349,934, ~11 min per
   candidate at ~535 texts/s) with arctic and stella, fit `stage0_ridge`'s closed form per candidate
   against those targets, then report (a) **cosine agreement to the teacher's query vector on the
   dev components' queries** as the primary metric, high-SNR and coverage-symmetric, and (b)
   retrieval nDCG on the two CQADupStack components, whose docs are already encoded for both. Fit on
   TRAIN, measure on dev: no optimism, real coverage. Paired-bootstrap the arctic-vs-stella
   difference. **A cheap closed-form ranking, not a trained one** — say so, since the released table
   is trained and phase 2 showed training moves it.
2. `encode_dev.py nq-250k hotpotqa` then `pool.py` under `M7_ENCODER=arctic-embed-l` — the long
   job (~3 h; 6.17M docs at 1024-d). Gates already passed for arctic: `validate_encoder.py`,
   `test_encoders.py`, `test_init_rows.py` (min cosine 0.99999986).
3. Re-run `stage0_ridge` + `gate.py` under arctic. Its refs file is separate
   (`work/devres/refs-arctic-embed-l.json`), so nothing mixes teachers.
4. Then the non-absorbable retention levers, cheapest first: count saturation, doc-side instruction.
   `phase2_negatives` is **demoted** — mined hard negatives lose to random at matched lr.

## The bars, and the tension in the teacher choice

`results/m7_bars_after_swap.json` (arithmetic only). Retention today is **0.7853**; arctic needs
**0.863** for Tier 2 and **0.824** for Tier 1 fused, i.e. +7.8 and +3.9 retention points. Stella
would need 0.824 / 0.787 — *less*, because it projects higher on the six.

So the two signals disagree: **measured** on our own dev components arctic is the best of five
(+0.0447 [0.0339, 0.0557] over bge-base; +0.0125 [0.0008, 0.0241] over stella, which would not
survive multiplicity over the ten pairs) and it is the only candidate disclosing **zero overlap with
our six**, where stella lists ArguAna and FiQA2018. **Projected**, stella maps ~0.025 higher onto the
six. The projection is known to mis-order — arctic has the lowest MTEB v1 of the three 1024-d
candidates and the highest measured macro, and bge-large ties bge-base across a 1-point MTEB gap —
so it is a scale, not a ranking. Run (1) before spending three hours on the loser.

Corrections Codex forced on the previous version of this file: the PI half-widths are **0.02818 /
0.03294 / 0.03446** (bge-base / gte-large / stella), not 0.024/0.030/0.035; **"bge-base cannot clear
Tier 2 at any retention" was false** — its teacher-only lower bound clears above ~0.955; and
multiplying a dev-macro retention by projected teacher PI endpoints does not compose the two
uncertainties, because the table macro is `mean_i(r_i x teacher_i)`.

## What phase 2 settled

`results/m7_phase2_screen_cis.json`, all arms from one fixed p1-objB checkpoint:

- **The contrastive objective was never broken — phase 1 measured its learning rate.** At 5e-5 to
  3e-4 it *improves* the table, CI-resolved; those three arms are mutually unresolved, so the
  optimum is a **band, not a point**. At 1e-3 the gain shrinks; at 3e-3 (phase 1's lr) it is
  negative and unresolved. The kill criterion may not fire and `contrastive_verdict()` records that
  as a file, not a memory.
- **Mined hard negatives HURT** at matched lr: random-only +0.0034 [0.0019, 0.0049]. The mandate's
  "exploit the cheap enormous negative pool" premise is restored; the ledger's "scale without
  hardness wasted the objective" is withdrawn. `fn_masked_frac` is 0.0051, so the false-negative
  filter is not the mechanism — this file previously predicted it would "bite far harder", and it
  does not.
- **The gain is small**: +0.0111 proxy macro, against the +3.9 to +7.8 retention points the bars
  need. Real, cheap, and not a tier-changer alone.
- An arm's final-step macro is **not** its best-step macro (3e-4 peaks at step 500). Fix the step
  budget in the config or select on best-eval consistently, and say which.

## Still open from the Codex gate (full list + dispositions in `LEDGER.md`)

Four blockers, none of which stop the encodes, all of which stop a *claim*: decontamination indexed
only the 855K positives while training touches the whole 6.17M-doc pool; the bootstrap p-values are
percentile tails, so Holm controls nothing; the frozen fusion function differs between dev selection
and final scoring; and the two-access rule is already breached (`bench_throughput` read FiQA qrels),
now logged.

## Open for Dylan

1. **Nothing is blocking.** The teacher ruling is made and logged.
2. **Host:** Windows Update rebooted the box mid-morning once already.
3. Later: HF release go. If (1) reverses the teacher choice, that comes back to you — it would mean
   shipping a teacher trained on 2 of our 6 eval sets, which is a credibility call, not a technical one.
