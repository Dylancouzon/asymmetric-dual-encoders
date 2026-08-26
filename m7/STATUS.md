# M7 status

**Stage:** Stage 0 complete · **go/no-go gate = GO** · next is phase 2 (negatives)
**Updated:** 2026-08-26

## Headline

A **pure-distillation lookup table, with no contrastive training at all**, retains **78.5%** of
its 109M-parameter bge-base teacher and beats BM25 by **+2.70 nDCG** on the dev suite at zero
query compute, shipping as a **23.4 MB int8** artifact whose quantisation is measurably free.
The mandate's central structural question — whether a *frozen, off-the-shelf* document space is
additively predictable from query tokens the way LightRetriever's co-trained one is — is answered
yes, first by a closed-form solve and then reproduced by gradient training.

## Gate: GO (`results/m7_gate_p1-objB.json`)

| condition | result |
|---|---|
| G1 Stage-0 table > potion (0.3801) | **PASS** d=+0.0994 CI=[0.0910,0.1078] p<1e-4 |
| G2 capacity probe > BM25 | **PASS** d=+0.5917 — trivially; see the caveat in LEDGER |
| G3 candidate > BM25 (0.4525) | **PASS** d=+0.0270 CI=[0.0188,0.0353] p<1e-4 |
| G4 int8 equivalence (bar 0.005) | **PASS** upper=0.00053 |

Dev macros, text-backed 4 components: candidate fp16/int8 **0.4795** · BM25 0.4525 ·
potion 0.3801 · teacher ceiling 0.6106.

## Objective grid (dev proxy macro-3)

| config | result |
|---|---|
| closed-form flat ridge (lambda=1e-2) | 0.4542 — provable optimum of flat MSE distillation |
| **p1-objB** distillation 8k | **0.4548** ← best; the gated candidate |
| p1-objC B(4k)→A(8k) | 0.3721 — the contrastive phase cost 7.3 points |
| p1-objA contrastive 12k | 0.3248 — monotone decline from 0.3532 |

**Contrastive as configured is destructive, from two different initialisations.** The stated
mechanism ("random negatives trivially separable") does not survive the author's own arithmetic
(loss ~3.4 at tau=0.02, not ~0) and is being re-diagnosed by measurement, not ablation —
`m7src/diag_scores.py` measures score geometry, softmax mass concentration per temperature, and
what fraction of the *hardest* negatives the `fn_margin` filter removes. **Run it first next
session.**

## Findings that constrain the report

1. **The untouched-final partition has no clean member.** Climate-FEVER dropped (no affirmative
   licence at any primary source); BEIR FEVER shares its corpus with fever-train by construction
   (11.3% TRAIN-positive overlap); DBpedia-entity — the intended clean probe — has **9.32%**.
   Structural: DBpedia abstracts and HotpotQA documents are both Wikipedia lead paragraphs.
2. **Dev cannot validate long queries.** Held-out length p50=13 WordPiece tokens, p90=24; only
   **55** of 7,325 reach the mandated >=64. ArguAna's are ~250. The ArguAna row will be an
   extrapolation and the learned-weight "long-query hypothesis" is untestable here.
3. **Learned per-token weights buy +0.0006** over the flat closed form (no CI attached yet).
4. **FiQA is the six-set row most at risk**: the ridge table retains only 64% of the teacher on
   cqadup-programmers vs 89% on nq-250k, and StackExchange-style retrieval is the nearest dev
   analogue to FiQA. FiQA is also where BM25 is weakest, so dense and fusion pull opposite ways.
5. **A mandate premise is wrong here.** "Frozen doc vectors make very large negative pools nearly
   free — exploit that first": scale without hardness wasted the contrastive objective.

## Not a crash

The 2026-08-26 reboot was **Windows Update** (Event 1074, TrustedInstaller, "Operating System:
Upgrade", 05:52), a clean shutdown with no Event 41/6008/bugcheck. Gate finished 03:03; box idle
~3 h before the reboot. Nothing lost. **Host action for Dylan: stop Windows Update rebooting
mid-run** (active hours / pause / "no auto-restart with logged on users").

## Next, in order

1. `m7src/diag_scores.py` — identify the contrastive failure by measurement.
2. `m7src/ridge_full_eval.py` — never ran; gives the closed-form table a full-suite,
   CI'd number comparable to the gate bars. Cheap (no solve).
3. Phase 2 negatives ablation (`program.phase2_negatives`) — load-bearing, not tuning.
4. Phases 3-5, fusion selection on dev, freeze (`m7src/freeze.py` → `m7/FREEZE.json`), final run.
5. Re-run the novelty freshness check before the report ships (mandate requirement).

## Open for Dylan

Nothing blocking. HF release go stays yours. Host-side Windows Update setting above.
