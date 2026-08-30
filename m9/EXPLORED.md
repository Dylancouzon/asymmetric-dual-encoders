# M9 closed avenues, each with what would reopen it

Written so a future session does not re-derive a decision that was already made, or re-open one on
a premise that has already been tested. Every row is a thing M9 deliberately did **not** do.

## Data

| avenue | why closed | reopens if |
|---|---|---|
| **FineWeb** as regression text | Dylan approved it; the mandate permits it only against *pre-existing* reserved-set **document** fingerprints, and none exist — M7 persisted query-side fingerprints and R3 *counts* only, discarding the streamed document index. Building one now would open reserved corpora | a reserved access is spent for another reason and document fingerprints are persisted as a by-product. **M10 at the earliest** |
| **MS MARCO** | terms are non-commercial-research-only; permanently out of the release stack since M7. Priced there at +0.0058 [−0.0015, +0.0131] on the six — unresolved and opposite in sign on dev | a licence change. Not a research question |
| `nqopen` / `triviaqa` / `pseudoq` | **not closed — resolved.** Excluded at M9.0 because re-screening needed a containment index only the allowlisted module could rebuild; `m9src/extended_screen.py` then admitted 1,143,936 texts (220,528 of them real questions) with no new capability | — |
| A **G2 allowlist extension** for `m9src.extended_screen` | authorised by Dylan and **reverted unused**: `m8src/protected_filter` claims its entry at import time and is the only module that opens a protected path, so importing it both grants and performs the read. `m8src/paths_guard.py` is byte-identical to M8's | a future source needs a contact class M8 never sanctioned. Adding an entry is a ledger amendment, not an edit |

## Recipe and screen design

| avenue | why closed | reopens if |
|---|---|---|
| **Batch 32 vs 128 pilot** | registered, then removed before any arm ran. Two matched epochs give batch 32 four times the optimizer updates and compress a separate warmup+cosine schedule into a miniature, so it measures early optimization speed, not the batch size that wins at final dose. LEAF's `bs=32` finding was made at ~100× this dose | the build's own dose regime changes materially, where LEAF's evidence would apply directly |
| **Stage-B reorder** (student/prompt/mix before the teacher arms) | adopted under time pressure, then **withdrawn**: teacher-first *prevents* invalid downstream experiments while "void later" runs them and promises to discard them. Not equivalent, and the time pressure that justified it disappeared | nothing. The mandate's order is restored |
| **`MDE = max(0.0051, 2F)`** with F a seed-replica range | withdrawn. A single absolute difference between two seeds is one half-normal draw, not an estimated σ — it can sit near zero under large real variance or inflate the threshold arbitrarily (`m8/CODEMAP.md` pitfall 18 at K=2) | never in this form. A variance estimate needs 3–5 seeds and a preregistered estimator |
| **Capacity probe** (109M student) | registered, authorised, **withdrawn before running** on Dylan's ruling. 60–70 minutes for a question M9 cannot act on under the ≤35M cap, against an hour that bought resume-equivalence testing for the seven-day build. `m9src/capacity_probe.py` is left intact | **M10**, where a larger student is in scope. Run it unchanged |
| **Head+tail long-query probe** | will not run in M9. The query pool's longest text is 108 words against ArguAna's 174-word average, so long-query *training* coverage is absent and is stated as a limitation rather than patched | M10. `heldout-longq` may not change any M9 decision |
| **>35M student** | mandate cap. A ~110M student would retain more but competes with `arctic-m` symmetric rather than with LEAF, and muddies the pair | M10+ scoping, Dylan's call |

## Build design

| avenue | why closed | reopens if |
|---|---|---|
| **Cosine-to-a-horizon** schedule for the build | a cosine commits to a length: stop early and the LR is still high, want more and you cannot extend. Replaced by warmup → stable → decay-on-demand, so the run length becomes an observation | the stable phase misbehaves — which is why it carries a hard token cap plus automatic plateau and regression stops |
| **20/10/70 token mix** | drafted, then rejected for *repetition*, not weight: it gives the 463,314 real queries ~438 presentations. At 5/5/90 they get 109.6, and because queries are short they are still ~23% of the objective under a true example mean | evidence that 109 presentations is under-training the query manifold — which the in-run SCREEN-3 curve would show |
| **Document-side co-adaptation** (`E14-LORA`) | `m8/FINDINGS.md`'s one untested high-capacity lever, and explicitly out of M9 scope: training the document tower breaks the one-index/two-query-paths pair that is the product | post-M10, with a real budget, as its own system |
| **A higher-dimensional or MRL stella index** | stella's alternative dims are separate learned heads, so a smaller index is a separate system and a full re-encode, not a free truncation | M10+ as its own system |
| **Re-deriving `zero` against a stronger teacher** | `T1`: a teacher's own retrieval quality does not predict its distilled *table* (Spearman 0.000 over eight candidates). Discards a frozen, confirmatory-verified artifact for a coin flip | the teacher screen fires AND the 400M ceiling is the diagnosed cause |

## Measurement

| avenue | why closed | reopens if |
|---|---|---|
| **fp16 ONNX parity threshold** | the shipped fp16 graph scores 0.99953 against a locked 0.9999. Recorded as a **fail** rather than re-thresholded after seeing the number — the threshold was written for same-precision export fidelity, not for a precision change | M10, with a **preregistered retrieval-impact tolerance** measured before the number is seen, the way M7 priced its int8 table as quality-free |
| **TurboQuant (int4) benchmarks** | Dylan's ruling: it is Qdrant's preferred method and belongs in the whitepaper's all-in comparison against binary/int8/fp16 on latency, footprint *and* recall — not piecemeal now | the whitepaper. 1M documents is the confirmed upper bound |
| **Recall under quantization** | measured nowhere on the Mac by design: device-dependent numerics mean quality numbers come from one machine. The Mac established the *cost* side of the trade | the training box, against the real stella index, once nano exists |
