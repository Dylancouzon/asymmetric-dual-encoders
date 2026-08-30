# M8 status — CLOSED

**M8 closed 2026-08-30 as a MEASUREMENT.** No candidate, no release, **no confirmatory access
spent**; the reserved four are clean and inherited by M9. Twelve probes ran. **Closure was an owner decision not to spend, not the registered
exit proving exhaustion** — see §15's 2026-08-30 ruling. `R-LIST`, `B10`, a trained `D2` and
document co-adaptation remain UNTESTED.

**Read `m8/FINDINGS.md` first** — it is what this milestone produced. `m8/EXPLORED.md` is the
closed-avenue register with reopening conditions. Everything else here is provenance.

## The verdict

M7 shipped avg-6 **0.4339** against `LR-dense-pertask` **0.4583** — a CI-resolved **−0.0243** miss.
**No M8 lever improved the dev endpoint by more than ~0.005**, against a class whose historical
transfer to the six is **0.000 ± 0.005**. The gap is ~5x the largest effect this class produces.

Under a **frozen document tower**, the deficit was not repaired by finer query resolution (a
closed-form equal-budget screen, *not* a trained-capacity test), by query placement, or by target
design. What was never tested at capacity is **document-side co-adaptation**, which is exactly what
the system we lost to does.

| probe | result |
|---|---|
| `D2-PRE` | all four new-row classes negative; best −0.0028, D2's own −0.0052 vs a +0.00519 bar |
| `B8` | doc-centroid target −0.167; 50/50 blend −0.003 |
| `VECTOR-PRF` | −0.051 dense, −0.021 fused, negative on all six components |
| `E14-HEAD` · `T1` · `B3` · `B2` | −0.0244 · −0.052/−0.109 · +0.00135 at 4x dose · objective inert |

## Not run, deliberately

`E14-LORA` (authorised by Dylan; what was affordable here is a **proxy** for co-adaptation, not a
test — deferred to M9/M10 with a real budget), `E14-PRE` (registered, launched, **cancelled** on
the judgement that neither outcome could change the decision it gated), `R-LIST` (the one open
lever with a mechanism: `B2`'s `teacher_top200` measures 0.777 nats, so the KL *class* is open),
`B10`.

## Files

| file | contract |
|---|---|
| `FINDINGS.md` | **What M8 learned.** Whitepaper and next-model input. Read first. |
| `EXPLORED.md` | Closed avenues, each with what would reopen it. |
| `LEDGER.md` | The protocol, the rulings, and the measured results a bar reads. |
| `registry.json` | The pre-registration record. Rows are marked closed, never deleted. |
| `RESULTS.md` · `CODEMAP.md` | Runs · modules and the pitfalls this milestone earned. |

## Inherited by M9

Reserved four unspent · calibrated noise model (A-leg sigma 0.00106, chain sigma 0.00153 -> bars
0.0040 / 0.00519; a **pool-varying** lever remains unbounded) · `blockcg` Gram-free ridge ·
`probe_guard` · `decide` · `noise_floor` · `d2_pre`'s serving-exact encoder · dev-reuse counter
(**494** cumulative in-training dev evaluations, M7+M8).
