# M10 findings — preparation and recipe screening

Read with `m10/RESULTS.md`; result JSONs are authoritative. These findings do not forecast a
release pass. The 200M build, both cloud E arms and final evaluation remain M13 work.

| Observation | Supported interpretation / limit | Evidence |
|---|---|---|
| F1 favors bge-small over MiniLM-L6: +0.011595 | The registered 20M comparison selected bge-small, not a universal backbone ranking | `results/m10_contrast_F1.json` |
| A4−A3: +0.012080 at 5M; MedicalQA contributes 81.8% | Bundled distribution/exposure change; seven forms added while existing forms receive fewer presentations. Not an isolated synthetic-vs-real or causal coverage test | `results/m10_contrast_A4-A3.json` |
| A4−A3: +0.016453 at 20M | The observed gain persists at this dose; BRIGHT's contribution becomes positive. One descriptive pair, no interval; no 200M guarantee | `results/m10_descriptive_corpus_at_20M.json` |
| G1 (1152 vs 384): +0.021651 | Width helps this screened recipe. The PCA probes do not bound all retrieval subspaces or prove normalized-L2 convergence | `results/m10_contrast_G1.json` |
| G2/G3, B1/B2, D1 do not resolve | Registered defaults retained. Failure to reject superiority is not a measured null or equivalence | `results/m10_screen_verdicts.json` |
| D2 is negative; its gradient scale differs greatly | Mechanism unresolved; no retrospective clip-rate evidence from those runs. Neither a clean class rejection nor a proven optimizer artifact | `results/m10_dcov_gradient_audit.json` |
| Second anchor seed: +0.000712 | One sensitivity observation; not bounded seed noise, replication of all contrasts or seed-adjusted significance | `results/m10_descriptive_seed_sensitivity.json` |
| Nano uses 34.54M parameters and bge-small's backbone | It must earn quality at comparable query cost and shared-index compatibility. Near-zero query compute belongs to zero | `results/m10_head_width_parity_mac.json` |

## Harness lessons worth carrying forward

- One decision constant needs one executable home. Keep registries authoritative and prose short;
  check artifact provenance without re-stamping completed experiments for cosmetic doc changes.
- Test the production caller: a passing trainer-resume test does not prove the arm CLI can resume;
  tested decision helpers do not provide a scoring executor. `m13/EXECUTION.md` owns the gaps.
- Form balance reallocates exposure. At build dose, small generated forms repeat far more than
  the factoid pool. Report per-form exposure with the corpus intervention.
- Short-text diversity checks need an attainable threshold; an 8-gram rule requiring 16 matches
  cannot fire on a text with fewer than 23 words. The corrective rule is already registered.
- A fixed memory load is not per-row growth. Measure two sizes and inspect host free memory;
  cache readers must not truncate append-only stores; resume must restore stream positions.
- Train/eval mode, bf16, warmup and weight-decay groups must be exercised in the real caller.
  The calibration used dropout off; its seed difference is not a calibrated bound for later arms.
- Serving parity includes tokenization limits, padding and pooling. A correct graph with the wrong
  tokenizer metadata produces the wrong experiment.
- **Withdrawn 2026-09-10:** neither 0.002633 nor 0.001317 is a minimum nonzero nDCG@10 change.
  Rank changes can cancel. The old bridge needs validation, not an impossibility claim.
  Arithmetic and review: `research/m10-cleanup-review-2026-09-10.md`.

Historical owner rulings, failed diagnoses and full dispositions remain in the archived ledger;
closed avenues remain in `m10/EXPLORED.md`. Do not erase those records to simplify the claim.
