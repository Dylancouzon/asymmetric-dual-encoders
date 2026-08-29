# Codex adversarial review — E14-HEAD implementation design

Model `gpt-5.6-sol`, reasoning effort high, read-only, 2026-08-29. Brief:
`research/m8-planning/e14-head-design-2026-08-29.md`. Verdict: **the probe should not run as
designed** — three BLOCKERs. All three were reproduced independently before being adopted; the
dispositions are in `m8/LEDGER.md` §15 and the amended registry row.

Findings verbatim below.

---

## Verdict

**The probe should not run as designed.** I found three protocol BLOCKERs and several MAJOR interpretation/engineering gaps. The raw false-negative masking idea is sound, and cross-process scoring does not inherently destroy pairing, but those strengths do not rescue the current design.

## BLOCKER findings

### BLOCKER 1 — The registration’s algebraic premise is false and contradicts the project’s own evidence

The registry says a linear document map is absorbable and therefore measures nothing ([registry.json](/home/dylan/asymetric-dual-encoders/m8/registry.json:410)). But the binding ledger already records the opposite for the system actually proposed:

\[
q^\top \frac{Md}{\lVert Md\rVert}
\]

contains the document-specific factor \(1/\lVert Md\rVert\), which cannot be absorbed into a shared query table. The ledger explicitly calls the absorbability dismissal “half wrong” and records rank agreement 1.000 without document renormalization versus 0.000 with it ([LEDGER.md](/home/dylan/asymetric-dual-encoders/m8/LEDGER.md:878)); the underlying result says the same in [m7_absorb_check.json](/home/dylan/asymetric-dual-encoders/results/m7_absorb_check.json).

Consequences:

- “It must be nonlinear” is not established.
- A normalized linear head is genuine document-side capacity and is a cheaper, easier-to-optimize probe: about 1.05M parameters rather than 4.2M.
- An MLP win could be achieved mostly through its effective linear anisotropic map plus renormalization, so it cannot be interpreted as evidence that *nonlinearity* mattered.

Amend the registration before running. Either make the normalized linear residual head the cheap first stage, or retain the MLP but add the normalized linear arm/control and stop describing linear as a no-op.

### BLOCKER 2 — Zero-init does not make the proposed system identical to R0

At \(W_2=0\), the output is:

\[
\operatorname{normalize}(d),
\]

not \(d\). R0 scores the cached fp16 vectors directly in both training ([train.py](/home/dylan/asymetric-dual-encoders/m7src/train.py:679)) and evaluation ([evalkit.py](/home/dylan/asymetric-dual-encoders/m7src/evalkit.py:16)).

The cached vectors are only approximately unit length. I sampled 100,000 pool vectors:

- only 0.36% had float32 norm exactly 1;
- maximum norm error was \(4.8\times10^{-5}\);
- mean absolute error was \(9.1\times10^{-6}\).

That may be numerically small, but it defeats all three claimed guarantees:

- step zero is not bit-identical;
- the actual initialized-head equivalence test cannot pass on arbitrary random inputs;
- existing B3 R0 rows are not the exact comparator to the normalized-head treatment.

Because normalization also changes Phase-A logits and therefore training trajectories, merely rescoring R0 with normalized documents is insufficient for the cleanest contrast. The defensible fix is a three-seed **identity-normalized comparator** trained through the same patched path, with the head frozen at identity. That also gives you an end-to-end harness null. Otherwise drop the “R0 plus capacity/exact identity” claim and treat normalization as a second intervention.

### BLOCKER 3 — The LR ladder will evaluate the forbidden dev endpoint

`train.run()` evaluates configured dev components every `eval_every` steps and then unconditionally evaluates them again at the end ([train.py](/home/dylan/asymetric-dual-encoders/m7src/train.py:725), [train.py](/home/dylan/asymetric-dual-encoders/m7src/train.py:770)). R0 has `eval_every=500` and includes both CQADupStack endpoints.

Therefore each LR ladder arm will expose endpoint results several times before the LR is selected—even if the selection script promises to ignore them. Setting `eval_every=0` is not enough because final dev is unconditional; passing an empty component list also falls back to the full suite.

The ladder subprocess must mechanically prevent all dev access, for example by replacing the final-dev evaluator with the training-holdout evaluator and asserting that no dev corpus was opened. This must be registered and tested before any ladder arm.

Two related protocol fixes are also required:

- Do not select on seed 0 and then include seed 0 in the reported three-seed mean. That creates selection-on-training-randomness bias. Use a disjoint tuning seed, such as seed 3, then evaluate seeds 0/1/2.
- State explicitly that the final three arms return to the full pair pool. If they retain the 98% ladder pool, the comparison is confounded against full-pool R0.

## MAJOR findings

### MAJOR — A positive does not specifically identify bag reachability

The head is supervised document-side metric learning. It can improve retrieval because it:

- corrects defects in the teacher’s relevance geometry;
- learns source/style/domain separation;
- calibrates angular density and margins;
- co-adapts to labels more generally.

None of those requires the original problem to be “documents were unreachable by a bag.”

The bank makes source shortcuts plausible: HotpotQA is about 85% of the document pool but only about 24% of positive pairs, while FEVER and SQuAD are heavily overrepresented among positives relative to random negatives. A shared head can lower InfoNCE by separating such domains. This is not a mask exploit, and the CQA endpoint tests transfer, but it still does not isolate bag-specific reachability.

Add a cheap mechanism control during the same corpus passes:

- raw documents + frozen teacher queries;
- headed documents + frozen teacher queries;
- raw documents + R0 bag;
- headed documents + E14 bag.

The difference between the bag gain and teacher-query gain is the evidence about *bag-specific* reachability. Without it, a positive is useful evidence for supervised document-side adaptation, but only indirect evidence for the E14 mechanism or LoRA specifically.

### MAJOR — Belief 5 is unsupported

Parameter count and number of examples do not prove 2,500 steps is adequate, especially with:

- delayed \(W_1\) learning;
- joint table/head optimization;
- a decaying LR;
- an unregularized head competing with a table pulled toward initialization.

Before opening the endpoint, continue the winning tuning-seed checkpoint from 2,500 to 5,000 steps and evaluate only the training holdout. Pre-register a plateau rule. If the validation statistic is still materially improving after 2,500, the primary probe must report **optimization-inadequate / UNINFORMATIVE**, not a method null.

Also, held-out InfoNCE is poorly aligned with the shipped endpoint: training uses mean pooling and fp32 parameters, while the decision uses int8/sqrt nDCG. Prefer a fixed-candidate ranking statistic using the released int8/sqrt query path. Freeze its candidates, mask, positive choice, draw count, and tie rule.

The holdout is also only “held out from E14 Phase A”: `p35b-2m` already trained on all those queries and used positive documents in its KL candidate sets. Word it that way; a genuinely untouched holdout would require rebuilding Phase B.

### MAJOR — Document evaluation must be streamed

A naïve `doc_vecs` patch that materializes headed float32 vectors requires approximately:

- 21.4 GB for HotpotQA;
- 25.3 GB for the 6.17M-document pool.

That exceeds the project’s 18 GB RAM limit. The current scorer deliberately slices raw vectors inside `topk_arrays` ([evalkit.py](/home/dylan/asymetric-dual-encoders/m7src/evalkit.py:29)).

The patch must return a lazy slice-transforming object or use a head-aware chunked top-k path. Each patched process must emit only its one head arm; any R0 row produced in that process would also receive the head and must be rejected.

### MAJOR — Persistence and scoring provenance are too weak

A `run_id` inside `dochead.pt` does not prevent a wrong or stale head from being paired with a table. Bind and verify at least:

- table artifact SHA-256;
- Phase-B checkpoint SHA-256;
- head state SHA-256;
- exact architecture, dtype, normalization epsilon and initialization;
- head LR, seed and schedule;
- training/validation split hash;
- M8 patch-source hashes and registry stamp.

The current evaluation code-identity record hashes M7 evaluator sources, not the new M8 monkeypatch code ([dev_audit.py](/home/dylan/asymetric-dual-encoders/m7src/dev_audit.py:67)).

## Other answers

- **False-negative reward hacking:** keeping the mask in raw teacher space is correct. The ID-based own-positive/all-positive masks are also independent of the head. I found no comparable direct loss-hacking channel after document normalization.
- **Distribution shift:** a CQA gain is genuine transfer beyond the five training sources, but both dense components are from the same forum family. It licenses a CQADupStack-family claim, not broad OOD reachability.
- **Pairing across processes:** separate processes do not break seed pairing. Reusing R0 does break the exact contrast here because of the normalization intervention. A later per-query bootstrap measures query sampling, not training-seed uncertainty; seed uncertainty must be reported separately or with a two-level bootstrap.
- **C2:** no redefinition occurs provided D1 remains disabled for C2, exactly as the ledger requires. A head win cannot satisfy strict C2 or the qualifying-table condition. The query artifact remains int8; the head enlarges the document ONNX and its bytes/precision must be reported separately.
- **B6-pre:** align and run it before the training campaign, using nondegenerate MLP weights so parity exercises the MLP rather than an identity/zero path. Export and parity-test the actual trained head again before calling it shippable.

My cheapest recommended sequence is: normalized linear head first, identity-normalized comparator, dev-blind tuning on a disjoint seed, pre-endpoint 5,000-step adequacy continuation, then three full-pool paired seeds with the teacher-query mechanism control. If you retain the MLP, the other fixes remain mandatory.
