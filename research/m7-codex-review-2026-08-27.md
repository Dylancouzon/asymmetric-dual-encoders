Stop the ablation launch. The final benchmark claim is still salvageable, but the current dev-selection record cannot support the causal story being attached to it.

## BLOCKER

1. **The six-component dev statistics treat duplicated queries as independent.**

`heldout-longq` is literally a subset of `heldout-train` ([heldout.py](/home/dylan/asymetric-dual-encoders/m7src/heldout.py:60)). Yet `signflip` generates independent signs per component ([boot.py](/home/dylan/asymetric-dual-encoders/m7src/boot.py:57)), and `paired` independently bootstraps each component ([boot.py](/home/dylan/asymetric-dual-encoders/m7src/boot.py:95)). The same 55 queries therefore contribute twice to the macro while their covariance is discarded.

Concrete failure: the s2500 decision is only +0.0023 [0.0012, 0.0035], and heldout-longq contributes +0.0062 while receiving a full one-sixth macro weight. Omitting its positive covariance with heldout-train can narrow that interval and lower the sign-flip p-value. This could reverse tonight’s most marginal decision.

Before more selection, recompute all three lever comparisons with one shared sign per underlying held-out qid. For bootstrap, stratify heldout-train into long/non-long, resample each stratum once, and reuse the long draw in both component means. Report ordinary and dependence-preserving results side by side.

2. **The claimed “pinned six-component dev suite” pins only four components.**

The mandate requires hashes before candidate results ([instructions-m7.md](/home/dylan/asymetric-dual-encoders/instructions-m7.md:63)). But the asset freezer iterates only `devsuite.COMPONENTS`, the four text-backed sets ([freeze_m7_assets.py](/home/dylan/asymetric-dual-encoders/m7src/freeze_m7_assets.py:74)). The committed dev manifest likewise contains only those four. Meanwhile, the evaluator silently adds held-out components according to current file existence ([dev_eval.py](/home/dylan/asymetric-dual-encoders/m7src/dev_eval.py:23)), and those files can be rebuilt from the current mix and pool ([heldout.py](/home/dylan/asymetric-dual-encoders/m7src/heldout.py:37)).

Concrete failure: a changed pool order, training mix, held-out JSON, or missing file changes the selection statistic while `freeze.py` still verifies the same manifest hash.

Pin now, before ablations:

- Ordered qids, query texts, qrels/positive indices and long-query membership.
- Ordered pool/corpus identity and teacher/vector-cache identity.
- Exact hashes of both held-out JSONs.
- An explicit six-name component list; missing components must abort, never disappear.

This is necessarily a late repair. Disclose that the two held-out components were deterministically defined but not cryptographically pinned before earlier selection.

3. **The planned prefix ablation is not the mandated runtime-prefix ablation.**

Changing `preproc` in `phase4_mandatory` ([program.py](/home/dylan/asymetric-dual-encoders/m7src/program.py:230)) flows directly into `get_init(cfg.init, pre)` ([train.py](/home/dylan/asymetric-dual-encoders/m7src/train.py:370), [train.py](/home/dylan/asymetric-dual-encoders/m7src/train.py:447)). `teacher_rows` then embeds every vocabulary token inside that prefix context ([init_table.py](/home/dylan/asymetric-dual-encoders/m7src/init_table.py:41)). The arm therefore changes both runtime tokenization and the initial row geometry.

The mandate explicitly says fixed runtime-prefix variants are mandatory while prefix-conditioned rows are exploratory ([instructions-m7.md](/home/dylan/asymetric-dual-encoders/instructions-m7.md:45)). This arm violates that distinction.

Add a separate `init_preproc`/`prefix_conditioned_init` field. The mandatory comparison must keep identical no-prefix teacher rows and toggle only runtime prefixing. A prefix-conditioned-row arm may be a separately labelled exploratory third arm.

4. **`phase4_mandatory` cannot reproduce the winning two-stage recipe.**

One `Cfg` has one row LR, one weight LR and one schedule ([train.py](/home/dylan/asymetric-dual-encoders/m7src/train.py:44)). More importantly, a single objective-C run keeps one optimizer and cumulative update counts across B and A ([train.py](/home/dylan/asymetric-dual-encoders/m7src/train.py:463), [train.py](/home/dylan/asymetric-dual-encoders/m7src/train.py:638)). The winner used separate processes: B at 3e-3 constant, then A with a fresh optimizer/update counter at 1e-3 and warmup-linear.

Thus an objective-C “full chain” differs in LR, schedule, Adam state and regularization scaling. Every arm must be two separate runs: its B checkpoint, then a fresh A run initialized from that exact B artifact.

## MAJOR

1. **The sequential p-values are not confirmatory evidence.**

All model selection is allowed on dev ([instructions-m7.md](/home/dylan/asymetric-dual-encoders/instructions-m7.md:64)), but the only confirmatory decisions are the three final-run comparisons ([instructions-m7.md](/home/dylan/asymetric-dual-encoders/instructions-m7.md:68)). The 500k adoption, 2m pick and s2500 extension repeatedly use overlapping dev data, with the step first selected on a proxy that is itself half of the full suite ([LEDGER.md](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:485), [LEDGER.md](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:520)).

A hostile reader will say:

- Pre-registering the next comparison after seeing the previous curve is adaptive model development, not prospective confirmation.
- Rerunning the selected step does not make the selection set independent.
- “Cross-arm pick” versus “adoption bar” is bookkeeping, not a statistical distinction.
- Query-bootstrap p-values condition on the trained artifacts and ignore training stochasticity.

Post-hoc Holm over just the three reported p-values would numerically pass even at α=.025, but that does not repair proxy step selection, prior searches, duplicated held-out queries, or adaptive extension.

Cheapest fix: label every dev p-value and CI as exploratory selection evidence. Do not claim “pseudo-query coverage was statistically confirmed.” The final three frozen-test comparisons remain the only confirmatory claims.

2. **“Pseudo-query coverage caused +0.0126” is confounded.**

The sequence changes more than coverage:

- Original winner: B 8k → A selected at 1000.
- 500k: B 8k with pseudo mix → A selected at 1500.
- 2m: both pseudo-pool size and B steps change, 8k → 16k.
- Final: A changes again, 2000 → 2500.

See [LEDGER.md](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:485), [LEDGER.md](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:510), and [LEDGER.md](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:528).

The valid conclusion is “adaptive recipe search selected a better dev artifact.” To attribute the gain to pseudo-query coverage, run a no-pseudo B16k → A2500 control. To claim a 500k→2m dose effect, also run 500k at B16k → A2500. These matched controls are more informative than `learned-noidf` or a 1e-2 regularization arm.

3. **The matrix evaluator is not exactly the released evaluator.**

The matrix path normalizes with a 1e-9 clip and has no fallback ([bigram_residual.py](/home/dylan/asymetric-dual-encoders/m7src/bigram_residual.py:56)). `QueryTable.forward` uses a 1e-6 threshold and substitutes the normalized CLS row for near-degenerate sums ([table.py](/home/dylan/asymetric-dual-encoders/m7src/table.py:100)). Sparse count multiplication also has a different summation order from `embedding_bag`.

Reproducing 0.5987 after rounding proves that ordinary baseline aggregate is close. It does not prove:

- Per-query equality.
- Candidate equality.
- int8 equality.
- Identical rankings near score ties.
- Correct near-zero behavior.

This matters for the +0.0023 comparison. Have the operator compare matrix evaluation against `dev_eval.eval_table(load_table(...))` for every candidate and variant, asserting exact qid sets and reporting maximum query-vector deviation, changed top-10 counts, maximum per-query nDCG deviation and macro delta. The final gate should use the released `QueryTable` path, not this matrix shortcut.

4. **The input-embedding init arm will have six dead rows.**

Stella’s input embedding matrix is padded to 30,528 rows while the tokenizer has 30,522; the registry explicitly documents this ([encoders.py](/home/dylan/asymetric-dual-encoders/m7src/encoders.py:69)). But `input_emb_rows` returns the entire matrix without slicing ([init_table.py](/home/dylan/asymetric-dual-encoders/m7src/init_table.py:62)).

The arm will train and save a 30,528-row artifact while coverage accounting says 30,522. Slice/assert exactly `tok.vocab_size` before spending on that mandatory arm.

5. **The bigram fit is not provenance-bound.**

`bigram_residual_fit.npz` is one global cache ([bigram_residual.py](/home/dylan/asymetric-dual-encoders/m7src/bigram_residual.py:76)). Reuse checks only K and λ ([bigram_residual.py](/home/dylan/asymetric-dual-encoders/m7src/bigram_residual.py:94)); it does not bind winner bytes, teacher spec, TRAIN query hash/order, preprocessing or bigram vocabulary.

Concrete failure: a stale residual fit can be scored against the correct current baseline, still reproduce baseline macro 0.5987, and falsely kill the lever. The λ≠.01 reruns give supporting current-fit evidence, but the committed adoption artifact itself is unauditable. Record those hashes and the actual bigram map, or rerun into a content-addressed cache.

6. **The comparison artifacts omit what review needs.**

`stats()` discards the bootstrap’s per-dataset CIs and raw per-query vectors ([bigram_residual.py](/home/dylan/asymetric-dual-encoders/m7src/bigram_residual.py:69)); `compare_release.py` writes only rounded macros and summary statistics ([compare_release.py](/home/dylan/asymetric-dual-encoders/m7src/compare_release.py:42)). That prevents dependence-aware recomputation from committed results and violates the expected per-dataset-CI reporting shape.

Save per-query values or a hash-addressed compressed artifact, exact unrounded macros, per-component CIs, encoder fingerprint, table hashes and evaluator version.

7. **The doc2query generation cache does not identify the generation recipe.**

The resumable file name binds only component and N ([doc2query_probe.py](/home/dylan/asymetric-dual-encoders/m7src/doc2query_probe.py:41)). Existing contents are returned without checking generator revision, `max_in`, `max_out`, `top_k`, library version or generation hash ([doc2query_probe.py](/home/dylan/asymetric-dual-encoders/m7src/doc2query_probe.py:42)). The result records only model ID and N ([doc2query_probe.py](/home/dylan/asymetric-dual-encoders/m7src/doc2query_probe.py:69)).

The actual generation/evaluation mapping otherwise looks correct. But at p=.085, stale expansions could change the closure. Verify the JSONL was created entirely under this recipe and record its SHA-256. The “closed at the cheap-test price, not disproved” wording is correct.

8. **STATUS is stale enough to launch the wrong work.**

[STATUS.md](/home/dylan/asymetric-dual-encoders/m7/STATUS.md:3) still calls `s2w-1e3-s1000` the winner and says pseudo-query arms are running, while [LEDGER.md](/home/dylan/asymetric-dual-encoders/m7/LEDGER.md:535) names `p35w-2m-s2500` and says ablations are next. The one-screen operator contract is currently broken.

## MINOR

- `phase4_mandatory` runs the same baseline under `teacher`, `noprefix`, and `learned`, wasting three nominal arms ([program.py](/home/dylan/asymetric-dual-encoders/m7src/program.py:227)). Keep one baseline replay if you want a nondeterminism estimate.
- The “reg on/off” grid is actually 0 versus 1e-2, while the winning recipe uses 1e-3 ([program.py](/home/dylan/asymetric-dual-encoders/m7src/program.py:235)). The on control is 1e-3.
- Int8 is evaluation, not a training arm. Evaluate each relevant artifact separately; do not count it toward GPU-arm count.
- `compare_release.py` does not store the active encoder fingerprint, although the active encoder defaults to bge-base when `M7_ENCODER` is unset ([encoders.py](/home/dylan/asymetric-dual-encoders/m7src/encoders.py:169)). Matching 0.5987 is strong evidence tonight’s runs used the correct Stella caches, but the JSON does not prove it.

## Ablation design to run

Use separate B→A runs, fixed at B16k and A2500, with no per-arm best-step selection. One variable changes per chain.

The minimal set is:

- One baseline replay, optional but useful given the documented CUDA nondeterminism.
- Input-embedding init.
- Random init.
- Runtime prefix only, with identical no-prefix teacher-row initialization.
- Flat weights.
- Learned weights with uniform rather than IDF initialization.
- Regularization off; baseline replay supplies regularization on at 1e-3.

That is seven chains including one repeat, not ten. Do not run a 1e-2 reg arm unless it is explicitly exploratory. Do not run a factorial: the mandate asks marginal controls, not interaction estimation.

If the report intends to credit pseudo-query coverage, replace lower-value extras with the matched no-pseudo B16k→A2500 control, and preferably 500k-at-B16k→A2500.

## What the final gate must do

The gate cannot repair adaptive dev reuse. Its defensible role is a mechanical eligibility audit after all selection:

- Use the exact frozen release artifact through `QueryTable`.
- Verify full encoder fingerprint, table/meta hashes and all six dev component hashes.
- Abort on any missing component or qid.
- Save unrounded per-query fp16 and int8 scores.
- Recheck G3 on the four text-backed components.
- Recheck int8 equivalence with dependence-preserving handling of the nested held-out components.
- Directly compare the final candidate with `s2w-1e3-s1000` as an exploratory audit, not confirmation.
- Freeze immediately afterward; no recipe change after seeing the gate.
- Leave the three six-set Holm comparisons as the only confirmatory inference.

If the final test clears the release bar, the system claim is sound despite ugly dev adaptation. What will not survive is a causal claim that pseudo-query coverage, the 2m dose, or the 2500-step extension was independently significant.

## Ideas

1. **Count saturation is the cheapest genuine capacity lever.** The repo already proves unique-token pooling is non-absorbable ([absorb_check.py](/home/dylan/asymetric-dual-encoders/m7src/absorb_check.py:80)), but it has not been tested. Pre-register binary counts, cap-at-2, and sqrt-count pooling. This needs no retraining and no extra table rows.

2. **Contrastive-only bigram residuals.** Freeze final unigram rows, initialize the fixed top-10k bigram rows at zero, and train only those rows under objective A. The failed closed-form integration used exactly the supervision known to undo A gains; an A-only residual is the direct test of unused phrase capacity and costs roughly 20.5 MB.

3. **TRAIN-count-conditioned row interpolation.** Interpolate rare A-phase rows back toward their B checkpoint using a fixed function of training update count, applied in folded-row space. It adds zero query cost and directly targets cross-query interference on rare vocabulary. Pre-register only a tiny fixed family of curves.

4. **One B-phase extension is better motivated than another A extension.** The 2m B proxy was still rising materially at 16k ([m7_run_p35b-2m.json](/home/dylan/asymetric-dual-encoders/results/m7_run_p35b-2m.json:115)). If spending one exploratory chain, extend B under a fixed cap and then run the already frozen A2500 recipe. Do not call the A edge “the recipe edge” while B remains unbracketed.
tokens used
