# Codex adversarial review — E14-HEAD IMPLEMENTATION, 2026-08-29

gpt-5.6-sol, read-only, reasoning effort high. Brief: the implementation against its own
pre-registration, with what I believed stated so it had something to attack. Verdict:
"the remaining campaign should pause" — 5 BLOCKER / 5 MAJOR / 4 MINOR. Disposition in
`m8/LEDGER.md` section 15, entry dated 2026-08-29 (later the same day).

Findings verbatim below; the full transcript including its file reads is not kept.

## Bottom line on your six beliefs

1. The present document-vector choke-point reading is correct; reported dense/fused documents are transformed once.
2. Raw-space masking closes only the mask channel. Bank membership, source priors, false-negative holdout scoring, and regularization arbitrage remain.
3. R0N is the right comparator. LIN is paired; MLP is not until initialization is seeded.
4. The ladder subprocess is dev-blind, but the overall scorer still permits explicit tuning-arm endpoint access.
5. `assert_fired()` does not prove head movement or external scorer reads.
6. Per-head LR selection is the reasonable reading. The fresh 5,000-step schedule is not a faithful continuation.

The remaining campaign should pause.
tokens used
230,816
## Verdict

Do not run the MLP ladder, adequacy arms, reported arms, or endpoint scoring as currently implemented.

At my final filesystem check, all three LIN ladder arms were already on disk, but I found no E14 endpoint-score artifacts. The repository also changed during this review; this verdict is against HEAD `e03fccb` plus the then-uncommitted change to `e14_run.py`.

## Document-vector trace

For an explicitly named reported arm, I found no current path that applies the head zero times or twice:

- Dense: `compare_full.main → multieval.eval_makers → dev_eval.doc_vecs → HeadedVecs → multieval`’s imported `topk_arrays → doc_vecs[lo:hi]`. The slice applies the head exactly once.
- Fused: `select_fusion.bm25_run_and_key` obtains a lazy proxy but ignores its vector field, so no transform occurs there. `fused_floor.main` then obtains the proxy used by `topk_ids_scores`, whose slices apply the head once.
- Mechanism: `dev_eval.doc_vecs` is deliberately left raw; `dv_head = HeadedVecs(dv_raw, head)` creates the headed side of the registered 2×2 control. Raw scoring there is intentional.
- In-training dev: the first `doc_vecs` call only gets query text; the second reaches scoring. Laziness means only the latter transforms rows.

Your import-binding analysis is correct: patching `evalkit.topk_arrays` would miss `multieval`’s early `from evalkit import topk_arrays`.

That said, scoring has no transform counter or post-score assertion, so this property is established only by today’s static call graph—not enforced.

## BLOCKER findings

1. **MLP learning-rate arms do not differ only in learning rate.**

The MLP is constructed at [e14_patch.py:297](/home/dylan/asymetric-dual-encoders/m8src/e14_patch.py:297), before `train.run()` executes `torch.manual_seed(cfg.seed)` at [train.py:434](/home/dylan/asymetric-dual-encoders/m7src/train.py:434). Its random `fc1` and bias survive initialization; zero `W2` only hides them in the initial output. The first `W2` gradient already depends on them.

Consequences:

- The three MLP ladder arms have different uncontrolled initializations as well as different LRs.
- The fresh adequacy arm does not reproduce the winning ladder initialization.
- Reported MLP arms are not reproducible from their recorded seeds.
- The claimed seed pairing is false for MLP.

Smallest repair: pass the arm seed into `install`, call `torch.manual_seed(seed)` before `build_head`, then let `train.run` reseed training as it already does. Test identical initial head-state hashes for equal seeds and different hashes for different seeds. Any MLP arm produced before that repair is unusable.

2. **The 5,000-step adequacy arm is not the registered continuation.**

[e14_run.py:98](/home/dylan/asymetric-dual-encoders/m8src/e14_run.py:98) starts a fresh 5,000-step run under a 5,000-step warmup-linear schedule. The registered arm says “continue the winning arm from 2500 to 5000.”

This is not a semantic nicety. At step 2,500:

- The actual 2,500-step winning arm is at the schedule floor: LR factor `0.1`.
- The proposed 5,000-step run is still at LR factor `0.5208`.

At step 1,250 the factors are `0.5435` versus `0.78125`. Therefore its 1,250/2,500/5,000 curve describes a different optimizer trajectory, and the 2,500 point is not the selected arm.

Smallest repair: resume the exact table, head, optimizer state, and RNG state from the selected 2,500-step run. If checkpoint continuation is impractical, amend the registration before endpoint access; the current “one fresh 5,000-step schedule” reading is not faithful.

3. **The holdout statistic is materially contaminated and mismatched to the endpoint.**

The fixed 8,192 negative rows at [e14_patch.py:380](/home/dylan/asymetric-dual-encoders/m8src/e14_patch.py:380) are not disjoint from the training bank. I reconstructed the actual sets:

- 2,627 of 8,192 holdout negatives—32.07%—are in seed 3’s training negative bank.
- Each bank row is sampled about 41 times in expectation over 2,500 steps.
- Nine holdout negatives are in `banned_pool_rows`.

Because this is a trainable document head, those evaluation negatives are directly optimized during training. A head can improve this statistic by learning to demote finite-bank membership without improving retrieval over the remaining 4.17M pool documents.

There are three additional mismatches at [e14_patch.py:383](/home/dylan/asymetric-dual-encoders/m8src/e14_patch.py:383):

- It uses training-time `mean` pooling, while the endpoint is `sqrt`.
- It evaluates the live fp32 table, while the endpoint is the folded int8 artifact.
- It disables own-positive, all-positive, and raw-teacher false-negative masks. The comment that masks would “drift” is wrong: with fixed rows and a raw-teacher mask, all three masks are arm-independent. Omitting them rewards pushing down valid/sibling positives.

This can select the LR best at suppressing seen random negatives under mean pooling, not the LR best at top-10 retrieval under int8/sqrt.

Smallest repair: reserve allowed negative rows disjoint from the training bank; restore the ID masks and raw-space `fn_margin=0.02`; use `sqrt` queries; and bind the negative-set hash. Ideally evaluate the release-shaped int8 query table or explicitly preregister the fp32 proxy. This selection construction should be amended before further ladder work.

4. **Tuning arms can still be endpoint-scored.**

The mid-review change made the default arm discovery enumerate reported arms, which is correct. But [e14_score.py:235](/home/dylan/asymetric-dual-encoders/m8src/e14_score.py:235) accepts arbitrary `--arms` without validating them. For example, an operator can explicitly score `m8e14-lad-lin-lr1e3` now.

That defeats “dev-blind before selection” at the workflow level even though the training subprocess itself is blind.

Smallest repair: reject every requested arm not in the exact reported allowlist, including through `--arms`. Also require a frozen selection/adequacy manifest before any reported scoring. Do not rely on the default discovery function as the security boundary.

5. **Recorded provenance is not enforced, and the live campaign already spans code vintages.**

`_exists` is only “does `<rid>.npz` exist?” at [e14_run.py:65](/home/dylan/asymetric-dual-encoders/m8src/e14_run.py:65). `select()` trusts sidecars without checking their config or source hashes. The score loader at [e14_score.py:75](/home/dylan/asymetric-dual-encoders/m8src/e14_score.py:75) checks only the table hash, and even that check is skipped when the recorded value is null. It does not verify:

- `head_file` or `head_state`
- current head/patch/run source hashes
- Phase-B hash
- registry hash
- exact schedule/config

The recorded schedule is only the string “inherits …”, and the split hash covers holdout indices but not the underlying pair identities. Scored outputs also do not bind the head hash, so stale scores can survive a retrained head.

This is live, not hypothetical: the three LIN ladder arms were written before the current uncommitted `e14_run.py` change, yet the planner will silently reuse them.

Smallest repair: make `_exists` mean “all expected config and provenance fields validate”; otherwise hard-refuse. Validate every recorded hash before selection and scoring, require non-null hashes, bind actual schedule values and pair identities, and attach head/table hashes to every score artifact. After that change, revalidate or rerun the existing LIN ladder consistently.

## MAJOR findings

1. **The finite-bank/source-prior loss channels remain open.**

Keeping the mask raw correctly closes mask inflation. It does not close:

- Finite-bank overfitting: negatives come only from a fixed 2M subset, while positives can come from the full pool.
- Marginal relevance/source priors: positives and uniform negatives have different source/document distributions even though both originate in the same cached pool.
- Regularization arbitrage: `reg_init` penalizes query rows only. The head is unregularized, so total training loss can fall by moving a global transformation from penalized query rows into the head without a corresponding retrieval gain.

The holdout shares the training source mixture and 32% of its negatives with the bank, so it does not reliably reject the first two.

Repair: report InfoNCE and `reg_init` separately; add a disjoint-bank diagnostic and source-balanced/source-conditioned retrieval controls. Do not interpret `bag_gain − teacher_gain` as uniquely identifying bag reachability: a source prior learned only by the co-trained bag table also appears “bag-specific.”

2. **`R0N` is the correct comparator, but the registered R0N-vs-R0 null is not implemented.**

For LIN and R0N, the training sampling, negative bank, query initialization, and Phase-B checkpoint are paired by seed. R0N is indeed preferable to rescored R0 because normalization changes training.

However, the scorer now enumerates only the nine E14 arms, while [e14_decide.py:188](/home/dylan/asymetric-dual-encoders/m8src/e14_decide.py:188) looks for `m8nf-seed*` inside those same inputs. It will report “R0 arms not present” rather than the registered end-to-end null.

Repair: load the existing R0 dense/fused scalars from their canonical artifacts separately and require all three paired comparisons.

3. **The mechanism control and actual-head export gate are optional in the final handoff.**

[e14_decide.py:213](/home/dylan/asymetric-dual-encoders/m8src/e14_decide.py:213) silently skips missing mechanism files, yet can still emit a strong positive headline. There is also no enforcement of the registered trained-head B6 parity test.

Repair: require all six treatment mechanism artifacts before a final positive interpretation. Require actual trained-head export/parity before any “shippable” statement.

4. **`assert_fired()` does less than claimed.**

It proves that Adam was constructed and a group was added, not that a trainable head moved. Movement is checked only by optional `collect()`. The external dense/fused scorer has no transform counter at all.

Worse, `collect()` walks directly into CODEMAP pitfall 17: it checks a treatment count only `if have`, so zero reported arms passes. It does not require the exact nine IDs or exact seed set.

Repair: make `assert_fired()` fail immediately if a trainable head has zero delta or a frozen head has nonzero delta. Make `collect()` enumerate and require all nine reported artifacts and validate their provenance. Add score-time row-transform counters and a post-score assertion.

5. **The Holm implementation is not the registered two-endpoint treatment test.**

The newly added decision code invents an external Gaussian p-value model not stated in the registry, then applies Holm using only the dense p-value at [e14_decide.py:255](/home/dylan/asymetric-dual-encoders/m8src/e14_decide.py:255). For an intersection-union treatment, its treatment-level p-value should be the maximum of its dense and fused p-values before Holm across LIN/MLP.

At the frozen 0.004 bar this probably cannot change a result because both inferred p-values are tiny, but it still is not the registered rule.

Repair: amend the row to define the p-values/noise model, then use `max(p_dense, p_fused)` per treatment. Also fix the headline path: `CLEARS-THRESHOLD-NOT-HOLM` currently falls through to the registered no-survivor/null headline.

## MINOR findings

- The current smoke covers ladder-LIN, reported-LIN, and R0N, but never executes MLP training or MLP streamed scoring. Smoke one MLP ladder and one MLP reported path after fixing seeding.
- `sweep.one` continues appending M8 arms to `m7/RESULTS.md`, reproducing CODEMAP pitfall 8. The worktree is already dirty there. Patch `sweep.append_row` in the subprocess or call the lower-level runner and write only the M8 record.
- The newly added frozen ladder reference was introduced after observing the LIN ladder values. It does not affect selection, so it can be an explicitly post-hoc diagnostic, but it is not part of the registered arm set and should be recorded as such.
- The ladder’s `dev_macro` fields contain negated holdout loss, not dev nDCG. Current arm filtering limits the damage, but the metadata remains dangerously mislabeled.

## Bottom line on your six beliefs

1. The present document-vector choke-point reading is correct; reported dense/fused documents are transformed once.
2. Raw-space masking closes only the mask channel. Bank membership, source priors, false-negative holdout scoring, and regularization arbitrage remain.
3. R0N is the right comparator. LIN is paired; MLP is not until initialization is seeded.
4. The ladder subprocess is dev-blind, but the overall scorer still permits explicit tuning-arm endpoint access.
5. `assert_fired()` does not prove head movement or external scorer reads.
6. Per-head LR selection is the reasonable reading. The fresh 5,000-step schedule is not a faithful continuation.

The remaining campaign should pause.
