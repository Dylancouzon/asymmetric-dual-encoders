# Codex adversarial review #5 — the seven-day trainer, before launch

Reviewer: `gpt-5.6-sol`, high effort, read-only, run while the screen trained.
Brief: `work/briefs/m9_review5.md`. Date: 2026-08-30. Read-exclusion honoured.

Verdict: **DO NOT LAUNCH this build tonight as it stands** — seven launch blockers, starting with
a fresh run that crashes before its first optimizer step. Its most valuable catches: the loss is
not the plain example mean its docstring claims, the corpus integrity check compares two copies of
a *declaration* rather than the bytes, resume silently accepts a different recipe, and the
mandated numeric kill envelope does not exist. Disposition: `m9/LEDGER.md` §15.

---

## Verdict: DO NOT LAUNCH this build tonight as it stands

The current artifact is not resumable exactly, does not optimize the loss its comment claims, does not enforce the M9.2 kill/extension protocol, and has several paths to silent target or recipe drift.

The immediate state also is not launchable:

- `work/m9long/config.json` is absent.
- The prepared document corpus contains only 40,000 documents, not 6,149,679.
- None of the required `work/enc9/m9long-*` extra-target caches exists.
- `m9src/longrun.py` changed during this review and is currently uncommitted. The new chunked packer is an improvement, but the executable is not frozen.

I remained read-only and honored the exclusions.

## Launch blockers

1. Fresh training currently crashes

After completing the warm start, [longrun.py:305](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:305) reads `warmfit.ARTIFACT.name`. `warmfit.py` defines no `ARTIFACT`. A fresh run dies before its first optimizer step.

This is loud and cheap, but proves the fresh-start path has not been smoked end to end.

2. The loss is not a plain mean over step examples

At [longrun.py:330](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:330), `Stream.take(k)` always returns exactly `k`, so:

```text
part = mean_s(loss) * len(idx)/k = mean_s(loss)
effective loss = Σ_s share_s · mean_s(loss)
```

It is not:

```text
mean over all examples from all sources
```

Token shares already determine source batch counts, then shares are applied again as source-level objective weights. Relative to a true combined-example mean, an individual 95-token document gets roughly six times the weight of a 16-token query.

Either objective could be defensible, but the lock must name the one intended. If the intended objective is the screen’s plain example mean, concatenate source losses as sums and divide by the total example count. Delete the `shares[si]` multiplication.

3. The corpus integrity check checks declarations, not bytes

[longrun.py:271](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:271) compares the hash stored in `corpora.json` with the hash stored in each `meta.json`. It never recomputes `flat.npy`.

Worse, it does not hash:

- `offs.npy`, whose corruption can silently assign one text’s tokens to another target.
- `documents/pool_rows.npy`, which binds document tokens to teacher vectors.
- `m9_screen_rows.npy`.
- Extra target vectors or their row manifests.
- The complete config or selected tokenizer/template.

Running `prepare` again overwrites both copies of the declared hash, so an altered corpus can be silently accepted under an old checkpoint.

4. Resume silently accepts a different recipe

The checkpoint stores `cfg`, but [longrun.py:287](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:287) never compares it with the current config. This can create a hybrid run:

- Adam betas and weight decay come from the checkpoint-loaded optimizer.
- LR, loss shares, gradient clip, eval cadence, and token budget come from the new config.
- Source names/order can change stream assignment.
- The prepared tokenizer/student/prompt is not checked against the config.

Require an exact canonical config hash and corpus/target manifest hash match before loading model or optimizer state.

5. Decay is not resumable

`decay` derives `decay_from` from the current `last.pt` every invocation at [longrun.py:426](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:426). If cooldown stops halfway, rerunning it restarts a fresh cosine at `lr_peak` from that midpoint. Running ordinary `train` after a decay interruption likewise returns immediately to the stable LR.

Persist phase, decay origin, decay length, and endpoint in the checkpoint. Decay should branch from an immutable stable checkpoint into a separate checkpoint lineage.

6. There is no implemented kill rule or “stop on evidence”

The process stops only for wall time, maximum steps, or a manually created `STOP` file. It does not stop for:

- Non-finite target, loss, gradient, parameter, or optimizer state.
- SCREEN-3 collapse.
- Plateau.
- Per-component regression.
- Throughput collapse.
- Target norm failure.

That directly contradicts the mandatory numeric kill envelope in [instructions-m9.md:83](/home/dylan/asymetric-dual-encoders/instructions-m9.md:83). One NaN can poison Adam state and every later checkpoint without causing the run to exit.

The earlier screen explicitly learned that full-target finiteness matters at [screen.py:226](/home/dylan/asymetric-dual-encoders/m9src/screen.py:226); longrun dropped that protection.

7. The run is outside the existing guard

`longrun.py` is absent from the guarded training scope at [guard9.py:41](/home/dylan/asymetric-dual-encoders/m9src/guard9.py:41), and it never calls `begin_run`. It can therefore run from dirty/unpushed code with unregistered config, corpora, or targets.

Add a distinct M9.3 build scope covering code, config, all manifests, row maps, target metadata, model revision, environment, and the prepared corpus hashes.

## Major operational defects

- The revised packer avoids the original ~21 GB token-ID heap, but it still retains all packed chunks before `np.concatenate`, materializes all 6.15M document strings through `row_texts`, and writes corpora directly over existing files. Its “idempotent” docstring is false: [longrun.py:123](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:123) always rebuilds and overwrites. Full-scale preparation still needs a monitored rehearsal and atomic versioned output.

- Extra target caches are optional during initialization but mandatory during `get`. Missing targets therefore fail only after the first source has already performed backward. Validate every target cache, shape, dtype, row ordering, hash, finiteness, and norm before constructing the optimizer.

- The query preparation hardcodes policy B, independently of `cfg["student_query_prefix"]`. If M9.2 selected another prompt, warm start and training inputs disagree.

- `--hours 168` means 168 hours per process invocation, not cumulatively. After a restart it grants another seven days. Token and example counters also reset, so history cannot express the locked cumulative dose.

- `status()` reports slope per 1,000 steps, while M9.2 requires retention change per million non-pad tokens.

- The history append and checkpoint are not a transaction. A crash after history append but before checkpoint produces duplicate step records on replay; a partial final JSON line makes `status()` fail.

- `os.replace` protects the old checkpoint from an ordinary process crash, but there is no file/directory `fsync`, previous-generation fallback, disk-space guard, or single-trainer lock.

- Longrun omits the `torch.cuda.empty_cache()` protection the screen added after a measured 1,990→786 examples/s collapse; see [nano.py:292](/home/dylan/asymetric-dual-encoders/m9src/nano.py:292).

## Dose recommendation

I would not register 20/10/70.

My conservative registration would be:

```text
5% real queries / 5% short document spans / 90% full documents by non-pad token
```

At 16.24B tokens, that is approximately:

- Real queries: 109 epochs.
- Spans: 21 epochs.
- Documents: 25 epochs.

With a true combined-example mean, the approximate objective contribution becomes 23% queries, 9% spans, and 69% documents because queries are much shorter. That preserves meaningful query supervision without presenting each real query 438 times.

Pool the three real-query sources into one logical batch, sampling them proportional to their corpus token totals. Current per-source forwards would otherwise produce tiny batches—sometimes one example—for low-share query sources.

If simplicity must win, cut the spans and use 5/95 query/document. The spans are document text, not queries, and are less valuable than getting the trainer correct.

SCREEN-3 alone cannot diagnose generalization outside its three families. At minimum register and log:

- The macro and all three existing component means.
- Per-source training loss.
- Fixed training-only held-out vector loss for each source.
- Cumulative tokens and source presentations.
- Major page faults, disk latency, and inclusive throughput.

A reasonable automatic rollback rule is two consecutive SCREEN-3 evaluations more than 0.0056 below the best checkpoint. A plateau rule should use cumulative tokens, not steps, and require that no component is materially degrading.

## WSD schedule

Warmup→stable→decay is defensible in principle. This implementation is not.

The anchor provides no evidence for millions of updates at constant `1e-4`: it warmed to `1e-4` and then immediately spent nearly all training decaying toward `1e-5`. Holding the peak for seven unattended days can oscillate around the optimum, fit repeated queries, forget pretrained structure, or drift after reaching the best checkpoint. A cosine horizon at least guarantees increasingly small late updates.

If WSD remains:

- Put a hard cumulative-token cap on the stable phase.
- Add automatic plateau/regression stopping.
- Preserve immutable stable checkpoints.
- Rehearse the stable LR for several evaluation intervals; absent that evidence, I would not hold `1e-4`.
- Make cooldown independently resumable.

The default 4,000-step cooldown is only 7.84M tokens—about five minutes at the measured throughput. The only empirical annealing scale available is the anchor’s 59.5M tokens, approximately 30,349 current-size steps and 37 minutes. I would register at least that 59.5M-token cooldown. Anything shorter is unsupported by the project’s own evidence.

## Page-cache behavior

The 12.6 GB pool target map is not automatically a disaster. Advanced indexing copies only the selected rows, and Linux’s page cache is reclaimable. The first document sweep will incur random page faults; later sweeps may be largely cached.

The machine has 25 GiB RAM, however, and the active mappings include roughly:

- 12.6 GB document targets.
- ~2.3 GB document token IDs.
- Extra query/span targets and token files.
- Evaluation vectors and normal process memory.

That is tight enough for eviction and repeated random faults. At 70% document share, the run logically consumes about 19.5 target epochs, or roughly 246 GB of row reads before page amplification. This is primarily a throughput/dose risk, not a correctness risk. Because the run is wall-clock bounded, any cache thrashing silently reduces delivered dose. Record actual cumulative tokens and monitor major faults/I/O; do not claim 16.24B from throughput multiplication alone.

## M9.2 items that must exist before launch

In priority order:

1. Freeze and push the exact reviewed trainer; add it and its config/manifests to the guard.
2. Fix the immediate `warmfit.ARTIFACT` crash.
3. Lock the exact loss equation and independently define token sampling versus loss weighting.
4. Create and validate the full corpus and every target cache; atomically publish a complete manifest.
5. Implement config/corpus/target hash checks and exact stable/decay resume state.
6. Add non-finite checks, kill rules, cumulative token accounting, and automatic evidence-based stopping.
7. Smoke the full path, then prove split-run equivalence: uninterrupted `N` steps versus `N1 + resume + N2`, comparing model, optimizer, streams, phase, LR, and counters.
8. Repeat that equivalence test across an interrupted cooldown.
9. Benchmark one eval-return-to-training cycle and verify throughput does not collapse.
10. Lock checkpoint cadence, retention, disk budget, and recovery from a deliberately damaged `last.pt`.

If these cannot fit in eight hours, cut spans, reduce evaluation/checkpoint complexity, and delay the launch. Do not cut resume validation or target integrity.

## Capacity probe

Skip it.

It costs 60–70 minutes, cannot affect M9 because of the 35M cap, and a positive result only informs M10. It also is not runnable currently: the registry names `bge-base-en-v1.5`, but `nano.STUDENTS` contains no such entry.

Spend that hour on full-scale preparation, one checkpoint/resume equivalence test, and a post-eval throughput test. Those can save the seven-day run; the capacity probe cannot.
