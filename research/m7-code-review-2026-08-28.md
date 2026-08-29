# Adversarial code review #2 — m7src, 2026-08-27/28 code

Scope: multieval, dev_audit, compare_full, lever5_shrinkage, longspan_probe, table (pool_mode /
occurrence_weights / encode_pooled / forward(extra_psw) / fingerprint), train (_fwd, init_preproc),
sweep (chain/chains/smoke_chain), program (ablation_recipe / phase4_*), adopt_pool_mode,
freeze_heldout, heldout.verify_pinned, boot (signflip_dep / paired_dep / strata /
upper_bound_one_sided / ci95_raw). Read-only; CPU-only numeric verification was run where the
math could be checked without the GPU (results inline below).

---

## BLOCKER

### B1. `sweep.chain` / `chains` reuse a B artifact by run id with NO config check
`m7src/sweep.py:125` (`chain`, `skip_b_if_exists`) and `m7src/sweep.py:150-155` (`chains`,
`share_b`).

What goes wrong: `chain` reuses `work/runs/{name}-b.npz` whenever the file exists. Nothing
compares the artifact's stored cfg (`work/runs/{bid}.json`) against
`replace(base, **b_over)` for THIS invocation. The A leg then trains from — and the arm is
reported as — a B phase that may have been trained under different overrides.

Concrete triggering states, all realistic for this repo:
- `ablation_recipe()` derives the B recipe from **whichever candidate survives the dependence
  recompute**. If `dev_audit` is ever re-run and the survivor changes (the code explicitly
  anticipates this: `SURVIVOR_STEPS_A` maps four candidates, with B phases of 8,000 vs 16,000
  steps and different pseudo mixes), re-running `run_ablations.sh` silently reuses
  `p4-input-emb-b` etc. trained under the OLD survivor's B recipe, while printing the new one.
- Any edit to an arm's `"b"` dict (bug fix, changed variable) followed by a driver re-run — the
  driver is documented as idempotent and re-runs are the normal workflow.
- `chains`' `share_b` reuses `{name}-{share}-b` from disk even when the sharing arm's assumed
  B recipe differs from what that artifact was actually trained with (e.g. after the same edits).

This is a silent wrong-arm: the ablation table then attributes to one variable a delta that
partly comes from a stale B phase. It is exactly the class of error the ablations exist to
exclude.

Smallest fix: on reuse, load `work/runs/{bid}.json`, build
`want = asdict(replace(base, run_id=bid, **b_over))`, and `raise SystemExit` on any differing
key (print the diff). Same check in the `share_b` branch. ~8 lines. (A missing `.json` next to
an existing `.npz` should also refuse.)

---

## MAJOR

### M1. `longspan_probe` slicing silently misaligns if any bucket under-fills
`m7src/longspan_probe.py:44-53` (`spans`) and `:96-101` (fixed-offset slicing).

`spans()` returns FEWER than `count` spans when the corpus runs out of documents of `n_words`
(no assertion). The Q / `bi` slicing at line 97-100 uses fixed offsets `bi_idx * 2 * per_bucket`.
If any non-final bucket is short, every later bucket's slices are shifted: teacher rows of one
bucket get paired with table rows of another. Cosine and overlap@10 are computed from the SAME
wrong slices, so nothing crashes until the array end — and a partial shift pairs teacher(span_i)
with table(span_j), depressing agreement most in the long buckets. That fabricates precisely the
"falling curve with length" the probe exists to detect, i.e. it could buy a long-span
distillation chain on an artifact of an off-by-N.

Measured today: cqadup-physics has 3,977 docs ≥ 256 words vs 300 needed, so the current
constants do NOT trigger it — this is latent, one changed `SOURCE`/`PER_BUCKET` away, and the
script has never executed.

Smallest fix: after building `sp`, `assert all(len(sp[n]) == per_bucket for n in BUCKETS)` —
or derive offsets from actual block lengths (`np.cumsum`). One line.

### M2. Lever-6 arm (a) cannot run: `get_init("run:...")` fingerprint check includes `pool_mode`
`m7src/init_table.py:127-130` vs the pre-registration in `m7/LEDGER.md` (lever #6).

The A-from-B-checkpoint init asserts
`meta["preproc_fingerprint"] == runtime_pre.fingerprint()`. The candidate's B checkpoint was
trained and saved under `pool_mode="mean"` (fingerprint without the field); lever-6 arm (a) sets
`cfg.pool_mode="sqrt"`, so `runtime_pre.fingerprint()` includes `pool_mode` and can never match.
`smoke_chain` will crash — a crash, not a wrong number — but the danger is the workaround: a
session under time pressure either trains arm (a) at `mean` (not the lever) or loosens the
assertion in a way that also silences the real prefix/truncation mismatches it exists to catch.

Smallest fix: compare fingerprints of the preproc with `pool_mode` normalized out
(`replace(rt, pool_mode="mean")` vs the stored dict minus `pool_mode`), and separately LOG a
pool-mode transition (`B trained under mean, A trains under sqrt`) rather than refusing it —
that transition is the entire point of lever 6(a). Keep the strict check for prefix /
add_special_tokens / max_length.

### M3. `gate.py` G1 decides on the ROUNDED CI endpoint
`m7src/gate.py:106`: `"pass": bool(g1["ci95"][0] > 0)` — G3 was fixed to `ci95_raw`
(`gate.py:136`) after Codex review #3b MAJOR 2, but G1 still reads the display value. A true
lower endpoint in (0, 5e-5] rounds to 0.0000 and flips G1 to FAIL. Wrong verdict, one-token fix:
`g1["ci95_raw"][0]`.

### M4. `compare_full` smoke output is indistinguishable from a real run
`m7src/compare_full.py:64-65, 85, 95`. With `--smoke`, every corpus is truncated to 200K docs
(nDCG meaningless by multieval's own docstring), yet the result is written to
`results/m7_compare_full_{tag}.json` — same filename as the real run, no smoke marker, no
`max_docs` field, and `_status` text that reads like a normal exploratory result. A smoke run
left behind (e.g. the real run crashes afterwards) is a committed-looking wrong number.
`dev_audit`, `lever5` and `longspan_probe` all suffix `_smoke`; this file is the one that
doesn't. Also `m7_devperquery_{tag}.json.gz` collides with `dev_audit`'s dumps if anyone passes
tag `full` or `smoke`.

Smallest fix: `tag = f"{tag}_smoke" if smoke else tag`, plus `"smoke": smoke, "max_docs": ...`
in the JSON, and refuse tags `{"full","smoke"}`.

---

## MINOR

### m1. `dev_audit` cannot be re-run post-adoption (preproc mismatch across CHAIN)
`m7src/dev_audit.py:130-133`: all four CHAIN artifacts must carry identical `Preproc`.
`adopt_pool_mode` has since written `pool_mode="sqrt"` into the candidate's meta only, so a
re-run now raises `AssertionError` at load. Crash, not a wrong number — but the audit is the
reproduction path for the committed lever numbers. If reproducibility matters, compare preprocs
with `pool_mode` stripped and serve every `|table` maker explicitly at `mean` (which is what the
historical numbers were).

### m2. `upper_bound_one_sided` (G4) and `paired` (G1) never align strictly
`m7src/boot.py:141-142` calls `paired_dep(..., strict=False default)`; `boot.py:151` uses
`_align(a, b)` permissive; `paired()` has no strict parameter at all. LEDGER says
"`_align(strict=True)` on every confirmatory path". G3 is protected because its `signflip`
runs strict and would abort the process; G1 and G4 are not — a qid/dataset mismatch between the
fp16 and int8 dicts (or candidate vs refs) would silently score the intersection. In practice
both sides come from one pass, so sets match today; this is a one-argument hardening
(`strict=True` through `upper_bound_one_sided`, add `strict` to `paired`, pass it in gate G1).

### m3. `lever5` tau=0 baseline is not byte-identical to the released artifact
`m7src/lever5_shrinkage.py:79-92`. The fp16 baseline keeps fp32 folded rows (the release stores
them fp16 at rest; tolerance 1e-3 asserted), and the int8 arm re-quantizes
`quantize_int8(folded_fp32)` instead of reading the artifact's stored `rows_int8` — the exact
distinction `dev_audit.load_release` documents ("int8 comes from the artifact's OWN stored
codes"). Differences: the softplus round-trip in `w` (≈1e-7) and fp16-at-rest rounding, so
codes can differ by ±1 in rare rows. All arms are built the same way, so the tau comparisons are
internally consistent; only the label "the released artifact" is slightly off. Cheap upgrade:
assert `(quantize_int8(folded(A,w))[0] == za["... release ..."]["rows_int8"]).all()` for tau=0,
or build tau=0 from `load_table(rel, variant=...)` directly.

### m4. `lever5` folds the blend with A's token weights
`row = a·A + (1−a)·B` is computed on UNFOLDED rows and then folded with the candidate's `w_A`
(`lever5_shrinkage.py:79-83`). Where the A phase moved a row's weight, `(1−a)·B_i` is served as
`w_A[i]·B_i`, which is not B's served row (`w_B[i]·B_i`). For `u_i = 0` rows the two coincide
(zero grad ⇒ Adam never moves `w_raw[i]` either — verified against `train.py`'s optimizer
setup), so the limit behavior matches the rationale; for small `u_i > 0` it is a definitional
choice the pre-registration's wording doesn't pin down. Record the choice in the artifact's
`_protocol` string before the numbers exist. `u` itself is correct: the candidate is an A-only
run, `updates` starts at zeros per run and counts only rows whose token ids appeared in a batch
— i.e. exactly the A-phase update counts LEDGER describes.

### m5. `capacity_probe.py:55` still calls `model(f, o, l)` directly
Mean-only; ignores any pool_mode. Gate-ineligible diagnostic, so no number is wrong today, but
it is now the only query-vector site outside `_fwd`/`encode`/`encode_pooled`. Route it through
`model.encode` or leave a comment saying it is deliberately mean-only.

### m6. `boot.paired_dep(share=False)` generates and discards the shared draw
`boot.py:291-295`: `draw` is sampled every stratum-chunk and unused when `share=False`. Harmless
(different RNG streams between the three bootstrap variants are expected; they are compared as
distributions, not per replicate) — but wasteful and worth a comment so nobody "fixes" it into a
behavior change.

---

## Sections with NO findings (checked hard, explicitly clean)

- **`multieval.eval_makers` block/offset arithmetic**: blocks are appended (comp-outer,
  tag-inner) with cumulative `start`; Q is filled per block from the same records; `span` and
  `rank_compare` slice `bi/bs` with the same `(start, n)`. Every per-query score lands on the
  right (tag, comp, qid): the maker output length is asserted per block, duplicate qids are
  refused, and the `set(nd) != set(q_ids)` check catches pytrec_eval's silent drop. The
  shared-corpus guard (`same_corpus`) is identity-first with a content fallback that fails
  closed (raises, never silently splits). `max_docs` truncates after Q is built and before
  scoring, consistently for ids and vectors.
- **`compare_full` alignment**: the maker closures bind `(model, pre)` by immediate invocation
  (no late-binding bug); comparisons pair `{key}|{q}` against `{base}|{q}` from the same pass.
  Override precedence in `load()` (meta first, explicit `:mode` second) is correct.
- **`longspan_probe` bucket slicing** given full buckets: `bi` and `Q` are sliced with the same
  offsets in the same [teacher, table] × bucket order as `blocks` was built; the teacher spans'
  encode cache is keyed on `sha_texts(texts)` (checked in `teacher.py`), so stale-span reuse
  cannot happen; smoke slices pool and doc_ids consistently.
- **`table.Preproc.fingerprint` back-compat is airtight** — verified by execution:
  old-style dict without the key, `Preproc()`, and `Preproc(pool_mode="mean")` all hash to
  `4f7978fa7f69b559`; a stored `preproc` dict with no `pool_mode` key round-trips through
  `Preproc(**d)` to the same value.
- **`forward(extra_psw)` math** — verified numerically on CPU against a hand-written reference
  for binary/cap2/sqrt/mean, learned and flat weights: numerator, denominator (index_add over
  `_bag_index`), normalization and the fallback path all correct. Weight folding is exact under
  non-mean pooling too (max dev < 1e-5 in fp32), so `save_release` + `occurrence_weights`
  compose correctly.
- **`train.py` `_fwd` coverage**: every training-time query vector goes through `_fwd`
  (step_a, step_b including the pseudo part, collapse_stats); in-training dev goes through
  `dev_eval.eval_table` → `QueryTable.encode`, which routes by `pre.pool_mode` — the same
  `pre = replace(PRE[cfg.preproc], pool_mode=cfg.pool_mode)` that is saved into the artifact's
  meta. No direct `model(f, o, l)` calls remain in train.py.
- **`program.phase4_negatives` precedence**: `a` (the surviving A recipe, minus run_id/init)
  fully overrides `base` in `replace(base, **over)`, `init` is the candidate's own `"run:<bid>"`
  string, and the four arms differ only in `hard_neg_k`/`hard_neg_source`. `mixed32` composes
  16 teacher + 16 bm25 columns as intended.
- **`ablation_recipe`** correctly follows the artifact (cfg JSONs) rather than the constants and
  asserts drift; candidates trained before `Cfg.pool_mode` existed replay at `mean` via the
  default, matching LEDGER's documented inconsistency.
- **`freeze_heldout` / `heldout.verify_pinned`**: `assert_structure` proves the nesting
  (subset, per-qid text/qrels/n_tokens identity, exact ≥64-token membership); the memo ("cheap"
  vs "full") cannot let a cheap early check satisfy a later pool-bytes request.
- **`adopt_pool_mode`**: refuses unless the committed lever-4 artifact adopted that mode on that
  run id; edits the training meta AND forces a release rebuild, so `ensure_release` cannot
  resurrect the old rule.
- **`boot` dependence machinery** — verified by simulation: strata partition covers every
  (component, position) exactly once; per-dataset replicate means are correct for mixed-size
  strata (sum of per-stratum resampled sums / n_ds); under full duplication the
  dependence-preserving CI is 1.392x the ordinary one (theory √2) and the sign-flip null
  rejection rate at α=0.05 is 0.055 for `signflip_dep` vs 0.12 for the dependence-blind
  `signflip` — the fix does what LEDGER claims. `upper_bound_one_sided(dep=True)` correctly
  equals the 97.5th percentile of the `paired_dep` distribution and refuses other levels.
  `ci95_raw`/`delta_raw` are present and are what `dev_audit`/`lever5` decide on.
- **`dev_audit` B1/L4 logic**: chain pairs, verdict short-circuit, surviving selection, the
  two-precision Holm-family construction, and `adjudicated_on=surviving` all match the
  pre-registered rules; the `equiv` per-component dict's `"n"` key collision between the nDCG
  row and `rank_compare` is benign (same value).

## Correct but under-tested — cheap assertions worth adding

1. `chain`/`chains`: the B-reuse cfg-equality check (the B1 fix) IS the missing test.
2. `multieval`: a conformance case registering the SAME maker under two tags across a two-comp
   group (one heldout) and asserting byte-identical per-query dicts — a permanent canary for the
   offset arithmetic. Optionally fill `Q` with NaN at allocation and assert `isfinite` before
   `topk_arrays` (proves every block was written).
3. `test_conformance`: pin `Preproc().fingerprint() == "4f7978fa7f69b559"` as a literal — the
   current check (`Preproc()` vs `Preproc(pool_mode="mean")`) is near-tautological and would
   pass even if both drifted from every fingerprint already on disk.
4. `test_conformance`: `forward(extra_psw=occurrence_weights(...))` vs `encode_pooled`
   equivalence — the training-forward path lever 6 depends on has no test (I verified it
   numerically here, but the check should live in the suite).
5. `longspan_probe`: `assert len(sp[n]) == per_bucket` (the M1 fix).
6. `lever5`: assert tau=0 int8 codes equal the release's stored `rows_int8` (m3).
7. `boot.unit_key` trusts the `heldout-` name prefix for unit sharing; `assert_structure` proves
   the actual nesting at pin time but nothing ties the two — a one-line check in `strata` that
   any multi-component stratum's components are exactly the pinned nested pair would close it.
8. `gate` G1: also run `signflip(..., strict=True)` (as G3 does) so a refs/candidate qid
   mismatch aborts instead of intersecting.
