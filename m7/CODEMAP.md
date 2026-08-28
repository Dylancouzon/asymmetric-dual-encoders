# M7 code map

Read `STATUS.md` first. This file exists so a future session can resume without reading 40
modules. Details live in the modules' own docstrings — every one states why it exists. Nothing
here is restated from `LEDGER.md` (protocol) or `EXPLORED.md` (dead ends).

## Layout

- `m7src/` — all M7 code. `bench/` and `scripts/` are the reused M1–M6 harness; do not fork them.
- `work/` — gitignored heavy data: encode caches, doc pool, tables, runs, dev caches.
- `logs/` — gitignored driver logs. `results/m7_*.json` — every committed M7 artifact.

## Drivers (in order; each is idempotent, takes an optional first-step number)

| driver | steps |
|---|---|
| `run_stage0.sh` | dev encodes → asset freeze → decontamination → field table → doc pool → dev reference rows |
| `run_stage0c.sh` | held-out dev slices (need the pool) → TRAIN↔held-out decontam → reference rows |
| `run_stage0b.sh` | ridge probe → capacity probe → objective grid → go/no-go gate |
| `run_ablations.sh` | chain smoke → attribution controls → 7 mandatory chains → negatives → exploratory |

**Strictly sequential, one GPU/memory job at a time**, enforced by the drivers and by `flock`.

## Modules

**Foundation** — `_paths.py` (paths, `sys.path`, the six datasets) · `encoders.py` **the registry:
the one place that knows how to run each candidate tower**; select with `M7_ENCODER`, add a `Spec`
rather than special-casing at a call site · `teacher.py` the frozen teacher; `encode_cached` is
shard-resumable and returns a **memmap**, its key covers model/revision/dtype/tokenizer/pooling/
prefix/length/corpus · `test_encoders.py` replays every cache key and requires the directory name
back unchanged · `hashing.py` streaming `sha` equivalents · `evalkit.py` chunked GPU brute force,
tiled on **both** axes (`topk_arrays` / `run_from_arrays` split out so many query blocks can share
one corpus pass) · `multieval.py` scores N query-side variants with ONE pass per **corpus** (the
two held-out components share the pool), plus `rank_compare` for path equivalence.

**The artifact** — `table.py`: `Preproc` (the one frozen query rule, now including `pool_mode`),
`QueryTable`, int8 per-row absmax, `apply_unseen_policy`, `occurrence_weights`/`encode_pooled`
(count saturation), `save_release` folds learned weights into rows (exact) — the shape G4 gates and
HF ships; `save_table` stays unfolded for training resume. `adopt_pool_mode.py` is the only
sanctioned way to change the served pooling rule. `test_conformance.py` 42 checks — run after any
`table.py` edit. `costs.py` the three cost numbers.

**Data** — `trainmix.py` (TRAIN mix, owns `heldout()`) · `mix.py` (memoized loader) · `pool.py`
(one fp16 memmap; `PoolIndex` per-store and lazy) · `pseudoq.py` (vocabulary-coverage
pseudo-queries; spans are first-sentence, ≤32 words) · `decontam*.py` (fingerprint decontamination;
`decontam.run()` has its own inline R1 matcher besides `query_grams` — change BOTH) ·
`decontam_pool.py` writes `banned_pool_rows.npy`, which `train.py` REFUSES to run without ·
`devsuite.py`/`heldout.py` the pinned components; `freeze_heldout.py` pins the held-out pair and
the pool's bytes, `heldout.verify_pinned()` enforces it.

**Training and selection** — `train.py` objectives A/B/C; `Cfg` is the whole experiment surface ·
`sweep.py` runs configs and appends every run to `RESULTS.md` including failures; `chain`/`chains`
run an arm as TWO runs (B, then a fresh A from that checkpoint) and `smoke_chain` proves that path
· `program.py` the phased plan; `ablation_recipe()` derives the chain recipe from the surviving
artifact's own config · `dev_eval.py` dev macro + pinned reference rows · `stage0_ridge.py`
closed-form flat table · `capacity_probe.py` **diagnostic, gate-ineligible** · `boot.py`
`signflip` is THE p-value, `signflip_dep`/`paired_dep` handle the nested components, `both_ways`
reports all three bootstraps · `gate.py` the eligibility audit · `fusion.py` one family, one
builder (`test_fusion_paths.py` guards the re-fork); `select_fusion.py` fits against the RELEASE
artifact.

**Diagnostics** (each answers one question the plan was assuming) — `calibrate.py` MTEB→six
(residual sd 0.0102; larger than the gap between top teacher candidates) · `absorb_check.py` which
query-side transforms are absorbable · `teacher_probe.py` symmetric ceiling (**refuted as a
selection criterion**) · `scripts/teacher_learnability.py` + `scripts/learnability_report.py` the
adopted criterion · `diag_scores.py` contrastive score geometry · `ridge_full_eval.py`,
`ridge_vs_trained.py` · `validate_encoder.py` **mandatory for any new `Spec`** (it exists because
stella's Spec initially omitted its published Dense head).

**Audits and levers** — `dev_audit.py` one full-suite pass producing the dependence-preserving
lever recompute, matrix-vs-`QueryTable` equivalence, the per-query dump and the lever-4 arms ·
`bigram_residual.py` (#1) · `doc2query_probe.py` (#3) · `lever5_shrinkage.py` (#5) ·
`lever4_readjudicate.py` re-runs the pooling family on a NAMED artifact (dev_audit derives it from
a hard-coded chain that no longer ends at the candidate) · `longspan_probe.py` is the length
diagnostic with one run id and lever #7's primary bar with two.

**Decision executors** — a pre-registered rule a session can re-read in its own favour is not a
pre-registration, so the rules that pick things now run as code: `negatives_decide.py` (promotion,
bar, Holm, three tie-break levels) · `simplify_decide.py` (non-inferiority at −0.0040). Both read
the committed comparison artifact and write their own.

**Honesty instruments** — each answers a question a reviewer will ask, with a number instead of a
paragraph. `dev_reuse.py` counts adaptive dev reuse (58 arms / 322 in-training evals / 90
eval-only variants as of 2026-08-28 -- quote the JSON, not this line, which goes stale) · `retention.py` retention on three nested component groups with BM25 on the
same rows, because all-six and out-of-domain differ by 0.93 vs 0.76 · `cold_rows.py` what ships in
rows training never touched (`apply_unseen_policy` is defined and never called) · `absorb_check.py`
which transforms are absorbable, now including the doc-side map in both the renormalized and
un-renormalized cases.

**Final and demo** — `final_run.py` refuses unless the tree is clean, HEAD equals the freeze
commit, that commit is pushed, and no prior final-run entry exists · `ann_sweep.py` real HNSW ·
`edge_demo.py` the two-collection architecture · `freeze_m7_assets.py` pins dev + untouched-final.

## Reusing this repo as a harness

The eval protocol, partitions, decontamination, bootstrap/Holm statistics, freeze and final-run
machinery are the reusable part and have been through three adversarial reviews. Different
**model**: add a `Spec` and set `M7_ENCODER` — do not edit `teacher.py`. Different **query-side
technique**: the surface is `table.py` plus `train.py`'s `Cfg`.

Adding an encoder, in order: write the `Spec` (pooling, prompt, `post_dense`, `config_kwargs`,
revision pinned) → `test_encoders.py` → **`validate_encoder.py`** → only then a probe or an encode.
Skipping the validator is how a comparison silently runs the wrong model. Two things that must
move with the encoder: anything assuming **dim 768**, and `table.py`'s `CLS_ID`. Pinning weights
does not pin `trust_remote_code` code, which comes from a separate repo at HEAD.

## Log size policy (context is the scarce resource)

`for f in m7/*.md; do echo $f $(( $(wc -c < $f) / 4 )); done`

| file | budget | rule |
|---|---|---|
| `STATUS.md` | ~1.2K | one screen. Rewritten, never appended. The only file always read. |
| `RECIPE.md` | ~1.5K | the released recipe end to end, for a third party. Rewritten when the recipe changes, which after the freeze is never. |
| `CODEMAP.md` | ~2.5K | grows only when a module is added or a pitfall is earned. |
| `RESULTS.md` | ~1.5K | one row per run; detail belongs in the run JSON. |
| `EXPLORED.md` | ~1K | one row per closed avenue. |
| `LEDGER.md` | ~4K, hard | at 4K, compact again: keep every protocol fact, cut settled justification to one line. |

**The rule that keeps it small: never restate a number that a `results/m7_*.json` already holds.**
Compaction is safe — git preserves every prior version, and each file says when it was compacted.

## Pitfalls that already cost time

1. **Never `git add -A` without checking `.gitignore`.** One did, committed the multi-GB encode
   cache, and `git push` hung.
2. **Subagents must be told to write only to the scratchpad.** One dumped files into `m7src/`.
3. **Run one memory-heavy job at a time.** Three at once took the WSL distro down.
4. **`pgrep -f` matches the shell that wrote the pattern.** Anchor to the interpreter
   (`pgrep -f "^[^ ]*python[0-9.]* -u scripts/foo.py"`). A driver that `exec`s python has no
   script name in any cmdline. And a compound command that BOTH kills by pattern AND relaunches
   kills its own shell first (exit 144) — kill and relaunch in separate commands. Cost time four
   times, most recently 2026-08-28.
5. **`np.isin` re-sorts its second argument every call.** Pre-sorted array + `np.searchsorted`.
6. **Never call a JSON loader inside a hot loop** (`mix.load_source` re-parsed 16 MB per pair).
   Same class: `dev_eval.doc_vecs` re-parses its corpus cache on EVERY call — HotpotQA's is 5.23M
   documents and peaks ~14 GB, so memoize the query texts if that is all you need.
7. **Loop order decides the cost of anything touching the pool.** Queries outside, the 9.5 GB pool
   inside = 1.6 TB of reads and 3.6 hours instead of 165 seconds.
8. **Batch budgets must come from the LONGEST sequence in the batch.** Do not fork the harness's
   batching, loader or pooling: call it.
9. **A config knob not in the encode cache key is a silent stale-vector bug.**
10. **Any width assumption must come from the registry.** `pool.py` had `DIM = 768` and would have
    rebuilt — overwriting a 9.5 GB pool — from a read-only call site.
11. **Nothing may materialize a whole corpus.** Assume 18 GB peak RAM.
12. **Smoke the path with NO execution history, not a convenient one.** A smoke over two small
    text components missed the shared-pool path and cost a 35-minute run; `multieval` now takes
    `max_docs` so a 6.17M-row corpus can be smoked cheaply.
13. **Object identity is not corpus identity.** Two callers can hold equal-but-distinct doc-id
    lists for the same corpus; compare content (and memoize the shared one) instead.
14. **One process per training arm.** A driver that runs a night of arms in one process
    accumulates every memoized cache here (`mix.load_source`, `heldout._DOC_IDS`,
    `dev_eval._HELD_CACHE`, encode memmaps), so each arm starts from more memory than the last and
    the third one thrashes. `run_arm.py` runs exactly one leg and exits.
15. **An arm's base recipe must come from the artifact it is varying against, not from a
    snapshot.** `program.ablation_recipe()` read the dev-audit survivor, which stopped being the
    candidate once the negatives arm and the simplification moved past it. `M7_RECIPE_FROM=<run_id>`
    overrides it. An arm that copies overrides out of this file measures its knob PLUS every
    change made since the copy.
16. **The update counter is NOT restored from a `run:` init.** `updates < 1` on an A-only arm
    means "the A phase missed this row", not "training never touched it" — the never-trained set
    is the intersection with the B checkpoint's. Reading it the other way overstated the untouched
    rows by more than 2x (3,750 vs 1,743) in a pre-registration.
17. **Take the rate check IN the slow region, not on the first batches.** Lever #7's teacher
    encode ran at 1,511 → 1,368 → 802 → **55 texts/s**: the objective-B text set is querytext
    first (short), then the pseudo pool ordered BY STORE, and `esci-prod` supplies 68% of the long
    spans. Extrapolating the first shards gave 36 minutes for a ~2-hour job. A pool with
    non-uniform composition has no single rate — find the slowest block and estimate from there.
18. **`rchar` is 0 while a memmap gather runs** — mmap access is page faults, not read syscalls.
    Do not read "zero I/O + 100% of one core" as a hang; check RSS and `free` instead. And never
    materialize a whole gather on the host when the destination is a GPU tensor: fill it in chunks.
