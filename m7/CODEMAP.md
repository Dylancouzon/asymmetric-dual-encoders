# M7 code map

Read `STATUS.md` first. This file exists so a future session can resume without reading 20
modules. Details live in the modules' own docstrings — every one states why it exists, not just
what it does. Nothing here is restated from `LEDGER.md` (protocol) or `EXPLORED.md` (dead ends).

## Layout

- `m7src/` — all M7 code. `bench/` and `scripts/` are the reused M1–M6 harness; do not fork them.
- `work/` — gitignored heavy data: encode caches, doc pool, tables, runs, dev caches.
- `logs/` — gitignored driver logs (`stage0.log`, `stage0b.log`, `stage0c.log`).
- `results/m7_*.json` — every committed M7 artifact. Small, machine-readable, safe to read.

## Drivers (run these, in order; each is idempotent and takes an optional first-step number)

| driver | steps |
|---|---|
| `run_stage0.sh` | dev encodes → asset freeze → decontamination → query-text decontam → field table → doc pool → dev reference rows |
| `run_stage0c.sh` | held-out dev slices (need the pool) → TRAIN↔held-out decontam → reference rows again |
| `run_stage0b.sh` | ridge probe → capacity probe → objective grid → go/no-go gate |

**Strictly sequential, one GPU/memory job at a time.** Enforced by the drivers, not by intention
— see the OOM incident in `LEDGER.md`.

## Modules

**Foundation**
- `_paths.py` — repo/work paths, puts `bench/` on `sys.path`, pins the six datasets.
- `encoders.py` — **the encoder registry: the one place that knows how to run each candidate
  tower** (repo, revision, dim, pooling, query/doc prompt, remote-code, tokenizer identity, vocab).
  Select with `M7_ENCODER`; default is the M7 teacher. Add a `Spec` rather than special-casing a
  model at a call site.
- `teacher.py` — the frozen teacher, whichever `encoders.active()` returns. `encode_cached()` is
  the workhorse: shard-resumable, cache key covers (model, revision, dtype, tokenizer identity,
  **pooling**, prefix, max length, corpus hash), returns a **memmap** over a stitched
  `combined.f16`.
- `test_encoders.py` — replays every `work/enc/*/meta.json` through the current `cache_key()` and
  requires the directory name back unchanged. Run it after touching `encoders.py` or `teacher.py`;
  a key drift orphans ~22 GB of encodes, and a key *collision* between two encoders that produce
  different vectors is worse.
- `hashing.py` — streaming equivalents of the M4 `sha(json.dumps(...))` convention, byte-identical,
  for corpora too large to serialize whole.
- `evalkit.py` — chunked GPU brute force + per-query nDCG. Tiles the score matrix on **both** axes
  under a byte budget.

**The artifact**
- `table.py` — the released lookup table: `Preproc` (the one frozen query rule), `QueryTable`,
  int8 symmetric per-row absmax, save/load, `apply_unseen_policy`.
- `test_conformance.py` — the mandated pre-training gate, 24 checks. Run it after touching `table.py`.
- `costs.py` — the three cost numbers (query asset / doc index / hydration).

**Data**
- `trainmix.py` — builds the TRAIN mix from approved sources only. Owns the `heldout()` rule.
- `mix.py` — loader over the built mix. `load_source` is memoized on purpose (see `LEDGER.md`).
- `pool.py` — the frozen doc-vector pool, one fp16 memmap; `PoolIndex` is per-store and lazy.
- `pseudoq.py` — pseudo-queries for **vocabulary**-coverage distillation (not domain coverage).
- `decontam.py` / `decontam_querytext.py` / `decontam_heldout.py` — fingerprint decontamination.
  The index is built over the TRAIN side and protected corpora are streamed against it.
- `devsuite.py` / `heldout.py` — the pinned dev components. `dev_eval.dev_components()` is the
  authoritative list.

**Training and selection**
- `train.py` — objectives A (InfoNCE) / B (distillation) / C (B then A). `Cfg` is the whole
  experiment surface.
- `sweep.py` — runs configs and appends every run to `RESULTS.md`, including OOM and failures.
- `program.py` — the phased plan (objective → negatives → hyperparams → mandatory ablations →
  coverage → FEVER in/out).
- `dev_eval.py` — dev macro + the pinned reference rows (cached in `work/devres/refs.json`).
- `stage0_ridge.py` — closed-form MSE-optimal flat table; the structural upper bound.
- `capacity_probe.py` — deliberate overfit on dev. **Diagnostic, gate-ineligible.**
- `boot.py` — paired bootstrap, one-sided tests, Holm. `gate.py` — the go/no-go gate.
- `fusion.py` — one fusion family, selected on dev, frozen before any test access.
  `select_fusion.py` runs that selection; `fusion_report.py` decomposes the gain per component,
  because a fusion macro can be one component wide exactly as the G3 win was.

**Diagnostics (cheap, and each answers one question the plan was assuming)**
- `calibrate.py` — fits MTEB v1 Retrieval → our six-set on the nine models we measured ourselves.
  Pure arithmetic over committed numbers; reads no eval data. Use it before quoting any teacher
  projection, and note its residual sd (0.0102) is larger than the gap between the top candidates.
- `absorb_check.py` — which query-side transforms are absorbable into the table (centering,
  whitening, top-PC removal, per-token weights: all of them) and which add capacity (n-grams,
  multiplicity-dependent pooling). Settles a lever's *theoretical* case in seconds.
- `teacher_probe.py` — ranks candidate teachers by measured ceiling on the two CQADupStack dev
  components, since the projection cannot separate them. Reads the shared registry.
- `diag_scores.py` — the contrastive score geometry: positive/negative distributions, softmax mass
  per temperature, and what the `fn_margin` filter actually removes.
- `ridge_full_eval.py` — the Stage-0.1 closed-form bound on the full pinned suite, not the proxy.

**Final and demo**
- `final_run.py` — the one-shot final run. Refuses to start unless the tree is clean, HEAD equals
  the freeze commit, that commit is pushed, and the ledger holds no prior final-run entry.
- `ann_sweep.py` — ANN behaviour on real HNSW via the standalone Qdrant binary.
- `edge_demo.py` — the two-collection architecture running our table.
- `freeze_m7_assets.py` — pins dev + untouched-final into the manifest and `frozen_eval/`.
- `field_table.py` — the objective-by-dataset field table (counts read, never hand-copied).

## Reusing this repo as a harness

The eval protocol, partitions, decontamination, bootstrap/Holm statistics, freeze and final-run
machinery are the reusable part and have been through two adversarial reviews. To run the same
question against a **different model**, add a `Spec` to `encoders.py` and set `M7_ENCODER` — do not
edit `teacher.py`. To run it with a **different query-side technique**, the surface is `table.py`
(the artifact and its one preprocessing rule) plus `train.py`'s `Cfg`; `program.py` holds the
phased plan and `sweep.py` records every run including failures.

Two things that must move together whenever the encoder changes: anything that assumes **dim 768**,
and `table.py`'s `CLS_ID` (101 is bge/BERT's; a different tokenizer has a different id). Everything
that assumes the *teacher* is bge-base now goes through the registry instead.

## Log size policy (this is a long project; context is the scarce resource)

Budgets, checked with `for f in m7/*.md; do echo $f $(( $(wc -c < $f) / 4 )); done`:

| file | budget | rule |
|---|---|---|
| `STATUS.md` | ~1.2K tokens | one screen. Rewritten, never appended. The only file always read. |
| `CODEMAP.md` | ~1.5K | grows only when a module is added or a pitfall is earned. |
| `RESULTS.md` | ~1.5K | one row per run. If it outgrows that, keep the verdict column and move detail to the run JSON. |
| `EXPLORED.md` | ~1K | one row per closed avenue. |
| `LEDGER.md` | **~4K, hard** | at 4K, compact again: keep every protocol fact verbatim, cut settled justification to one line. |

**The rule that keeps it small: never restate a number that a `results/m7_*.json` already holds.**
Put the finding and its consequence in the ledger; point at the JSON for per-component values,
per-step curves and reference rows. LEDGER.md reached 7.4K tokens by session two because this was
not being followed; compacting to 3.8K lost nothing that was not recoverable from
`results/` or `work/devres/refs.json`.

Compaction is safe despite the ledger being append-only in spirit: git preserves every prior
version (`git log -p m7/LEDGER.md`), and the file states when it was compacted.

## Pitfalls that already cost time

1. **Never `git add -A` without checking `.gitignore`.** One did, committed the multi-GB encode
   cache, and `git push` hung. History was cleaned; `work/` and `logs/` are ignored now.
2. **Subagents must be told to write only to the scratchpad.** One dumped files into `m7src/`.
3. **Run one memory-heavy job at a time.** Three at once took the WSL distro down.
4. **`np.isin` re-sorts its second argument on every call.** Use a pre-sorted array plus
   `np.searchsorted`. This turned a 2-second scan into hours.
5. **Never call a JSON loader inside a hot loop.** `mix.load_source` was re-parsing 16 MB once
   per training pair (352,190 times) and the step never finished. It is memoized now.
6. **Nothing may materialize a whole corpus.** Encodes are memmapped, the pool is chunked, the
   score matrix is tiled, and hashes are streamed. Assume 18 GB is the peak-RAM budget.
