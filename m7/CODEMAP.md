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
- `teacher.py` — the frozen teacher (bge-base-en-v1.5, revision-pinned). `encode_cached()` is the
  workhorse: shard-resumable, cache key covers (model, revision, dtype, tokenizer, prefix, max
  length, corpus hash), returns a **memmap** over a stitched `combined.f16`.
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

**Final and demo**
- `final_run.py` — the one-shot final run. Refuses to start unless the tree is clean, HEAD equals
  the freeze commit, that commit is pushed, and the ledger holds no prior final-run entry.
- `ann_sweep.py` — ANN behaviour on real HNSW via the standalone Qdrant binary.
- `edge_demo.py` — the two-collection architecture running our table.
- `freeze_m7_assets.py` — pins dev + untouched-final into the manifest and `frozen_eval/`.
- `field_table.py` — the objective-by-dataset field table (counts read, never hand-copied).

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
