# M10 code map

`m10src/` imports `m7src` and `m9src` and edits neither. `m8/CODEMAP.md` and `m9/CODEMAP.md` still apply.

| module | what | artifact |
|---|---|---|
| `rank_probe.py` | PCA-k projection of stella query vectors, retrieval retention on the CQA dev components; 1024/768/256-d heads | `results/m10_rank_probe_mac.json` |
| `rank_probe_mix.py` | the same from caches, with bases fit on NQ, the other component, mixtures, oracle | same file, `mixture_bases_1024d` |
| `head_width_probe.py` | frozen bge-small + ridge head to stella targets with 384 / 768 / 1152-d pooled features | `results/m10_head_width_probe_mac.json` |
| `head_width_parity.py` | ONNX export of the per-token three-layer head; fastembed serving parity | `results/m10_head_width_parity_mac.json`, `work/m10onnx/nano-3layer/` |
| `head_mlp_parity.py` | the same export/parity check for a per-token **nonlinear** head; `residual` mode (`W_lin·x + W₂·GELU(W₁·x+b₁)`, W₁ 1152→192) is arm G-MLP, `mlp` mode is the cut bottleneck form; proves the serving path and the parameter count (34.96M) | `results/m10_head_mlp_parity_box.json`, `work/m10onnx/nano-3layer-mlp/` |
| `scripts/m10_conjunct_arithmetic.py` | per-dataset retention each conjunct demands, uniform-retention planning proxies (0.025 quantile), per-dataset stress scenarios, fiqa's share of the avg-6 margin (comparator rows only) | `results/m10_conjunct_arithmetic.json` |
| `forms.py` | the 12 synthetic-query form prompts, output contract and parser. **`RUBRIC` is the frozen gate standard; `FORMS` is the revisable prompt** | — |
| `cov_admit.py` · `cov_screen.py` · `cov_ledger.py` · `cov_encode.py` | the COV surface: admission, the fingerprint screen, LEDGER's page split, teacher encoding | `work/m10cov/*.json` |
| `cov_probe.py` · `cov_macro.py` · `cov_resolution.py` | the two non-candidate probes, **the family-weighted macro and its paired stratified bootstrap** (the estimator every screen contrast uses), and the resolution number | `results/m10_cov_resolution.json` |
| `protected10.py` | the M10 protected index = M7's protected queries + admitted COV queries AND documents + the `arxiv-title` draw. Cached on an identity that names every component | `work/m10cov/protected10/` |
| `screen_lock.py` | validates `m10/screen_registry.json`, which IS §0a. **The only reader a rule may use** | — |
| `student_parity.py` | the family-F serving-parity gate: a failing head disqualifies its arm | `results/m10_student_parity_box.json` |
| `wikibody.py` | the `wikipedia-body` seed store: lead exclusion, chunking, the full-dump scan, the screens, the build draw, and T2-8 rung 1's subject filter | `work/m10gen/wikibody_*.json` |
| `seeds.py` | the registered topical router, the seed draw and its cache. `SCREEN_VERSION` is DERIVED from `protected10._ident()` | `results/m10_seed_supply.json` |
| `gate_sample.py` | the blinded, interleaved judged-precision gate and its controls | `results/m10_wikibody_precision*.json` |
| `onform_diag.py` | T2-7's report-only build on-form diagnostic; **admits nothing** | `results/m10_onform_diag_*.json` |
| `arxiv_draw.py` | the registered `arxiv-title` draw from the Kaggle artifact | `work/m10arxiv/arxiv_draw.json` |
| `nano10.py` | the M10 student (per-token head, pooled after), the family-D objectives, the mix window, the cyclic schedule, the kill and plateau rules, and the ONNX export | — |
| `trainer10.py` | the training loop, checkpointing and resume | — |
| `rescreen10.py` | the M9 query pool and document pool re-screened against `protected10` (`instructions-m10.md`:462); the keep mask is cached on (protected index identity, pool identity) and a training path refuses without it | `results/m10_rescreen10.json`, `work/m10cov/rescreen10/` |
| `corpus_loader.py` | the M10 query corpora → the trainer: declared sources (M9 pool · PAQ · harvest · the generated half, unbuilt), the hashed manifest, packed pretokenization, the data cut, and the **form-balanced sampler** (`balanced=False` is A2's variant) | `results/m10_corpus_manifest.json` |
| `targets10.py` | stella query targets keyed by **content hash** into an append-only fp16 memmap — resumable, appendable, and cold == warm by construction | `work/m10targets/`, `results/m10_targets10.json` |
| `m10/report-draft.html` | source of the owner report artifact (https://claude.ai/code/artifact/fce61c94-5444-4c78-bb2e-46112cb7547a); republish from a session, never edit the live page | — |

## Pitfalls this milestone earned

1. **Stella runs on the Mac only in `.venv-mac`** (transformers 4.57); transformers 5.x breaks its remote code (`get_extended_attention_mask(..., device=)`). MPS: ~170 short queries/s, 20–100 documents/s by length.
2. **fastembed 0.8.0 custom models need `config.json` and `special_tokens_map.json` in the model directory**, and transformers 5.x fast tokenizers no longer write `special_tokens_map.json` — write it from `tok.special_tokens_map`.
3. **A per-token linear head over concatenated layer states is exactly reproduced by fastembed's mean pooling** (2e-7), because mean pooling is linear. A **post-pooling** nonlinear head is not; a per-token nonlinear head is (`head_mlp_parity.py`).
4. **PCA of teacher vectors is the reconstruction-optimal subspace, not a retrieval bound.** Say "the subspace L2 regression aims at", never "upper bound" (Codex pass 4).
5. **fastembed serves `min(model_max_length, max_length)` from `tokenizer_config.json`**, and `all-MiniLM-*-v2` ships `max_length` 128 beside `model_max_length` 512. A correct export therefore read 0.93–0.95 min-cos against a 512-token reference and would have disqualified both MiniLM students. Write both keys explicitly on every export. bge-small ships no `max_length` key, which is why it alone read 1.0.
6. **`load_dataset_builder(...).info.features` can be STALE** — it cost the LEDGER admission a wrong refusal. Load the rows.
7. **Sub-8-word boilerplate fakes contamination.** Always re-screen length-filtered.
8. **An encode cache that stores fp16 must ROUND-TRIP on the cold path too**, or a first run and a re-run score differently (`cov_probe.encode`).
9. **A `limit=` smoke must not write to a shared report path.** One did, leaving `complete: false` where the full scan's report belongs; `_load_jsonl` refused it, but the trap was the defect. Reports live beside their own output.
10. **The tests and the validators run only under `.venv/bin/python`** — the system python has no numpy, and that ImportError is not a failure. `uv pip` is the installer (`.venv` has no `pip`).
11. **A data loader that counts its own calls cannot resume.** `data10.batch_fn` advanced a
    per-kind counter, so an arm resumed at step 34 restarted both streams at batch 0 and re-trained
    seen data. `test_trainer10`'s resume test could not see it — its `batch_fn` is a pure function
    of the step, so the defect lived only in the data path. The position is now derived from the
    step (`data10.kind_index`). **Ask of any resume test what it stubs out.**
12. **A corpus file grouped by form makes any prefix one form.** `harvest_train.jsonl` is claim,
    then keyword, then title, so `head_per_source` is a smoke device and never a sample; every real
    draw is uniform (`apply_data_cut`, `harvest.draw`, `paq.draw`, `corpus10.draw_seeds`).
13. **`batch_tokens` above ~32768 hits `CUDA driver error: device not ready`** on this card, the
    same failure as `E-bs128`. Measured stella query-encode rates: **345 texts/s** on harvested
    text, **484** on PAQ questions, at the M9 default.
14. **`protected10.build()` must run BEFORE anything imports `m9base`.** It reads the reserved
    QUERY text through `decontam.protected_query_index`, and `paths_guard` (installed by `m9base`)
    refuses that with no allowlist claim. `harvest.draw` relies on the same order; `rescreen10`'s
    CLI builds the index as its first statement. Training paths never build it -- they read the
    cached mask.
15. **A reader must never truncate an append-only store.** `TargetCache._truncate` ran on every
    open, so opening the cache read-only while an encode was appending would have cut the chunk in
    flight. Excess bytes are now repaired only by a writer holding the lock; a reader tolerates
    them (`n` is the authority) and refuses only a SHORT file.
