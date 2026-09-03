# M11 status

**Partially delivered ahead of schedule (2026-09-03, Dylan's ask): `zero` is published.** The rest
of M11 (nano, ONNX, fastembed, whitepaper) waits on M10, which waits on cloud GPU budget.

## Released: `zero` v1

**https://huggingface.co/DylanCouzon/zero-query-encoder-v1 — PRIVATE.** Run `p35w-2m-s2500`,
table sha `a7007b1a…`, the M7 frozen artifact unchanged. Verified by clean re-download: published
bytes hash to `m7/FREEZE.json`'s `table_sha256`.

Contents: `model.npz` (int8 + fp16 rows), `config.json`, stella's tokenizer at the pinned
revision, `zero_encoder.py`, model card.

## Tooling (this directory)

| file | what |
|---|---|
| `release/zero_encoder.py` | the shipped query path — 89 lines, numpy + tokenizers, **no torch** |
| `release/verify_bundle.py` | conformance: shipped encoder vs `m7src/table.py` (5.5e-7 max-abs) |
| `release/push.py` | build + 4 gates + upload; private unless `--public` |
| `release/MODEL_CARD.md` | the card; `REPO_ID` is substituted at push time |

Four gates, all re-run at every push: (1) table bytes hash to `FREEZE.json`, (2) both lineage run
records hash to what the freeze recorded, (3) `freeze.assert_releasable`, (4) conformance.

## Two traps the release found

- **stella crashes without `config_kwargs={"use_memory_efficient_attention": False,
  "unpad_inputs": False}`** (`assert ... 'please install xformers'`). Same pinned setting as
  `FREEZE.json`'s `encoder_spec`. Every doc-side snippet must carry it.
- **stella's `tokenizer.json` ships with padding-to-512 enabled.** A naive `tokenizers` load puts
  ~500 `[PAD]` rows in every bag; cosine against the frozen path drops to 0.35. `zero_encoder.py`
  calls `no_padding()`. The transformers path never saw this because padding is off by default there.

## Open

- **Licence sign-off.** Card declares `license: mit` (matching stella) with CC BY-SA attribution
  for NQ/SQuAD/HotpotQA/FEVER/Mr.TyDi. Fine while private; **needs Dylan's explicit answer before
  any public flip**, per the CLAUDE.md licensing rule.
- Deliverables 2–4 (ONNX incl. the document tower, fastembed, whitepaper) and `nano`: blocked on M10.
- Qdrant: dense-only reproduces 0.4339 exactly; the fused 0.4911 needs **convex fusion at w=0.8**,
  which `Fusion.RRF` does not reproduce (dev 0.5504 vs 0.5727). Recorded in the card.
