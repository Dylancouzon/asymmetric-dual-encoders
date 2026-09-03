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

## M11a in flight (opened 2026-09-03, branch `m11-work`)

The zero half does not depend on M10. Four rulings and the slice: `instructions-m11.md`
Amendment A. Tasks, graph design and gates: `m11/PLANNING.md`. Nothing here reads a quality set.

Reviewed adversarially before execution (Codex + Fable, 2026-09-03); both logs audited clean for
reserved-set reads. The review moved two blockers ahead of everything else — **nothing is
published until T0 and T1 land.**

| task | state |
|---|---|
| T0 bind the release path | **blocker** — gates hash the source table, not the uploaded bundle; gate 4 self-compares |
| T1 sanitise tokenizer (both repos) | **blocker** — stella ships truncation 8000 + fixed-512 padding; fastembed mistruncates >512-token inputs and crashes on mixed batches |
| T2 zero query path → ONNX | pending — design verified exact; int8 initializer, no `Unique`, two graphs |
| T3 doc tower publish (PUBLIC, new repo) | fp32 passes; **fp16 fails §11.4 (1.37e-3)**, no `config.json`, no `model_tokens.onnx` |
| T4 fastembed fork branch, no PR | fork cloned to `/home/dylan/fastembed`; **upstream 0.8.0 regression found and fixed** on branch `fix-fixed-padding-ragged-batch` (breaks `thenlper/gte-base`) — T4 runs against that branch |
| T5 card fixes | `MODEL_CARD.md:90` raises; must not go public as-is |
| flip `zero` PUBLIC | **last**, after remote byte verification — `create_repo(exist_ok=True)` ignores `private` |

## Open

- ~~Licence sign-off~~ — **RULED 2026-09-03 (Dylan): MIT, including for a public release.** Card
  declares `license: mit` (matching stella) with CC BY-SA attribution for
  NQ/SQuAD/HotpotQA/FEVER/Mr.TyDi. No further approval needed on licence to flip the repo public.
- `nano`, its ONNX port, any upstream fastembed PR, and the whitepaper: blocked on M10.
- Qdrant: dense-only reproduces 0.4339 exactly; the fused 0.4911 needs **convex fusion at w=0.8**,
  which `Fusion.RRF` does not reproduce (dev 0.5504 vs 0.5727). Recorded in the card.
