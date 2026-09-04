# M11 status — CLOSED 2026-09-03

**Delivered: the zero half of the pair, end to end.** Two public models, both ONNX, both served by
FastEmbed as built-in entries, both byte-verified against the bytes the gates signed off. Everything
that depended on M10 moved to **M13** (`instructions-m13.md`); the image model became **M14**.

| | |
|---|---|
| query encoder | https://huggingface.co/DylanCouzon/constella-zero — commit `dc31fedf7f`, 10 files |
| document tower | https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx — commit `ab61de6e45`, 7 files |
| FastEmbed | `Dylancouzon/fastembed@add-constella-models` — 2 entries in `supported_onnx_models`, 2 canonical vectors |
| serving parity | **4.470e-08** vs the numpy reference over 1,024 real dev queries |
| upstream issue | qdrant/fastembed#703 (padding regression breaking `thenlper/gte-base`) |

**Final verification, 2026-09-03** — 8 zero gates, 18 `test_gates`, 11 ONNX parity checks, 5 doc
gates, 6 serving checks, negative control 4.475e-04, and 8 anonymous live-repo checks
(`m11/release/verify_published.py`). Re-run any of them:

    .venv/bin/python m11/release/push.py --build --gates
    .venv/bin/python m11/release/test_gates.py
    .venv/bin/python m11/release/export_onnx.py --check --no-write
    .venv/bin/python m11/release/push_doc.py --gates
    .venv/bin/python m11/release/verify_published.py
    PYTHONPATH=/home/dylan/fastembed .venv/bin/python m11/release/verify_fastembed.py

**`m11/CODEMAP.md` is the reusable part** — the ONNX-port checklist, each item paid for by a real
defect. Read it before porting nano or M14's image model, not this file.

## What shipped — `constella-zero`

**https://huggingface.co/DylanCouzon/constella-zero — PUBLIC**, commit `dc31fedf7f`. Run
`p35w-2m-s2500`, table sha `a7007b1a…`, the M7 frozen artifact unchanged; the published bytes hash
to `m7/FREEZE.json`'s `table_sha256`, re-verified anonymously after the final push.

**The repo was PUBLIC from its first push, not private**, contrary to what this file,
`m11/PLANNING.md` and `instructions-m11.md` Amendment A all said at the time. The M11a ordering
("flip PUBLIC last, after remote byte verification") assumed otherwise, so **that guarantee is
spent — it cannot be recovered, only reported.** The end state was authorised (Amendment A ruling
1, and the MIT ruling); the sequence was not. Every earlier revision stays publicly reachable,
including the pre-T1 bundle (truncation 8000, fixed-512 padding, a card whose Qdrant block raises).
Nothing non-releasable is in any of them.

Contents: `model.npz` (int8 + fp16 rows), `model.onnx` and `model_tokens.onnx` (opset 17, int8
initializer + per-row fp32 scale, ~31 MB each), `config.json`, stella's tokenizer at the pinned
revision (sanitised, see T1), `zero_encoder.py`, model card. A caller needs exactly ONE of the
three weight files.

## Tooling (this directory)

| file | what |
|---|---|
| `release/zero_encoder.py` | the shipped query path — 93 lines, numpy + tokenizers, **no torch** |
| `release/verify_bundle.py` | gate 4: **staged** encoder vs `m7src/table.py` on the **frozen source** table (5.5e-7 max-abs) |
| `release/verify_tokenizer.py` | gate 7: what fastembed's own `load_tokenizer` makes of the shipped files |
| `release/export_onnx.py` | T2: builds both graphs and re-derives all 11 parity checks |
| `release/test_gates.py` | 18 checks: 2 that must pass, 16 breakages each gate must refuse |
| `release/push.py` | build + 8 gates + upload + re-download verification |
| `release/MODEL_CARD.md` | the card; `REPO_ID` is substituted at push time |
| `release/export_doc.py` | T3: stella doc tower → ONNX, checks on 259 frozen real passages |
| `release/push_doc.py` | T3: build + 5 gates + create-private → upload → verify → public |
| `release/MODEL_CARD_DOC.md` | the doc-tower card; repo id and measured numbers substituted |
| `release/doc_fixtures.json` | 259 real nq-250k passages, six length strata, asserted on load |
| `release/verify_fastembed.py` | T4: serves the BUILT-IN model through `TextEmbedding`, 6 checks + `--negative-control` |
| `release/verify_published.py` | what the two live repos actually serve, checked anonymously |

**`m11/CODEMAP.md` is the reusable part** — the ONNX-port checklist, 24 items, each one paid for by
a T2 or T3 defect. Read it before porting nano (M13) or M14's image model, not this file.

**Eight gates**, all re-run at every push: (1) the frozen source AND the staged `model.npz` hash to
`FREEZE.json`, (2) lineage records unchanged, (3) `assert_releasable`, (4) conformance — the
**staged** encoder against the **frozen source** table, preproc rule read from `FREEZE.json`,
(5) the staging dir is exactly the manifest, (6) the card's python executes against the staging
dir, (7) fastembed's loader gets the frozen 512 rule and dynamic padding, (8) the staged ONNX graphs
re-checked against the numpy path by re-running the parity arithmetic, not by reading a recorded
verdict. `--push` requires `--build` in the same invocation; after upload the commit is
re-downloaded and compared file by file.

`test_gates.py` is what makes the gates worth having — passing gates prove nothing on their own,
since the previous set all passed while gate 4 compared the bundle against itself.

**Scope note (Dylan, 2026-09-03): "do not over-engineer anything; the code should be a realistic
representation of what can be merged."** Two adversarial reviews had been briefed to name any
wrong-but-passing bundle, which produced a malicious-actor threat model — nine gates, committed
tokenizer digests, 30,522-row vocab round-trips, PR-ref-then-merge uploads, a 22-case attack
suite. Rolled back to the accident threat model that actually applies (one researcher, one box, a
research artifact). Their findings were kept **only where the fix also catches a plausible
mistake**: gate 4 importing the source encoder instead of the staged one, gate 4's reference table
being the bundle's own, and the mutable `.meta.json` rather than `FREEZE.json` setting the shipped
preproc rule. Dropped: digest-snapshot machinery, structural tokenizer pinning, the git-clean
freeze gate, the PR-ref upload dance, and speculative ONNX gating.

## Two traps the release found

- **stella crashes without `config_kwargs={"use_memory_efficient_attention": False,
  "unpad_inputs": False}`** (`assert ... 'please install xformers'`). Same pinned setting as
  `FREEZE.json`'s `encoder_spec`. Every doc-side snippet must carry it.
- **stella's `tokenizer.json` ships with padding-to-512 enabled.** A naive `tokenizers` load puts
  ~500 `[PAD]` rows in every bag; cosine against the frozen path drops to 0.35. `zero_encoder.py`
  calls `no_padding()`. The transformers path never saw this because padding is off by default there.

## M11a — the slice, all done (2026-09-03)

Rulings: `instructions-m11.md` Amendments A and B. Tasks and evidence: `m11/PLANNING.md`.
No qrels were read and nothing was scored: the dev **query texts** and nq-250k **passage
texts** were used as parity fixtures only. The reserved access is unspent.

| task | outcome |
|---|---|
| T0 bind the release path | gates bound to a build snapshot; `test_gates.py` proves 13 attacks refused |
| T1 sanitise tokenizer | `push.sanitise_tokenizer`; gate 7 measures what FastEmbed's own loader gets |
| T2 zero query path → ONNX | two opset-17 graphs, 11 checks, parity 4.47e-08 on 1,024 dev queries |
| T3 doc tower → ONNX, published | fp32 only — **fp16 rejected on a CUDA measurement**; 259 frozen real-passage fixtures |
| T4 FastEmbed integration | built-in registration, **not** `add_custom_model`; `parallel>1` works as a result |
| T5 card fixes | folded into T2/T6 |
| T6 rename + card rewrite | `constella-zero`; both cards FastEmbed-first, lean, no competitive framing |

Reviewed adversarially at every stage (Codex + Fable); all logs audited clean for reserved-set
reads. The reviews changed the design twice — see `m11/PLANNING.md`.

## Released: stella document tower, ONNX

**https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx — PUBLIC**, commit `ab61de6e45`,
7 files, 1.75 GB. fp32 only (`model.onnx`, opset 17, standard domain, no external data). First push
`e34cc6dd1e` went private → uploaded → verified against a post-gate hash snapshot → flipped public;
later card rewrites were pushed over it with `--update`, and the graph's LFS sha256 (`fe31555e…`) is
unchanged between the two. Verified again
anonymously after the fact: the published LFS sha256 `fe31555e…` is the gated byte string.

Parity vs the torch module on 259 frozen real NQ passages: **cos 0.99999988, max-abs 3.76e-07**,
batch invariance bit-identical, and **CUDA vs CPU min-cos 1.000000 / max-abs 9.07e-05** so an index
built on one holds on the other. Build+gates+push: `m11/release/push_doc.py`.

## Addendum 2026-09-04 — the doc graph IS a complete query encoder

Verified live, not inferred. stella is one tower used symmetrically: prepend the `s2p_query` prompt
before tokenizing and the PUBLISHED doc graph reproduces the torch query path at **min-cos
1.00000000, max-abs 8.9e-08** on 8 queries. No weights are missing and there is no second artifact
to export. What is missing is **prompt handling in the serving layer** — neither the graph nor
FastEmbed's `query_embed` prepends it, and unprompted queries embed at cos **0.80** to the correct
vector (card says 0.82–0.87 on five; 0.80 is the min over eight). Silent, not an error.

Consequence for the Cloud Inference ask: Cloud Inference already auto-prefixes the E5 family by
call type. Applied to stella's prompt pair, **one hosted graph serves documents, full-quality
queries, and the index zero/nano target** — three frontier points, not one.

`MODEL_CARD_DOC.md`'s "the only supported way to embed queries" (Sentence Transformers) overstates
the restriction — it is about tooling, not the graph. **Card left AS IS (Dylan, 2026-09-04.)** The
conservative wording protects users from FastEmbed's `query_embed`, which silently returns a
wrong-protocol vector; the graph's true capability is recorded here instead.

## T3 — what it cost and what it caught (2026-09-03)

Seven defects, none of which would have announced itself. The two that matter most:

- **A CPU parity gate cannot qualify a reduced-precision ONNX graph.** The fp16 document graph
  passes CPU parity (cos 0.99999923 on all 259 fixtures) only because ORT up-converts fp16 to
  fp32 — the same fact that makes it 9.9x SLOWER on CPU. On CUDA it disagrees with fp32 on
  **255 of 259 passages** (min-cos 0.662). Shipping on the CPU number would have put inverted
  advice on the card. fp32 only. `results/m11_doc_fp16_gpu.json`.
- **`2_Dense_1024` has a bias**, and the M9 builder selected the weight by rank
  (`first tensor with dim()==2`). The matrix is square, so a wrong or transposed key stays
  shape-valid and a torch reference built the same way certifies it. Assert exact state-dict keys.

Also: `convert_float_to_float16(node_block_list=…)` emits an unloadable graph (duplicate node
names + broken topological order); a bare dot product recorded as `min-cos` flips an fp16 verdict
(dot 0.99954 vs true cosine 0.99999940); `export_onnx.py` check 1 claimed "checker passes" and
never ran the checker, so `zero`'s live gate 8 asserted something it did not test (**fixed**);
`PoolingType.DISABLED` serves an already-pooled graph, so no per-token graph is needed;
`parallel>1` cannot work for `add_custom_model` repos in fastembed 0.8.0.

New files: `m11/release/export_doc.py`, `push_doc.py`, `MODEL_CARD_DOC.md`, `doc_fixtures.json`
(259 real nq-250k passages, six strata, re-asserted on load).

## Carried forward

- **`nano`, its ONNX port, its FastEmbed entry, and the whitepaper are M13** — blocked on M10,
  which is blocked on the cloud GPU budget. `instructions-m13.md`.
- **The upstream FastEmbed PR is M13** (Dylan, 2026-09-04): one clean PR adding all three model
  entries once nano exists, not a two-model PR now. The branch that exists today,
  `add-constella-models`, is deliberately not mergeable — it also carries the #703 padding fix, and
  the models sit on a personal account where upstream hosts under `Qdrant/`.
- **Qdrant fusion**: dense-only reproduces 0.4339 exactly; the fused 0.4911 needs convex fusion at
  w=0.8, which `Fusion.RRF` does not reproduce. Stated on the card.
- **Spent, not recoverable**: `constella-zero` was public from its first push, so the "flip public
  last, after remote byte verification" ordering guarantee was never held. Every earlier revision
  stays publicly reachable, including the pre-T1 bundle. Nothing non-releasable is in any of them.
