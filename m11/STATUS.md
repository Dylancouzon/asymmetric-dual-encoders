# M11 status

**Partially delivered ahead of schedule (2026-09-03, Dylan's ask): `zero` is published.** The rest
of M11 (nano, ONNX, fastembed, whitepaper) waits on M10, which waits on cloud GPU budget.

## Released: `zero` v1

**https://huggingface.co/DylanCouzon/zero-query-encoder-v1 — PUBLIC.** Run `p35w-2m-s2500`,
table sha `a7007b1a…`, the M7 frozen artifact unchanged. Verified by clean re-download: published
bytes hash to `m7/FREEZE.json`'s `table_sha256`.

**CORRECTION 2026-09-03: the repo has been PUBLIC since the first push, not private.** Anonymous
`GET https://huggingface.co/api/models/DylanCouzon/zero-query-encoder-v1` returns 200 with
`private: false`; three commits, all 2026-09-03 15:23–15:37 UTC. This file, `m11/PLANNING.md` and
`instructions-m11.md` Amendment A all said PRIVATE, and the M11a ordering ("flip PUBLIC last,
after remote byte verification") assumed it. **The ordering guarantee is spent — it cannot be
recovered, only reported.** Amendment A ruling 1 and the MIT ruling already make public the
intended end state, so the outcome is authorised; the sequence was not. Consequence: the bytes
publicly served between 15:23 and the T1 push are the PRE-T1 bundle — tokenizer truncation 8000,
fixed-512 padding, and a card whose Qdrant block raises. Every earlier revision stays publicly
reachable at its commit regardless of what is pushed next.

Contents: `model.npz` (int8 + fp16 rows), `model.onnx` and `model_tokens.onnx` (opset 17, int8
initializer + per-row fp32 scale, ~31 MB each), `config.json`, stella's tokenizer at the pinned
revision (sanitised, see T1), `zero_encoder.py`, model card. A caller needs exactly ONE of the
three weight files.

## Tooling (this directory)

| file | what |
|---|---|
| `release/zero_encoder.py` | the shipped query path — 89 lines, numpy + tokenizers, **no torch** |
| `release/verify_bundle.py` | gate 4: **staged** encoder vs `m7src/table.py` on the **frozen source** table (5.5e-7 max-abs) |
| `release/verify_tokenizer.py` | gate 7: what fastembed's own `load_tokenizer` makes of the shipped files |
| `release/export_onnx.py` | T2: builds both graphs and re-derives all 10 parity checks |
| `release/test_gates.py` | 14 checks: 1 positive control, 13 breakages each gate must catch |
| `release/push.py` | build + 8 gates + upload + re-download verification |
| `release/MODEL_CARD.md` | the card; `REPO_ID` is substituted at push time |
| `release/export_doc.py` | T3: stella doc tower → ONNX, checks on 259 frozen real passages |
| `release/push_doc.py` | T3: build + 5 gates + create-private → upload → verify → public |
| `release/MODEL_CARD_DOC.md` | the doc-tower card; repo id and measured numbers substituted |
| `release/doc_fixtures.json` | 259 real nq-250k passages, six length strata, asserted on load |

**`m11/CODEMAP.md` is the reusable part** — the ONNX-port checklist, 19 items, each one paid for by
a T2 or T3 defect. Read it before porting nano or M12's image model, not this file.

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

## M11a in flight (opened 2026-09-03, branch `m11-work`)

The zero half does not depend on M10. Four rulings and the slice: `instructions-m11.md`
Amendment A. Tasks, graph design and gates: `m11/PLANNING.md`. Nothing here reads a quality set.

Reviewed adversarially before execution (Codex + Fable, 2026-09-03); both logs audited clean for
reserved-set reads. The review moved two blockers ahead of everything else — **nothing is
published until T0 and T1 land.**

| task | state |
|---|---|
| T0 bind the release path | **DONE** 2026-09-03 — 9 gates on a build snapshot; Codex reviewed the fix and broke it, all 9 findings actioned; `test_gates.py` proves 13 attacks refused |
| T1 sanitise tokenizer (`zero` repo) | **DONE** 2026-09-03 — `push.sanitise_tokenizer`; gate 8 measures what fastembed's own loader gets; the doc-tower repo still needs the same edit under T3 |
| T2 zero query path → ONNX | **DONE** 2026-09-03 — two opset-17 graphs, 10 checks, parity 4.47e-08 on 1,024 real dev queries; live at `fb8e5c5b`. `m11/PLANNING.md` §T2, incl. the measurement that the count-mask defect is unreachable by real text (0/7,325 dev queries produce id 0) |
| T3 doc tower publish (PUBLIC, new repo) | **DONE 2026-09-03** — live at commit `e34cc6dd1e`, PUBLIC, byte-verified anonymously (published LFS sha256 == gated bytes, `fe31555e…`). Repo `DylanCouzon/stella-en-400M-v5-doc-onnx`. Re-exported fp32 from the pinned revision; 259 frozen real-passage fixtures; **fp16 rejected on a CUDA measurement**, `model_tokens.onnx` proved unnecessary. `m11/PLANNING.md` §T3 |
| T4 fastembed fork branch, no PR | fork at `/home/dylan/fastembed`; **upstream 0.8.0 regression found, fixed and filed as qdrant/fastembed#703** (breaks `thenlper/gte-base`), branch `fix-fixed-padding-ragged-batch`. **De-risked by T3**: the `DISABLED` route is settled (bit-identical to ORT) and `parallel>1` is settled (cannot pass — not a gate). Remaining: serve `zero` end to end, gate parity vs the numpy encoder, leave the branch PR-ready. |
| T5 card fixes | **DONE** 2026-09-03 — the raising snippet fixed, ONNX usage block added, the by-caller tokenizer table and cost rows corrected; gate 6 executes every block |
| T6 rename + card rewrite (**after T4**) | `zero` → **`constella-zero`** (name locked in `m8/LEDGER.md` §6.1; milestone suffix dropped, Dylan 2026-09-03). Both cards rewritten: fastembed examples, competitive comparison and missed-bar framing removed, contamination caveat and measured numbers kept, more about the model itself. `m11/PLANNING.md` §T6 |
| flip `zero` PUBLIC | **moot** — already public since the first push (see the correction above). `push()` now detects this, says so, and does not pretend to have published privately first |

## Released: stella document tower, ONNX

**https://huggingface.co/DylanCouzon/stella-en-400M-v5-doc-onnx — PUBLIC**, commit `e34cc6dd1e`,
7 files, 1.75 GB. fp32 only (`model.onnx`, opset 17, standard domain, no external data). Created
private → uploaded → verified against a post-gate hash snapshot → flipped public. Verified again
anonymously after the fact: the published LFS sha256 `fe31555e…` is the gated byte string.

Parity vs the torch module on 259 frozen real NQ passages: **cos 0.99999988, max-abs 3.76e-07**,
batch invariance bit-identical, and **CUDA vs CPU min-cos 1.000000 / max-abs 9.07e-05** so an index
built on one holds on the other. Build+gates+push: `m11/release/push_doc.py`.

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

## Next session starts here (2026-09-03)

**T0, T1, T2 are done, pushed, and live.** Working tree clean on `m11-work`. `zero` is at HF head
`1cfae6cc`, 10 files, PUBLIC, byte-verified. Re-check anything with:

    .venv/bin/python m11/release/push.py --build --gates    # 8 gates, uploads nothing
    .venv/bin/python m11/release/test_gates.py              # 14 checks, must all hold
    .venv/bin/python m11/release/export_onnx.py --check     # 11 ONNX parity checks

**Next is T3** (`m11/PLANNING.md` §T3): five gaps, artifacts confirmed present on this box.
Then T4 (fastembed, fork branch only, no PR) and T5 (done — folded into T2's card work).

**Read `§Scope note` below before adding any gate or test.** Two adversarial reviews produced a
malicious-actor threat model; the rollback to the accident model is Dylan's explicit instruction,
not a shortcut.

## Open

- **`zero` is published under the WRONG NAME.** The family name was locked in `m8/LEDGER.md` §6.1
  as `constella`, and `:716` required a ruling on the milestone suffix **before anything shipped**.
  That ruling was not sought and the push went out as `zero-query-encoder-v1`. Ruled 2026-09-03:
  **`constella-zero`**. Rename in T6; `move_repo` leaves a redirect so the old URL keeps working.
- ~~Licence sign-off~~ — **RULED 2026-09-03 (Dylan): MIT, including for a public release.** Card
  declares `license: mit` (matching stella) with CC BY-SA attribution for
  NQ/SQuAD/HotpotQA/FEVER/Mr.TyDi. No further approval needed on licence to flip the repo public.
- `nano`, its ONNX port, any upstream fastembed PR, and the whitepaper: blocked on M10.
- **The already-public repo is serving pre-T1 bytes.** The corrected bundle is built and passes
  all 9 gates; pushing it replaces the head. Prior revisions stay reachable — nothing
  non-releasable or secret is in them, only the two known bundle states.
- Qdrant: dense-only reproduces 0.4339 exactly; the fused 0.4911 needs **convex fusion at w=0.8**,
  which `Fusion.RRF` does not reproduce (dev 0.5504 vs 0.5727). Recorded in the card.
