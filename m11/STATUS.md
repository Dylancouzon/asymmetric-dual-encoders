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
| T2 zero query path → ONNX | **DONE** 2026-09-03 — two opset-17 graphs, 10 checks, parity 4.47e-08 on 1,024 real dev queries; live at `fb8e5c5b`. `m11/PLANNING.md` §T2 |
| T3 doc tower publish (PUBLIC, new repo) | fp32 passes; **fp16 fails §11.4 (1.37e-3)**, no `config.json`, no `model_tokens.onnx` |
| T4 fastembed fork branch, no PR | fork cloned to `/home/dylan/fastembed`; **upstream 0.8.0 regression found and fixed** on branch `fix-fixed-padding-ragged-batch` (breaks `thenlper/gte-base`), reported as qdrant/fastembed#703 — T4 runs against that branch |
| T5 card fixes | `MODEL_CARD.md:90` raises; must not go public as-is |
| flip `zero` PUBLIC | **moot** — already public since the first push (see the correction above). `push()` now detects this, says so, and does not pretend to have published privately first |

## Open

- ~~Licence sign-off~~ — **RULED 2026-09-03 (Dylan): MIT, including for a public release.** Card
  declares `license: mit` (matching stella) with CC BY-SA attribution for
  NQ/SQuAD/HotpotQA/FEVER/Mr.TyDi. No further approval needed on licence to flip the repo public.
- `nano`, its ONNX port, any upstream fastembed PR, and the whitepaper: blocked on M10.
- **The already-public repo is serving pre-T1 bytes.** The corrected bundle is built and passes
  all 9 gates; pushing it replaces the head. Prior revisions stay reachable — nothing
  non-releasable or secret is in them, only the two known bundle states.
- Qdrant: dense-only reproduces 0.4339 exactly; the fused 0.4911 needs **convex fusion at w=0.8**,
  which `Fusion.RRF` does not reproduce (dev 0.5504 vs 0.5727). Recorded in the card.
