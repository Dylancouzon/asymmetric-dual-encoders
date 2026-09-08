# M10 status — 2026-09-08. **Family F DONE and F1 RESOLVED: the student is bge-small.** **W8 band-1 remainder RUNNING** (11 arms, ~25 h, `work/m10arms_run_rest.sh`). §0b registered; the screen is live.

**Read this, then `m10/LEDGER.md`.** The box is **preparation for the cloud GPU run, not a
measurement target** (Dylan, 2026-09-05): run as much as it can here first, no re-shaping; the
remainder moves to the A100 under the same registry. **The weekend timeline is not binding.**
Working model (Dylan, 2026-09-05): the session is the ML lead, Opus/Sonnet subagents do the build
work, Codex is the adversarial reviewer.

## RUNNING: `work/m10arms_run_F.sh` → F-bge-small then F-MiniLM-L6, under `m10src/memtrace.sh`

**TRAINING CONFIRMED 19:53** — `step 12500/625000 loss 0.3898 lr 9.54e-05 628 ex/s`, the first
arm to get past data prep after three crashes. **628 ex/s measured in the real arm**, 1.23× the
registered `PLAN_RATES[32] = 512`, so **~8.8 h each**, not 10.9 (and not the old 6.2, which read
the 890 ex/s `fixed_bucket_compile` row — `--compile` is SMOKE-ONLY for registered arms).
`PLAN_RATES` is deliberately NOT changed mid-run: the W8 band arithmetic reads it, and 512 being
conservative costs nothing. Re-derive it at the next planning pass. Kill by PID only.
`--resume` needs a parseable non-terminal record AND the rolling checkpoint — it never falls back
to a fresh run, and the arm dir was empty after the crashes, so this is a clean fresh start.

**Memory: RESOLVED and VALIDATED (2026-09-07).** Two causes: document targets materialized
(19.1 GiB at 5M → `DocTargetView`) and ~3 GB of glibc arena residue in the doc-id stream
(→ `release_arena()`). **Full-dose pre-flight at the registered `n_docs=5,000,000` PASSES**:
peak arm RSS **8,228 MB** (was 24,383), peak guest used **9,799** (cap 26,624), and the metric that
actually failed — **min host free 3,297 MB, against 239 MB pre-fix**. LEDGER has the numbers, four
withdrawn claims of mine, and four open non-blocking items.

**If it crashes again, read this first:** the peak is dominated by `m9src/data.row_texts`, not the
tokenizer — ten rows cost 2,200 MB `RssAnon` because it materializes a whole store via
`mix.load_store`, once per chunk. Left unfixed on purpose (guard9 "train"/"eval" scope, M9's
close-out pending); the M10-side fix is to sort the draw by store and load each once.
Trust `arm_hwm_mb` in the trace, not `arm_rss_mb` — a 5 s sample of an instantaneous gauge is a
lower bound. `n_match=0` prints **-1**, so a bad pattern cannot look like an idle arm.

## RUNNING: the W8 band-1 remainder, 11 arms in registered order (~25 h)

`work/m10arms_run_rest.sh` (tracked copy `m10src/run_rest.sh`), launched 10:54 after **12/12 on the
90-step CUDA shape smoke with the current code** (`results/m10_arm_smoke.json`, `all_shapes_pass`).
Order is the registry's: ANCHOR · A1 · A2 · A3 · G-384 · G-1536 · G-MLP · B-100/0 · B-50/50 ·
D-NORM · D-COV. **C is CUT; E-bs128 is CLOUD_ONLY** (the A100) and `run_arm` refuses it on the box.

Doses are NOT uniform — read them from `run_arm.py <arm> --plan`, not from the screen dose:
5,000,000 for ANCHOR/A/G/D (156,250 steps), **3,750,000 for B-100/0**, **7,500,000 for B-50/50**.
Projected 52.2 box hours for the whole band at the conservative 512 ex/s, of which F's 21.7 h are
already spent; the remainder is ~30 h projected, ~25 h at the measured 624.

**`results/m10_F_verdict.json` is what unblocks these arms.** `run_arm.f_verdict` refuses every
post-F arm without it — *"this runner never guesses the student"* — and validates it hard: the
schema (`winner`, `contrast{rule,point,lower,resolved}`, `registry_sha256`,
`sha256_of_F_records`), that it was decided under the CURRENT registry hash, that each F record
hashes exactly, that each has `status: "complete"` and a `final_checkpoint_sha256`, and that the
winner is the student of a `trained`, uncut family-F arm. Writing the contrast to
`m10_contrast_F1.json` alone left all 11 arms refused in 14 s.

**The chain distinguishes a refusal from a failure:** non-zero within 120 s = a shared
precondition, so it STOPS; a later non-zero is that arm's own outcome and the chain continues. The
first version lacked this and burned all 11 arms on one missing file.

## F chain — arm 1 done, arm 2 in flight

| | F-bge-small | F-MiniLM-L6 |
|---|---|---|
| status | **complete, `exit 0`** 04:41:25 | running, ~60% at 08:12 |
| rate | 624 ex/s, 33,298 s (9.25 h) | **1,066 ex/s** (6 layers vs 12), ETA ~09:55 |
| `stopped` | **`plateau at cycle 3`** = registered SUCCESS (`_stopped_note`: PLATEAU_FROM_CYCLE is 3 and a screen arm runs 3 cycles, so it finished its dose and stopped one step short). `complete: true` | — |
| COV cycle ends | 0.5080 → 0.5137 → **0.5160** (+0.0057, +0.0023) | 0.4923 (cycle 1) |
| artifacts | `results/m10_arm_F-bge-small.json`, `work/m10arms/F-bge-small/{record.json,ckpt.pt,cov_*.json}` | — |

**No verdict exists yet and none may be inferred here.** `f_verdict` is `null` and `_contrasts`
says *"NOT computed here — the contrast step reads the per-query COV files and passes each
contrast's own registered quantile"*. The raw macro gap at the four matching read points is
**0.013–0.016 in bge-small's favour** (0.4871/0.4889/0.4923/0.4942 vs 0.4996/0.5048/0.5080/0.5075),
wider than the 0.0086 resolution distance — but that is a **raw difference, not the registered
statistic**, and the contrast is a separate step over the per-query files.

**DEV-6 ran once at the final checkpoint** as registered (`nq-250k`, `hotpotqa`,
`cqadup-programmers`, `cqadup-physics`, `heldout-train`, `heldout-longq`) and is **never
selection-bearing** — it informs nothing here, and its CQADupStack components are exactly where
M9's coverage failure showed, which makes it tempting to read as a verdict. It is not one.

## What is DONE and verified

| step | outcome |
|---|---|
| **0a/0b/1/2/3** | vLLM + generator smoke; all seven generated forms approved; four COV families admitted (13,416 queries); §0a design lock |
| **4 harvest** | **A3 corpus = 1,250,000 rows** from 21,087,043 harvested (wiki 16.06M · arXiv 4.98M · pool 46.6K). title/keyword/claim at 417K/417K/416K, no form short. FORMS-12 hold-out applied by DOCUMENT across all forms: **1,500 docs held → 1,614 eval rows, 1,248,386 train rows** (`work/m10harvest/harvest_{train,forms12}.jsonl`) |
| **5 PAQ** | **A2 4,037,000** (`8f32bcdf…`) + **build 1,000,000** nested inside it, from `dl.fbaipublicfiles.com`, CC BY-SA 3.0 shipped in the tarball. Protected screen removed only 0.35% |
| **6 trainer** | `nano10` · `data10` · `trainer10` · `qfilter` · `corpus10` · `arm_smoke` · `screen_lock` · **`corpus_loader` · `targets10` (built 2026-09-05 evening, Codex review pending)**. **190 tests green.** `assemble_arm` is the only launcher path. M9 pools re-screened against the M10 protected index (709 queries / 79,630 docs removed). Arm-shape smoke **12/12 on CUDA at max_len 512** |
| **M10.0-e** | **COMPLETE.** P0 0.477528 · P1 0.476141 · P2 0.473892. **Same-init distance 0.00288**, seed effect 0.00139 (n=1). `results/m10_calib_report.json` |
| **seeds** | resolved for **all seven** generated forms; health/finance 33,000 each from `wikipedia-body`, howto 37,927, the other four route `"general"` |

## OPEN — needs Dylan (none blocks the next step; W12/W13 ruled 2026-09-05, kept one line each)

| # | what |
|---|---|
| ~~W12~~ | **RULED: STOP removed.** Family A reports three labels and its DEV-6 CQADupStack + FORMS-12 reads; C1b is the failure condition (`instructions-m10.md` §Amendment 2026-09-05 C1) |
| ~~W13~~ | **RULED: L12 CUT.** F = bge-small vs MiniLM-L6 at 20M, 12 contrasts, 14 trained arms (§Amendment 2026-09-05 C2) |
| **paired row** | **DRAFT REGISTRATION READY for ratification: `research/m10-paired-row-registration-draft.md`** (8 numbered items, drafted from both reviewer passes, nothing new added). **Item 7 expires first** — both six-set transactions must emit aligned PER-QUERY scores and qids, and that must be in force BEFORE M9's close-out runs or the row cannot be built at all, ever. Original: **Register it — as a whole-protocol RECIPE delta, not a coverage test.** M9→M10 changes dose 4.5× (3.69B → ≈16.8B), head width 384→1152, schedule, objective, mix, batch, and possibly the student. **Normalise on the TEACHER**, not M9 (`Δr_d = (S10−S9)/T_d`); never call `S10/S9` "retention". Register now: the confound list, a fixed non-causal claim sentence, the exact datasets/statistic/B/seed, a conditional "same backbone family" label (F can pick a MiniLM), and **that both six-set transactions emit aligned PER-QUERY scores and qids** — otherwise the paired row cannot be built at all |
| **W14** | **No decision-bearing surface sees the headline forms.** COV selects on forum/medical/legal/finance; C1b is clean-4 (scientific, biomedical). The screen can optimise the build away from the release bar and nothing would notice until M10.4. Codex frames the root question: **is family A a causal experiment, a catastrophe veto, or diagnostics? It cannot be all three** |

## NEXT, in order — rewritten 2026-09-08; three items on the old list were already DONE

**DONE, removed from this list** (they were still listed as pending and cost a re-read):
corpus→trainer path · M10.0-c baseline · generation (finished 09-05/06, `work/m10gen/*.jsonl`;
ANCHOR consumed 834,463 generated rows) · `data_cut.unique_text_count` = **2,651,572**, registered ·
**A8 gate 2 re-run 09-07 across all 12 forms** (`results/m10_a8_gate2.json`) · family F, and F1
resolved to **bge-small**.

| # | open item | blocks | who |
|---|---|---|---|
| 1 | **The W8 band-1 remainder is RUNNING** — 3 of 11 done (ANCHOR · A1 · A2, all `exit 0`, avg 2.39 h vs 2.71 h projected). Then the registered contrasts per family | the M10.2 recipe lock | in flight |
| 2 | **CUREv1 admission** (decision 12, adopted 2026-09-04, **never executed**). Its own precondition: the harvest, PAQ and seed draws were screened against an index lacking it — **re-screen or disclose before reading it** | nothing on the critical path; it is a reported diagnostic, never selection-bearing | session |
| 3 | **`results/m10_data_manifest.json`** — still MISSING (§1 owes it) | the recipe lock's provenance | session |
| 4 | **Own-source word-5-gram screen** on the harvest corpus (W11's second half): run it if a pass over the sources is cheap, else **disclose it as not run** in §1 (near-vacuous by construction) | a disclosure in the report | session |
| 5 | **Register the paired M9-vs-M10 row** — **draft ready for ratification**, `research/m10-paired-row-registration-draft.md`. Its only conditional item (§5, backbone family) self-resolved to *within one family* when F picked bge-small. **§7 expires first:** both six-set transactions must emit aligned per-query scores AND qids before M9's close-out runs, or the row is unbuildable forever | M9's close-out | **Dylan** |
| 6 | **W14** — is family A a causal experiment, a catastrophe veto, or diagnostics? *"It cannot be all three."* Note its first option NARROWED: a clean-4 surface can no longer be added to COV, which is now observed | the M10.2 recipe lock. Tier 3 | **Dylan** |
| 7 | **`torch.compile` for registered arms** (~1.7×). Currently SMOKE-ONLY | screen wall-clock only | **Dylan** |

## Screen design, settled

**W8 band 1** (P2 = 0.00288 ≤ 0.0056): run F · ANCHOR · A · G · B · E · D, **cut C** — `C-M9init`
at 5M starts from 3.69B tokens and wins its own contrast by construction. **L12 cut, A's STOP
removed** (2026-09-05). Both E arms run on the **A100**, together.

## Hazards a cold session will hit

- **`E-bs128` fails on this box** at max_len ≥ 256 (`CUDA driver error: device not ready`)
  on three morning runs, then PASSED at 512 on the evening re-run — intermittent, not deterministic. Passes at 128 (2,188 ex/s). **Accepted, not worked around** — do
  NOT add gradient accumulation. The 1,517 ex/s in the rate table is a random-token microbenchmark.
- **`arm_smoke` reaches ~8 GB RSS**; torch's CPU allocator does not release across 12 model loads.
  Run it alone. **Kill by PID** — `pkill -f arm_smoke` matches your own waiting shell.
- **Smokes overwrite real artifacts.** `calib.run_arm` writes the real `P0.json`; `harvest.draw`
  wrote the real `harvest_draw.json` until `out_dir` was added. Check `total_steps`/`_partial`.
- **The system python has no numpy.** Use `.venv/bin/python`.
- **`m10src/forms.RUBRIC` is the frozen gate standard; `forms.FORMS` is the revisable prompt.**
- Never overwrite `results/perquery.json`. No six/reserved/LoTTE read outside a registered
  transaction. Every review brief carries the reserved read-exclusion; **audit the log afterwards**.

## Do NOT redo

- Do not move the seed-precision bar (W6: there is no bar). Do not re-weight or drop BRIGHT
  (LEDGER §5 — **legal** is the uninformative family, 32.5% of variance for 12% of signal).
  Do not run T2-8's rungs 2/3 (demoted to diagnostics). Do not treat the on-form diagnostic as an
  admission instrument. There is **no post-generation admission test** — only A8's manifest gates
  and the FORMS-12 hold-out.

