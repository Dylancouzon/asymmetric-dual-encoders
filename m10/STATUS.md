# M10 status — 2026-09-05 evening. Data is BUILT, nothing registered has trained; the corpus→trainer path is being built by a worker.

**Read this, then `m10/LEDGER.md`.** The box is **preparation for the cloud GPU run, not a
measurement target** (Dylan, 2026-09-05): run as much as it can here first, no re-shaping; the
remainder moves to the A100 under the same registry. **The weekend timeline is not binding.**
Working model (Dylan, 2026-09-05): the session is the ML lead, Opus/Sonnet subagents do the build
work, Codex is the adversarial reviewer.

## Nothing is running. The next command is in "NEXT", below.

## What is DONE and verified

| step | outcome |
|---|---|
| **0a/0b/1/2/3** | vLLM + generator smoke; all seven generated forms approved; four COV families admitted (13,416 queries); §0a design lock |
| **4 harvest** | **A3 corpus = 1,250,000 rows** from 21,087,043 harvested (wiki 16.06M · arXiv 4.98M · pool 46.6K). title/keyword/claim at 417K/417K/416K, no form short. FORMS-12 hold-out applied by DOCUMENT across all forms: **1,500 docs held → 1,614 eval rows, 1,248,386 train rows** (`work/m10harvest/harvest_{train,forms12}.jsonl`) |
| **5 PAQ** | **A2 4,037,000** (`8f32bcdf…`) + **build 1,000,000** nested inside it, from `dl.fbaipublicfiles.com`, CC BY-SA 3.0 shipped in the tarball. Protected screen removed only 0.35% |
| **6 trainer** | `nano10` · `data10` · `trainer10` · `qfilter` · `corpus10` · `arm_smoke` · `screen_lock`. **121 tests green.** Arm-shape smoke **12/12 on CUDA at max_len 512** |
| **M10.0-e** | **COMPLETE.** P0 0.477528 · P1 0.476141 · P2 0.473892. **Same-init distance 0.00288**, seed effect 0.00139 (n=1). `results/m10_calib_report.json` |
| **seeds** | resolved for **all seven** generated forms; health/finance 33,000 each from `wikipedia-body`, howto 37,927, the other four route `"general"` |

## OPEN — needs Dylan (none blocks the next step; W12/W13 ruled 2026-09-05, kept one line each)

| # | what |
|---|---|
| ~~W12~~ | **RULED: STOP removed.** Family A reports three labels and its DEV-6 CQADupStack + FORMS-12 reads; C1b is the failure condition (`instructions-m10.md` §Amendment 2026-09-05 C1) |
| ~~W13~~ | **RULED: L12 CUT.** F = bge-small vs MiniLM-L6 at 20M, 12 contrasts, 14 trained arms (§Amendment 2026-09-05 C2) |
| **paired row** | **Register it — as a whole-protocol RECIPE delta, not a coverage test.** M9→M10 changes dose 4.5× (3.69B → ≈16.8B), head width 384→1152, schedule, objective, mix, batch, and possibly the student. **Normalise on the TEACHER**, not M9 (`Δr_d = (S10−S9)/T_d`); never call `S10/S9` "retention". Register now: the confound list, a fixed non-causal claim sentence, the exact datasets/statistic/B/seed, a conditional "same backbone family" label (F can pick a MiniLM), and **that both six-set transactions emit aligned PER-QUERY scores and qids** — otherwise the paired row cannot be built at all |
| **W14** | **No decision-bearing surface sees the headline forms.** COV selects on forum/medical/legal/finance; C1b is clean-4 (scientific, biomedical). The screen can optimise the build away from the release bar and nothing would notice until M10.4. Codex frames the root question: **is family A a causal experiment, a catastrophe veto, or diagnostics? It cannot be all three** |

## NEXT, in order

1. **The M10 corpus→trainer path — THE BLOCKER, and it does not exist.** `data10` loads only the
   M9 pool. Needed: a loader for harvested/PAQ/generated rows, the **form-balanced sampler**
   (a registered default, `instructions-m10.md`:480-485 — zero hits for `balanc` in `m10src/`),
   and teacher targets for **~6.3M new stella query encodes**. Nothing trains without this.
2. **A8 gate 2** — MS MARCO dev distribution overlap (mandate :467-475, "before any arm"). ~1 h.
   It is the one outside measurement of M9's failure mode the plan has.
3. **CUREv1 admission** (decision 12, adopted 2026-09-04, **never executed**). The harvest, PAQ and
   seed draws were screened against an index that lacks it — re-screen or disclose before reading.
4. **M10.0-c baseline** — DEV-6 read of the M9 candidate incl. `heldout-longq`. Minutes; it is the
   denominator every retention-vs-M9 comparison needs.
5. **15-minute diversity pilot**, then **generation** (~10 box-h). `health` reads **26.50%** on the
   amended A8 gate, **above the 25% cut**, and rising with n — its prompt needs fixing first, via
   decision 15's machinery (it has used 1 of 2 revisions).
6. **Register the paired row** (whole-protocol recipe delta, teacher-normalised; both six-set
   transactions must emit aligned per-query scores and qids) — before M9's close-out runs.
7. **Disclose the own-source 5-gram screen** as not run on the harvest corpus (W11 second half;
   near-vacuous by construction) in §1, or run it if a pass over the sources is cheap.
8. §0b's `data_cut`, then **family F** (F1 only: bge-small vs MiniLM-L6 at 20M).

## Screen design, settled

**W8 band 1** (P2 = 0.00288 ≤ 0.0056): run F · ANCHOR · A · G · B · E · D, **cut C** — `C-M9init`
at 5M starts from 3.69B tokens and wins its own contrast by construction. **L12 cut, A's STOP
removed** (2026-09-05). Both E arms run on the **A100**, together.

## Hazards a cold session will hit

- **`E-bs128` fails on this box** at max_len ≥ 256 (`CUDA driver error: device not ready`),
  reproducibly, on an idle card. Passes at 128 (2,188 ex/s). **Accepted, not worked around** — do
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

