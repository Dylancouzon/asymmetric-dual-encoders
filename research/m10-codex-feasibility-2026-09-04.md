# Codex adversarial review of the M10 plan after the feasibility review — 2026-09-04 (gpt-5.6-sol, read-only, high effort)

Brief: break amendments B1–B6 and decisions 11–13 (`instructions-m10.md` §Amendment 2026-09-04b).
**Read-exclusion audit: CLEAN** — the reviewer opened only the named files plus `results/perquery.json`
(tracked comparator rows, not reserved) to re-check the bootstrap quantile; both greps were
file-scoped; no `work/`, `untouched-*`, reserved or LoTTE path. Full log: the gitignored `.log`
beside this file. Verdict: "not lock-ready" — 4 BLOCKER / 8 MAJOR / 4 MINOR / 1 missed. **All actioned.**

## Dispositions

| # | finding (short) | disposition |
|---|---|---|
| B1 | pass points used M9's 0.0125 quantile; gatekeeping tests at 0.025 | **adopted** — `scripts/clean4_bars.py` q=0.025, JSONs regenerated (proxies 89.3/91.3/91.6/94.9%), renamed *planning proxies*; §Final run names `lower_q025_raw` and puts the sign-flip inside the sequence |
| B2 | G-MLP returns per-token outputs; the inherited trainer expects pooled | **adopted** — §Recipe: training wrapper pools after the head, export wrapper emits tokens, wrapper-parity test before the arm; STATUS item 7 |
| B3 | family F rule not executable (L12 at 5M cannot win a 20M rule; 3 arms need 2 comparisons) | **adopted** — L12 is a 5M elimination probe extended to 20M iff within the MDE of the better 5M reading; F carries two contrasts; **fourteen contrasts, 0.025/14 fixed** whether or not the second runs |
| B4 | decision 11 default (ship on C1a pass, C1b fail) is motivated — 73% of that margin is fiqa | **adopted as the default** — release needs C1b; a C1a-only pass is published as a frontier measurement labelled not recommended. Dylan may loosen (decision 11) |
| M1 | uniform-retention lens cannot order C1b vs C2a | **adopted** — "a lens, not an ordering"; per-dataset stress scenarios added to the JSON (one set at 65%, rest 94%) |
| M2 | "out of reach" exceeds the evidence; EmbedDistill 95–97% at ~10× | **adopted** — "low-prior stretch aim", search criteria stated, the screen result that would raise the prior named |
| M3 | 1152→512→1024 caps output rank at 512; warm start underspecified | **adopted** — residual form `W_lin·x + W₂·GELU(W₁·x+b₁)`, W₁ 1152→192, 34.96M, parity re-run and passed; exact two-solve warm start specified (per-token PCA for W₁, centred, sign-fixed; ridge from pooled GELU features to the residual) |
| M4 | `arxiv-title` as an equal-weight family is circular with A3 and dominates power; leakage via versions | **adopted** — demoted to a registered *secondary* surface with one action (harvested scientific forms in/out of the build); drawn by id-without-version, all versions excluded, harvest screened against it |
| M5 | B4 surface built at M10.1 but needed at M10.0-d | **adopted** — drawn and protected at M10.0-d everywhere |
| M6 | CUREv1 is not the CQADupStack precedent; Qwen-annotated pools | **adopted** — decision 12 now asks only for a reported diagnostic; selection-bearing use recommended against |
| M7 | MedlinePlus/CDC harvest lets MedicalQA reward A3; ClinicalTrials.gov conditional is needless | **adopted** — both out of M10 (`m10/EXPLORED.md`); consumer-health stays generated |
| M8 | `MDE = max(0.0056, distance)` + remedy is an adaptive α | **adopted** — A4 struck; MDE 0.0056 and 0.025/14 fixed; the resolution number is a power disclosure; LEDGER is an ordinary admission; underpowered contrasts reported unresolved |
| M9 | superseded prose contradicts the amendments in five places | **adopted** — "head stays linear", "nonlinear head out of scope", "Holm sequence", EXPLORED's LEAF-Appendix-B claim, COV_CANDIDATES' "no scientific surface" all rewritten |
| m1 | 70.8 MB is a projection; MB vs MiB; MLP latency | **adopted** — labelled projection, 10⁶ bytes, target not cap, measured at export with latency |
| m2 | "not re-proposable" too absolute (PMC-OA records carry article licences) | **adopted** — reopens with per-record licence provenance and the contamination rule resolved |
| m3 | open-decision states inconsistent across files | **adopted** — one states line above the owner-decisions table; STATUS points there |
| m4 | cutting G-768 prevents a trained width curve | **adopted** — the paper reports 384 vs 1152 vs 1536 contrasts, not a curve |
| missed | the headline has no untouched confirmatory surface | **adopted as wording** — §Unimpeachable item 8: no general retrieval-quality language; the reserved four are the only untouched surface and are descriptive. No new surface added (over-engineering) |

## Verbatim findings

## Verdict

The plan is not lock-ready. First repair the contradictory final-statistics contract and the G-MLP training path; then reject decision 11’s default unless “release” is explicitly decoupled from scientific success. B4 cannot remain an equal-weight selection family without leakage controls and an independent-family sensitivity analysis.

## BLOCKER

- [instructions-m10.md:175–202](/home/dylan/asymetric-dual-encoders/instructions-m10.md:175), [scripts/clean4_bars.py:33–81](/home/dylan/asymetric-dual-encoders/scripts/clean4_bars.py:33), [m9/FINAL_LOCK.md:80–93](/home/dylan/asymetric-dual-encoders/m9/FINAL_LOCK.md:80) — “full one-sided 0.025” / script default `q=0.0125` — The advertised 89.5/91.5/91.9/95.3% “pass points” use M9’s 1.25% quantile, not B2’s full 2.5%. An independent 20,000-draw check gives proxy widths about 0.0089/0.0121 rather than 0.0101/0.0139, moving the four ratios to roughly 89.3/91.3/91.6/94.9%. Worse, no uniform retention deterministically “passes”: the actual interval width depends on nano’s per-query differences. — Pick one inferential procedure, encode it once in the registry, regenerate the JSON, and rename these values “comparator-pair planning proxies,” never pass points.

- [m10src/head_mlp_parity.py:32–46](/home/dylan/asymetric-dual-encoders/m10src/head_mlp_parity.py:32), [m9src/longrun.py:775–780](/home/dylan/asymetric-dual-encoders/m9src/longrun.py:775) — “return self.head(tok)” / `v = F.normalize(v.float(), dim=-1)` — G-MLP returns `[batch, tokens, 1024]`; the inherited trainer expects `[batch, 1024]`. It will either fail against the target tensor or regress every token separately, neither matching the parity reference. — Implement separate training and export wrappers: training must masked-mean the token outputs and then normalize; export must expose the token outputs for FastEmbed pooling. Add a parity test between those wrappers before G-MLP is admissible.

- [instructions-m10.md:514–522](/home/dylan/asymetric-dual-encoders/instructions-m10.md:514), [m10/LEDGER.md:9–16](/home/dylan/asymetric-dual-encoders/m10/LEDGER.md:9) — “L12 at 5M” / “best COV macro at 20M wins” / “1” contrast — Family F cannot execute its own rule: L12 has no 20M result, and three competing students require two inferential comparisons, while the row declares one. The claimed thirteen-contrast correction is therefore unreproducible. — Either run all three to 20M with two registered comparisons, or make L12 a 5M elimination probe that cannot win until extended. Recount arms and multiplicity afterward.

- [instructions-m10.md:214–222](/home/dylan/asymetric-dual-encoders/instructions-m10.md:214), [results/m10_conjunct_arithmetic.json:75–91](/home/dylan/asymetric-dual-encoders/results/m10_conjunct_arithmetic.json:75) — “C1a passes, C1b fails: release” — Decision 11 would ship a model that failed the paper’s contamination-controlled headline partition, while the plan’s own scenario attributes 72.8% of its avg-6 margin to contaminated FiQA. Calling C1a a “release bar” then releasing on precisely the partition known to confer that advantage is motivated. Zero’s earlier release-after-miss proves that publication is independent of statistical success; it does not justify relabeling failure as a pass. — Release checkpoints for reproducibility regardless, but reserve “recommended/shipping model” for C1b. If product release truly requires only C1a, stop calling clean-4 the headline criterion.

## MAJOR

- [results/m10_conjunct_arithmetic.json:52–128](/home/dylan/asymetric-dual-encoders/results/m10_conjunct_arithmetic.json:52), [m9/FINDINGS.md:20–31](/home/dylan/asymetric-dual-encoders/m9/FINDINGS.md:20) — “uniform retention” — The arithmetic is internally correct for a uniform multiplier, but the lens contradicts the central finding that retention is distribution-specific. It cannot establish that C1b is harder than C2a: heterogeneous retention, unequal ceilings, and different bootstrap covariance can reverse that ordering. — Replace the ordinal claim with per-dataset stress scenarios, including M9-like covered/uncovered splits and leave-one-dataset-out pass calculations.

- [research/m10-feasibility-review-2026-09-04.md:33–55](/home/dylan/asymetric-dual-encoders/research/m10-feasibility-review-2026-09-04.md:33), [CLAUDE.md:40–67](/home/dylan/asymetric-dual-encoders/CLAUDE.md:40) — “no published precedent reaches it” / “C2b … out of reach” — The review’s own table lists EmbedDistill at a 10× gap and approximately 95–97%; excluding it because its loss also uses scores and labels makes the universal claim definition-dependent. The seven heterogeneous systems are not a systematic bound, and G-MLP/D-COV have not run. “Out of reach” violates the standing directive’s evidentiary threshold. — Say “low-prior stretch aim; no directly comparable pure-regression result found in this bounded search,” publish inclusion criteria, and name the screen result that would change the prior.

- [m10src/head_mlp_parity.py:55–56](/home/dylan/asymetric-dual-encoders/m10src/head_mlp_parity.py:55), [instructions-m10.md:518](/home/dylan/asymetric-dual-encoders/instructions-m10.md:518) — `1152→512→1024` — The parity result and 34,475,648 parameter count are correct, but the final linear map confines every pooled output to an affine subspace of rank at most 512. The arm advertised as attacking M9’s width ceiling discards the 1152→1024 linear arm’s potential rank. Its PCA warm start is also underspecified: centering, `b1`, PCA sign convention, and whether ridge fits `GELU(W1·mean(x))` or the correct `mean(GELU(W1x_t))` are absent. — Use a full-rank nonlinear or residual head under the cap, and lock a deterministic token-level warm-fit algorithm. Otherwise describe G-MLP as a nonlinearity-versus-rank trade, not added capacity.

- [instructions-m10.md:579–617](/home/dylan/asymetric-dual-encoders/instructions-m10.md:579) — “shares its form and source with arm A3 … by design” — Disclosure does not cure circular selection. A3 is rewarded on a held-out task constructed from the same source and exact form it adds; title→own abstract among random negatives is likely lexical and easy. Seed-0 document exclusion does not block arXiv versions, paper-family duplicates, or titles copied into other papers’ citations. At equal family weights, this low-variance 2,000-query toy gets 20% of a five-family macro and can artificially shrink the resolution estimate. — Make it diagnostic, not selection-bearing. If retained, split by canonical paper family and time, remove citation-title leakage, and require conclusions to survive removing the arXiv family from the macro.

- [instructions-m10.md:246–260](/home/dylan/asymetric-dual-encoders/instructions-m10.md:246), [instructions-m10.md:582–586](/home/dylan/asymetric-dual-encoders/instructions-m10.md:582) — “COV resolution … before (e)” / documents drawn “at M10.1” — B4’s surface is created after the stage in which it must already be admitted, encoded, and used to size the screen. STATUS silently uses the opposite order. — Move the draw and protection transaction explicitly into M10.0-d everywhere.

- [research/m7-data-licensing.md:45–55](/home/dylan/asymetric-dual-encoders/research/m7-data-licensing.md:45), [research/m7-data-licensing.md:74–85](/home/dylan/asymetric-dual-encoders/research/m7-data-licensing.md:74), [research/m10-feasibility-review-2026-09-04.md:112–128](/home/dylan/asymetric-dual-encoders/research/m10-feasibility-review-2026-09-04.md:112) — “Excluded for contamination … OUT of training AND validation” / admit CUREv1 — Decision 12 is not analogous to using different CQADupStack splits. It explicitly overturns a source-family contamination rule because fingerprinting cannot detect all provenance overlap. Qwen-annotated candidate pools add retriever- and judge-dependent label bias to a model-selection surface. — Reject CURE as selection-bearing. Permit it only as a reported diagnostic unless a human audit establishes agreement and results are robust to clinician-only qrels and removal of every PubMed-family component.

- [instructions-m10.md:328–335](/home/dylan/asymetric-dual-encoders/instructions-m10.md:328), [m10/COV_CANDIDATES.md:12](/home/dylan/asymetric-dual-encoders/m10/COV_CANDIDATES.md:12) — “MedlinePlus and CDC are MedQuAD sources” — Fingerprint screening removes copied strings, not same-source distribution advantage. MedicalQA will preferentially reward the A3 consumer-health harvest, contaminating A3−A2 as a general coverage decision. — Exclude MedicalQA from the primary A3/A4 gate or require a predeclared macro both with and without it. Remove ClinicalTrials.gov entirely until its affirmative clause is recorded; a speculative conditional source is needless machinery.

- [instructions-m10.md:590–626](/home/dylan/asymetric-dual-encoders/instructions-m10.md:590) — “MDE = max(0.0056, that distance)” — A CI width from one unrelated comparator pair is not an intrinsic resolution of the surface. Adding LEDGER after observing that width, then abandoning correction for unadjusted α=.05, is an adaptive maze; two-seed confirmation does not control multiplicity or winner’s curse. — Keep the substantive MDE at 0.0056, estimate power per contrast from blinded variance assumptions, enlarge the surface if possible, and report underpowered contrasts as unresolved without changing α.

- [instructions-m10.md:237–240](/home/dylan/asymetric-dual-encoders/instructions-m10.md:237), [instructions-m10.md:461–470](/home/dylan/asymetric-dual-encoders/instructions-m10.md:461), [instructions-m10.md:760–765](/home/dylan/asymetric-dual-encoders/instructions-m10.md:760), [m10/LEDGER.md:61–64](/home/dylan/asymetric-dual-encoders/m10/LEDGER.md:61), [m10/EXPLORED.md:21](/home/dylan/asymetric-dual-encoders/m10/EXPLORED.md:21) — “head stays linear” / “nonlinear head … out of scope” / “Holm” — B3 and B2 contradict three live instructions; EXPLORED also retains the withdrawn claim that LEAF found auxiliary regression terms ineffective. COV_CANDIDATES still says no scientific surface exists. These are operationally dangerous, not cosmetic. — Delete superseded prose rather than relying on “authoritative” amendment layering; leave one executable specification.

## MINOR

- [results/m10_head_mlp_parity_box.json:12–16](/home/dylan/asymetric-dual-encoders/results/m10_head_mlp_parity_box.json:12), [instructions-m10.md:425–428](/home/dylan/asymetric-dual-encoders/instructions-m10.md:425) — “≈70.8 MB fp16” — B6 records a projection, not a packaged fp16 asset; the only measured graph is 137.5 MB fp32. The plan also ignores the MLP’s per-token latency. — Measure final ONNX plus tokenizer bytes and CPU latency, define MB versus MiB, and state whether 70 MB is a target or cap.

- [research/m10-feasibility-review-2026-09-04.md:89–96](/home/dylan/asymetric-dual-encoders/research/m10-feasibility-review-2026-09-04.md:89) — “not re-proposable without a licence change” — Withdrawal of an unfiltered PubMed harvest is correct: NLM’s disclaimer is not a grant. The absolute wording is false because individual PMC/open-access records can carry affirmative article-level licences. — Say the proposed bulk artifact fails the evidence rule; allow reopening only with per-record licence provenance and the separate contamination rule resolved.

- [m10/STATUS.md:55–68](/home/dylan/asymetric-dual-encoders/m10/STATUS.md:55), [instructions-m10.md:132–152](/home/dylan/asymetric-dual-encoders/instructions-m10.md:132) — “Open decisions” — The files disagree on whether the open set is 11/12/A7, 11–13, or seven older defaults; decision 13 is simultaneously “adopted” and awaiting an owner ruling. — Publish one pending-decision table with explicit states: proposed, default-active, adopted, or blocking.

- [m10/PLANNING.md:348–364](/home/dylan/asymetric-dual-encoders/m10/PLANNING.md:348) — “384 → 768 → 1152” — Replacing trained G-768 does not erase the frozen probe, but it prevents a trained monotonic-width claim. — Keep G-768 cut; restrict the paper to the registered 384-versus-1152 contrast rather than implying a trained width curve.

## What the review missed

[m9/FINAL_LOCK.md:109–127](/home/dylan/asymetric-dual-encoders/m9/FINAL_LOCK.md:109), [instructions-m10.md:741–748](/home/dylan/asymetric-dual-encoders/instructions-m10.md:741) — “six development-informed datasets” — The proposed headline has no untouched confirmatory benchmark. M10’s data taxonomy is explicitly designed around clean-4 identities; COV selects the recipe; LoTTE can veto it; the reserved batch is descriptive. A one-time final read prevents numerical tuning but does not undo benchmark-form tuning. — Either reserve a licensed external surface now for a genuinely confirmatory claim, or restrict the paper’s conclusion to these development-informed datasets and prohibit general retrieval-quality language.

## Files opened

- `CLAUDE.md`
- `instructions-m10.md`
- `research/m10-feasibility-review-2026-09-04.md`
- `results/m10_conjunct_arithmetic.json`
- `scripts/m10_conjunct_arithmetic.py`
- `results/m10_bars.json`
- `scripts/clean4_bars.py`
- `results/m10_head_mlp_parity_box.json`
- `m10src/head_mlp_parity.py`
- `m10/STATUS.md`
- `m10/LEDGER.md`
- `m10/EXPLORED.md`
- `m10/RESULTS.md`
- `m10/COV_CANDIDATES.md`
- `m10/PLANNING.md`
- `m9/FINDINGS.md`
- `m9/FINAL_LOCK.md`
- `research/m7-data-licensing.md`
- `m9src/nano.py`
- `m9src/longrun.py`
- `results/perquery.json` — additional file used only to check the bootstrap-quantile mismatch.
tokens used
