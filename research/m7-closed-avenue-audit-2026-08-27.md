# Audit: every closed avenue vs the five conditions of "push for the best model, not a model"

Date: 2026-08-27. Read-only audit of branch `m7-query-encoder` against CLAUDE.md's standing
directive ("Standing directive: push for the best model, not a model", Dylan 2026-08-26).
Conditions, abbreviated in the table:

- **C1** arithmetic redone with the best available component, not the current one
- **C2** every failing component diagnosed mechanistically, not merely observed
- **C3** literature swept for the specific failure, with real numbers extracted
- **C4** capability claims checked algebraically before belief or dismissal
- **C5** negative result reported with what would change it

C-marked "n/a" where the condition has no purchase (a licence ruling has no failing component to
diagnose). The project's own precedent for this audit's standard is EXPLORED.md:19 — the
"Snowflake justify-max tier" row, closed on projected numbers that measurement inverted, kept as
a self-labelled warning.

## Table — one row per closed/killed/parked/demoted/withdrawn avenue

| # | avenue | recorded at | conditions SATISFIED (evidence) | conditions NOT satisfied | verdict |
|---|---|---|---|---|---|
| 1 | MIRACL-en as training source | m7/EXPLORED.md:7; m7src/trainmix.py docstring (lines 15–18) | C1: cost arithmetic committed — miracl-en encode 24.23 h fp32, 50.53 GB fp16 storage (results/m7_throughput.json) vs 2,863 queries = +0.8% of the 340,850-pair TRAIN (m7/LEDGER.md:46). C5: the blocker is named (no parquet mirror, removed loader) | — (C2/C3/C4 n/a: cost close, nothing failed) | **SOUND** |
| 2 | Climate-FEVER in UNTOUCHED-FINAL | m7/EXPLORED.md:8; m7/LEDGER.md:59–62 | Licence-class close under the affirmative-evidence standard applied uniformly (same rule that excluded Quora, LEDGER:62). CLAUDE.md reserves licence reopening to Dylan | C1–C5 n/a (commercial-reality class) | **SOUND** |
| 3 | fp32 teacher encodes on dev/train | m7/EXPLORED.md:9; m7/LEDGER.md:25–27 | C1: measured — cos 1.000000 on 10K docs, \|Δ nDCG\| ≤ 3e-4 on both dev components; fp32 kept where the frozen comparators require it | — | **SOUND** |
| 4 | Bare (unprefixed) teacher vectors as distillation target | m7/EXPLORED.md:10 | none directly — the close cites only the teacher's own +1.85 from the prefix (`work/devres/refs.json`, a gitignored file) | **C1**: no table was ever fitted against bare targets; the close is an inference from *teacher-side* quality to *table* quality — the exact inference class the project itself refuted (EXPLORED.md:24: Spearman(ceiling, table)=0.000). **C3** none recorded. **C5** absent. The queued p4 `prefix`/`prefix-init` arms (m7src/program.py:281–284, 309–317) vary runtime preproc and row conditioning, not the distillation target — none of the seven mandatory chains or the exploratory arm runs a bare-target B phase | **UNDER-DIAGNOSED** |
| 5 | A dev component validating long-query behaviour | m7/EXPLORED.md:11 (closed); m7/EXPLORED.md:46–50 (reopened, unactioned) | C1 partially: mix statistics measured (held-out p50=13 WP, p90=24; 55/7,325 reach 64). The ArguAna row is labelled an extrapolation | The project's own reopened note (EXPLORED:46–50) refutes the close — overlap@10/cosine need no qrels and are implemented (m7src/stage0_ridge.py:95); the mandate pre-authorises synthetic long queries — and **nothing was built**: the only long-query artifact remains heldout-longq, 55 queries, 54/55 HotpotQA multi-hop, and now known to be a strict subset of heldout-train (m7src/boot.py:183). ArguAna is 1 of the 6 final datasets, avg 193-word queries, the architecture's pre-identified worst case (CLAUDE.md M1 findings) | **PREMATURE** (as a standing "cannot test" claim) |
| 6 | Fusing with opensearch-doc-v3-gte | m7/EXPLORED.md:12 | C4-class logic: Tier 1 is *defined* as beating it — fusing it in is circular, airtight. Vendor rule is Dylan's commercial reality | — | **SOUND** |
| 7 | BM42 as sparse arm | m7/EXPLORED.md:13 | Structural: query-side attention violates the zero-query-compute premise; independent reproduction failures cited (Reimers, Bergum) | — | **SOUND** |
| 8 | Centering/whitening/top-PC/SIF/IDF as new capacity | m7/EXPLORED.md:14; results/m7_absorb_check.json | **C4 exemplary**: machine-precision absorbability proofs (max diff ≤ 1e-13), plus empirical confirmation that learned weights already became IDF-like. This row is the directive's own cited example | — | **SOUND** |
| 9 | Length scaling 1/sqrt\|T\| | m7/EXPLORED.md:15; results/m7_absorb_check.json | C4: removed by the final L2 normalize — algebraic no-op | — | **SOUND** |
| 10 | `fn_margin` as contrastive-collapse cause | m7/EXPLORED.md:16; results/m7_diag_scores.json | C2: measured — removes 0.18% overall, 4.3% of top-100 hardest at 0.02. The true cause (lr) was later isolated and confirmed (m7/RESULTS.md:73–77) | — | **SOUND** |
| 11 | "Random negatives trivially separable" as collapse cause | m7/EXPLORED.md:17; results/m7_diag_scores.json | C2: measured — 32.7 random negatives/query outscore the positive; 15.9% of queries ≥1. The earlier "confirmed" assertion was itself corrected in-repo (m7/RESULTS.md:28–37) | — | **SOUND** (a model self-correction) |
| 12 | Qwen3-Embedding-0.6B as teacher | m7/EXPLORED.md:18 (struck through, REOPENED) | The reopen note is honest: closed on symmetric MTEB, "the criterion this project refuted". Blocker named (151,669-vocab ridge needs an iterative solver) | Original close: **C1** never run on the adopted table criterion; the shortlist's "dominated" verdict (research/m7-teacher-shortlist-2026-08-26.md, disqualified table) uses the refuted MTEB ordering (EXPLORED.md:22). The blocker is softer than stated — see experiment 12 below | **PREMATURE** (original), correctly reopened, blocker soluble |
| 13 | Snowflake "justify-max" vendor tier moot | m7/EXPLORED.md:19 | Self-labelled **FALSIFIED**, kept as a warning. Not audited further — it *is* the audit standard | — | (project's own precedent) |
| 14 | granite-embedding-english-r2 as teacher | m7/EXPLORED.md:20 (struck through, REOPENED) | Reopen note honest, same refuted criterion | Original close: **C1** untested on the table criterion. The 50,368-vocab Gram is 50,368² × 4 B ≈ 10.2 GB fp32 — fits the 25 GB RAM budget chunked; "borderline" is not "blocked" | **PREMATURE** (original), reopened, cheap to test now |
| 15 | "A 2025–2026 teacher we had not seen — swept; nothing survives" | m7/EXPLORED.md:21; research/m7-teacher-shortlist-2026-08-26.md | C3: real sweep with real numbers (config.json-verified vocab, licences, MRL curves). Structural kills (vocab 128K–256K, non-commercial weights, never-released) are C4-clean arithmetic | **C1**: three shortlist *survivors* were never run through the adopted table criterion: **arctic-embed-m-v1.5** (30,522 vocab / 768d — the existing solver runs unchanged; it is LEAF's teacher and the best group-A system on the six, 0.5264 per CLAUDE.md; justify-max precedent already granted once), **gte-base-en-v1.5** (30,522/768), **gte-modernbert-base** (50,368/768). Their dismissal is "not a quality upgrade" on MTEB v1 — the ordering EXPLORED.md:22 declares "wrong on this evidence". Worse, the project's own within-family finding (m7/RESULTS.md:124–125: bge-base 0.686 > bge-large 0.613, e5-base > e5-large) predicts the untested *base* variants out-approximate their probed larger siblings — gte-large's table was the worst of eight (0.2033) while gte-base was never tried. **C2**: stella's advantage "remains unexplained" (EXPLORED.md:27), so no argument can substitute for measurement here | **UNDER-DIAGNOSED** |
| 16 | MTEB v1 as a teacher-ranking signal | m7/EXPLORED.md:22 | C1/C2: refuted by paired measurement (probe vs calibration) | Its consequence was not propagated back into the shortlist's quality-based disqualifications (feeds rows 12/14/15) | **SOUND** (the close itself) |
| 17 | Fixed-step objective-C sweep as the contrastive-lr test | m7/EXPLORED.md:23; m7src/program.py:113–131 | C2: diagnosed (B not converged at low lr — 0.2731 vs 0.4449 at 4k steps), redesigned before any A-phase result was read; Codex independently concurred | — | **SOUND** |
| 18 | Symmetric teacher probe as selection criterion | m7/EXPLORED.md:24; results/m7_learnability_report.json | C1/C2: Spearman 0.000 over eight candidates, arctic −0.0480 CI-resolved; n=8 caveat stated (m7/RESULTS.md:110–111) | — | **SOUND** |
| 19 | arctic-embed-l as teacher (withdrawal) | m7/EXPLORED.md:25; CLAUDE.md decision log | C1: measured on the adopted criterion, −0.0480 [−0.0608, −0.0349] CI-resolved. C5: what would change it is the criterion itself | mechanism gap belongs to row 21, not this withdrawal | **SOUND** |
| 20 | gte-large / e5-large / e5-base / bge-large as teachers | m7/EXPLORED.md:26 | C1: each CI-resolved below the incumbent's table (−0.032 to −0.104); e5 added deliberately to test a hypothesis | closed-form-only caveat acknowledged (m7/RESULTS.md:127–129; results/m7_ridge_vs_trained.json exists) | **SOUND** |
| 21 | Mean pooling as stella's approximability mechanism — and the derived "no attribute to search new candidates on" | m7/EXPLORED.md:27 | The refutation itself is C1/C2 model work: a controlled same-weights test (arctic-l-mean ratio 0.526→0.472) | The **derived search-stopper** fails **C3**: no literature sweep on token-linear approximability of encoder query spaces is recorded anywhere (`grep -ri approximab research/` returns nothing) — anisotropy/outlier-dimension/embedding-geometry literature is directly on point. Fails **C1**: cheap correlational arithmetic over the eight candidates' already-cached vectors (anisotropy, effective rank, in-sample ridge R² vs table ratio) was never done | **UNDER-DIAGNOSED** (the stopper, not the refutation) |
| 22 | Cosine agreement as a selection metric | m7/EXPLORED.md:28 | C1: measured mis-ranking (e5-large highest cosine, sixth on retrieval); demoted to diagnostic, not deleted | — | **SOUND** |
| 23 | Bigram rows, closed-form onto the trained winner | m7/EXPLORED.md:51; m7/LEDGER.md:468–481; results/m7_bigram_residual_k10000.json | C1: pre-registered bar, full suite, −0.0301 CI-resolved, baseline reproduces the gate macro exactly. C2: mechanism stated (teacher-ward correction undoes A-phase gains) with a λ-sweep behaving as the mechanism predicts. C4: absorb_check already proved bigram rows are genuine capacity. C5: escalation (joint retrain) explicitly left open with its own pre-registration requirement | **C2 "shown" is incomplete**: the λ-sweep's numbers (0.1/1/10, proxy-3) exist only as LEDGER prose (LEDGER:473) and the f6dcac6 commit message — no committed JSON holds them, violating the repo's own "never restate a number a results JSON doesn't hold" rule in the inverse direction. Also the cheapest escalation is not the joint retrain: A-only training of appended bigram rows from the frozen winner is queued as exploratory idea #2 (m7/STATUS.md:34) but unscheduled | **SOUND** (scoped close), with a provenance debt and an under-prioritized cheap escalation |
| 24 | doc2query expansion | m7/EXPLORED.md:32–39 (demoted), :52 (closed); m7/LEDGER.md:497–506; results/m7_doc2query_probe.json | C3: exemplary — Weller et al. regime analysis plus the observation that no cited source covers this architecture. C5: revival conditions enumerated (clean generator ruling, bigger N, doc re-encode). Pre-registered rule set before the number; the row honestly says "parked, not disproved" | **C1**: closed at the weakest form of the treatment — N=5 sampled queries/doc, T5-base, vs docTTTTTquery's published 40/doc — with a *positive-leaning* result (+0.0054, p=0.085, positive on both components). The dose–response was never measured, and a bigger-N run is still *diagnostic* (same MS MARCO generator), i.e. needs no licensing ruling | **SOUND per protocol**, but the escalation trigger was set below where the literature says the effect lives — run the dose test before spending Dylan's ruling |
| 25 | **Mined hard negatives — and the mandate's BM25-mined / teacher-mined / mixed comparison** | closure is *implicit*: m7/RESULTS.md:78–84 ("Mined hard negatives HURT"); m7src/program.py:164–165 ("the screen resolved that mined negatives HURT"), :170, :189, :241 (hard_neg_k=0 hard-coded into every later arm incl. ABLATION_A); m7/LEDGER.md:488 (lever-#2 arms mirror hard_neg_k=0). **No EXPLORED.md row exists** | C1 partially: one matched pair, CI-resolved (+0.0034 [0.0019, 0.0049] for random-only), honestly one-variable | **C1**: the pair is bge-era, lr 5e-5, k=16, teacher-mined, fn_margin 0.05, 2,000 steps, proxy-3. Never tested: **BM25-mined** (mandate-ordered, LEDGER:156–158; `phase2_negatives` with bm2516/mixed32/teacher32 arms exists at program.py:200–211 and **never ran** — zero p2n rows in RESULTS.md, which records every run), mined negatives **at the shipping lr 1e-3**, anything **under the stella teacher**, and the seven mandatory chains contain no negatives arm (program.py:277–289). **C2**: no mechanism for *why* mined hurt — only the fn-filter was exonerated (fn_masked_frac 0.0051, RESULTS.md:84). **C3**: none recorded, though false-negative contamination of mined negatives is the canonical published cause of exactly this symptom (RocketQA-style denoising), and the 0.05 margin in teacher score space may simply be the wrong filter. **C5**: absent — there is no recorded close at all, so no reversal condition | **PREMATURE** (and silently propagated into the shipping recipe) |
| 26 | K=20000 bigram probe rung | m7/RESULTS.md:172–173 | C1: refused on RAM arithmetic; moot after row 23's close | — | **SOUND** |
| 27 | Linear post-hoc projection into a frozen contextual doc space (M2-era) | CLAUDE.md "Rerun outcomes" | C1: oracle-λ selected on test — the strongest possible shot for the linear class; M7's trained-table program is the non-linear escalation actually pursued | — | **SOUND** (superseded by M7 itself) |

Not audited as closes (checked, still live): learned-weights/KL "buy nothing" (corrected to
undecided, m7/RESULTS.md:38–40, p4 `flat`/`uniform-w` arms queued); contrastive kill bar (may not
fire, results/m7_contrastive_verdict.json); lever #2 chain (adopted, pending the dependence
recompute); lever #4 count saturation (pre-registered, queued); FEVER in/out (phase5, queued);
untouched-final repair (reopened and actioned, LEDGER:117–126). Note in passing: `phase3_hparams`
(temp/n_neg sweeps, program.py:215–221) has also never run — not a recorded close, but temp=0.02
and n_neg=32768 have been fixed since phase 1 without a sweep.

## Settling experiments for every non-SOUND row

Costs anchored to: full dev-suite eval ≈ 45 min; training chain ≈ 20 min; RTX 3080 10 GB / 25 GB RAM.

**Ranked by potential to change the project's outcome:**

### 1. Row 25 — mined/BM25 negatives (PREMATURE). ~1–2 h.
Two A-only arms from the surviving stella B-checkpoint at the shipping recipe (lr 1e-3,
warmup_linear, steps_a per SURVIVOR_STEPS_A): `hard_neg_k=16 hard_neg_source=teacher` and
`hard_neg_k=16 hard_neg_source=bm25`, judged on the proxy against the recorded baseline arm.
2 × ~20 min training + proxy evals ≈ 1 h; promote the winner (if any) to one full-suite compare
(+45 min). This is the mandate's own ordered comparison, deferred indefinitely on one bge-era
pair at a different lr. Pre-register the bar first (same signflip+CI form as the levers). If
mined still hurts at 1e-3 under stella, also log the mechanism check: score the k=16 mined set
against qrels to measure actual false-negative rate (CPU, minutes) — that converts "observed"
into "diagnosed" and closes the row properly in EXPLORED.md.

### 2. Row 15 — untested teacher survivors (UNDER-DIAGNOSED). ~30–45 min per candidate.
Run the existing learnability probe (closed-form table on TRAIN query vectors, scored on the two
CQADupStack dev components, paired vs incumbent — exactly `m7_learnability_report.json`'s recipe)
on **arctic-embed-m-v1.5** and **gte-base-en-v1.5** — both 30,522-vocab/768d, so the current
solver and CLS_ID work unchanged; encode throughput for a 109–137M model ≈ bge-base's 891 texts/s
fp16 → ~630K texts ≈ 12 min + solve. The project's own within-family dim finding predicts these
beat their probed larger siblings' tables, and the criterion that excluded them is the one the
project refuted. If either lands near stella's 0.3439, that is a second data point on the
unexplained axis — feeding row 21 — and a possible teacher upgrade for the price of an hour.

### 3. Row 5 — long-query blind spot (PREMATURE). <30 min GPU, zero qrels.
Doc-as-query teacher-agreement probe: sample ~2K pool documents of 100–250 WordPiece tokens,
treat each as a query, compute table-vs-teacher overlap@10 and cosine against the pool
(evalkit tiling; the machinery exists — stage0_ridge.py:95 implements overlap@10). Plot
degradation vs query length against the same curve for short queries. This does not need
synthetic counter-arguments or any six-set access, and it turns the ArguAna row of the final
report from an unmeasured extrapolation into a measured length-degradation curve. (The full
mandate-authorised version — synthetic long argumentative queries — needs a generator and can
stay optional.)

### 4. Row 4 — bare vs prefixed distillation target (UNDER-DIAGNOSED). ~1–1.5 h.
Closed-form only, no training: encode the 561K TRAIN queries with stella *without* its query
prompt (~40 min at ~250 texts/s for a 435M model), ridge-solve at the committed λ=0.01, score
proxy-3 against the same doc vectors, paired vs the committed prefixed-target ridge
(results/m7_stage0_ridge_stella.json, 0.4973). If bare targets win or tie, the finding matters
for the B phase of the shipping recipe; if prefixed wins, the EXPLORED row gains the C1 evidence
it currently lacks. Either way the close stops resting on the ceiling→table inference the project
refuted.

### 5. Row 21 — stella's unexplained approximability (UNDER-DIAGNOSED). <1 h CPU + a zero-GPU literature sweep.
(a) Pure arithmetic over the eight candidates' already-cached probe encodes: per candidate,
compute doc/query-space anisotropy (mean pairwise cosine), effective rank, and the ridge's
in-sample R², then Spearman each against the table/ceiling ratio. n=8 is weak but it is free and
either yields a searchable attribute or kills three hypotheses at once. (b) A parallel literature
sweep on token-linear approximability / embedding-space anisotropy with numbers extracted —
condition 3 verbatim, twice-proven cheap in this repo, and currently absent for this specific
question. An attribute here converts row 15's brute-force probing into directed search.

### 6. Row 24 — doc2query dose–response (flagged). ~4 h GPU, still diagnostic.
Rerun `doc2query_probe.py` at N=20/doc on the same two components with the same MS MARCO
generator (generation scales linearly: 2,534 s at N=5 → ~10,100 s ≈ 2.8 h, + re-encode + paired
eval). The current close binds at the cheap-test price with a positive-leaning point estimate at
1/8 the published dose; N=20 settles whether the mechanism scales *before* asking Dylan for a
clean-generator licensing ruling — the same order of operations the doc2query row itself says
matters.

### 7. Rows 12/14 — Qwen3-0.6B and granite-r2 reopened rows (blocked-but-soluble). ~2 h (granite) / ~1 day (Qwen3).
granite: the 50,368² fp32 Gram is ~10.2 GB — chunk it in RAM (the repo already chunks bigger
things) and the probe runs today. Qwen3: restrict rows to tokens actually observed in TRAIN
queries (the incumbent table itself only trains 27,314/30,522 rows; unseen tokens already fall to
`apply_unseen_policy`) — that collapses the solve to an observed-vocab Gram and dodges the
iterative solver entirely; add MRL-256 truncation for the size arithmetic. Worth doing only after
experiment 2's cheap candidates are exhausted.

### 8. Row 23 — commit the λ-sweep numbers (provenance, minutes).
The lever-#1 diagnosis's λ-sweep (0.1/1/10, proxy-3) lives only in LEDGER prose and a commit
message. Re-emit it into `results/m7_bigram_residual_k10000.json` (or a sibling) from the saved
fit npz — the repo's own rule is that numbers live in results JSONs. Separately: promote the
queued A-only bigram-rows arm (STATUS.md:34) above other exploratory work — per
`m7_absorb_check.json`, n-gram rows and multiplicity pooling are the *only* genuine capacity
levers, lever #4 covers the second, and the closed-form failure explicitly does not test the
trained form.

## Summary judgment

The project's algebra-first and pre-registration culture is real and mostly honored: 17 of 26
audited closes are SOUND, several exemplary (rows 8, 17, 23's scoping, 24's labeling). The
directive violations cluster in two patterns:

1. **Closes that rest on the criterion the project itself refuted.** The MTEB/ceiling→table
   inference was disproven (Spearman 0.000), and EXPLORED reopened two rows accordingly — but the
   same refuted inference still silently underwrites row 4 (bare targets), row 15's untested
   survivors, and the "nothing survives" summary. The refutation was applied to the rows a
   reviewer named, not propagated to everything built on the criterion.
2. **Closes that were never recorded as closes.** The most consequential one (row 25, mined/BM25
   negatives) has no EXPLORED row, no mechanism, no literature check, no reversal condition — it
   lives in code comments and hard-coded `hard_neg_k=0`, was decided on one bge-era pair at a
   non-shipping lr, and the mandate-ordered BM25/mixed comparison it displaced never ran. An
   unrecorded close is exempt from every gate the project built, which is exactly why it is the
   top finding.
