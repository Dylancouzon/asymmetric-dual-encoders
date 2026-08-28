# M7 avenues explored and closed

Check this before starting anything new. One row per avenue; detail lives in the cited artifact.
A row here is a *claim that further work is not worth it*, and CLAUDE.md's standing directive sets
the bar it must clear. **Audited against that bar on 2026-08-27** (full table:
`git show` the scratchpad audit summarised in the reopened section below).

| avenue | why killed | evidence |
|---|---|---|
| MIRACL (English) as a training source | 2,863 train queries need the 32.9M-passage corpus (no parquet mirror). Cost/benefit fails by 2 orders; Wikipedia coverage already comes from HotpotQA/FEVER/SQuAD/NQ-open. | `trainmix.py` docstring |
| Climate-FEVER in UNTOUCHED-FINAL | No affirmative licence at any primary source; a wrapper tag is not evidence. | `LEDGER.md` |
| fp32 teacher encodes on dev/train corpora | fp16 is 2.4x faster and indistinguishable (cos 1.000000, \|Δ nDCG\| ≤ 3e-4). fp32 kept for the six + untouched-final. | `m7_throughput.json` |
| fusing with `opensearch-neural-sparse-doc-v3-gte` | Vendor rule **and** circularity — Tier 1 is *defined* as beating it. | `instructions-m7.md` |
| BM42 as a sparse arm | Query-side attention breaks the premise; independent reproductions failed to beat BM25. | Qdrant BM42 article |
| **query-side centering / whitening / top-PC removal / SIF or IDF weighting as NEW CAPACITY** | **Refuted algebraically to machine precision** — all absorbable into the table, so they cannot raise the ceiling, only act as a prior. Confirmed empirically: the learned weights already *are* IDF-like. | `m7_absorb_check.json` |
| length scaling (1/sqrt\|T\|) | No-op: any positive scalar function of \|T\| is removed by the final L2 normalize. | same |
| `fn_margin` as the contrastive-collapse cause | Refuted by measurement: at 0.02 it removes 0.18% of negatives, 4.3% of the top-100 hardest. | `m7_diag_scores.json` |
| "random negatives are trivially separable" as the collapse cause | Refuted by measurement: 32.7 random negatives/query outscore the positive on average. | same |
| a fixed-step objective-C sweep as the contrastive-lr test | Cannot isolate the lr — B runs at the same lr and is far from converged at the low end, so arms enter the A phase from different tables. | `LEDGER.md` |
| **the symmetric teacher probe as the selection criterion** | **Refuted by measurement**: Spearman(ceiling, distilled table) = 0.000 over eight candidates. Select teachers on the table, never on the tower. | `m7_learnability_report.json` |
| arctic-embed-l as teacher | Chosen on the ceiling, **withdrawn on the table**: −0.0480 [−0.0608, −0.0349] vs the incumbent. | same |
| gte-large, e5-large-v2, e5-base-v2, bge-large as teacher | All CI-resolved BELOW the incumbent's table (−0.104 to −0.032). e5 was added to test mean pooling; both sit below CLS bge-base. | same |
| MTEB v1 Retrieval as a *ranking* signal between teachers | Not merely imprecise, **wrong on this evidence**. Shortlist filter only, never an ordering. | `m7_teacher_probe.json` vs `m7_calibration.json` |
| cosine agreement to the teacher's query vector as a selection metric | Rises with lambda while nDCG falls, and mis-ranks (highest cosine → sixth of eight on retrieval). Diagnostic only. | same |
| bigram rows, closed-form onto the trained winner | −0.0301 resolved on the full suite; a λ-sweep shows it is structural (teacher-ward correction undoes the A-phase gains), not under-regularized. Joint-retrain escalation stays open with its own pre-registration. | `m7_bigram_residual_k10000.json` |
| doc2query expansion | Closed per the pre-registered rule: +0.0054 [−0.0007, +0.0114] p=0.085 at N=5/doc — positive-leaning but unresolved **at the cheap-test price**, which is 1/8 the published dose. Parked, not disproved. | `m7_doc2query_probe.json` |
| count saturation: `binary` and `cap2` | +0.0030 and +0.0016, neither clears Holm within its precision's three-arm family on `p35w-2m-s2500`. `sqrt` did and was adopted **there**; on the negatives candidate NO arm clears, so the whole family is closed and `pool_mode` stays `mean`. | `m7_lever4_pooling_p35w-2m-s2500.json`, `m7_lever4_pooling_p4n-teacher16-a.json` |
| the unseen-row policy (`init` / `mean_of_trained` / `zero`) as a decision worth making | **Measured, not assumed.** Only 1,743 of 30,522 rows (5.71%) are never trained by either phase; 994 are `[unusedN]` placeholders the tokenizer cannot emit and the 749 reachable ones are `##`-punctuation continuations and non-Latin characters. Their median bag contribution is **0.143x** a trained row's — an untrained token is nearly ignored, which is the benign direction. `apply_unseen_policy` stays uncalled, now as a checked non-choice rather than an oversight. | `m7_cold_rows_p4n-teacher16-a.json` |
| capacity lever #5, update-count row shrinkage | No `tau` adopted: nothing cleared the pre-registered bar. | `m7_lever5_shrinkage.json` |
| capacity lever #6, training through the `sqrt` pooling rule | Arm (a) +0.0011, p=0.051 fp16 / 0.073 int8, CI straddling zero — fails its own bar, so arm (b) never ran, per its own pre-registered falsifier. (An earlier version of this row added "moot besides: `sqrt` is no longer the served rule" — that was written while `p4n-teacher16-a` was the candidate; the negatives revert put `p35w-2m-s2500` back, where `sqrt` **is** the served rule. The lever still fails on its own bar.) | `m7_compare_full_lever6.json` |
| capacity lever #7, long-span distillation | Closed **without training the arm**: its own pre-registration made the length probe the pre-condition, and the corrected probe (nested prefixes of the same documents, so length is the only thing that varies) is **flat from 16 to 256 words**. The first probe's "length effect" was a document-population confound. | `LEDGER.md` lever #7 |

| **mined hard negatives (teacher / BM25 / mixed)** — the mandate's ablation, now actually run | **CLOSED 2026-08-28.** Under the step-selection correction none of the three arms clears the bar (+0.0023 p=0.107 · −0.0007 · −0.0056 resolved LOSS), the out-of-domain subset spans 0.3658–0.3688 across every arm *including the baseline*, and the mechanism is diagnosed: the apparent +0.0072 is `heldout-train` +0.0297 and `hotpotqa` +0.0187, i.e. a seen-document slice and a component whose train split is a TRAIN source, while `heldout-longq` gets worse for every arm. Sharpens memorisation, does nothing out of domain. | `m7_negatives_decision.json`, `m7_compare_full_steprule.json` |
| the pre-registered false-negative-rate check on the mined set | **VACUOUS, not run.** `mine_hard_negatives` takes the query's positives as `exclude`, so the rate against known qrels is 0 by construction. The real hazard is *unlabelled* positives, which qrels cannot reveal. Recorded because a pre-registered check that is a no-op is itself a finding. | `train.mine_hard_negatives` docstring |
| **per-arm proxy step selection**, as an instrument | **Failed on its own evidence.** The proxy peak did not reproduce on re-run (0.5130 → 0.5126), and the proxy ranked three arms exactly backwards from the full suite. Amended for future decisions to "match the baseline's step count"; explicitly NOT retroactive. | `LEDGER.md` step-rule section |

## Reopened or under-diagnosed — do NOT treat as closed

Findings of the 2026-08-27 audit against the standing directive (17 of 26 closes SOUND, 4
under-diagnosed, 4 premature). Two systemic patterns: a refuted criterion was un-applied only to
the rows a reviewer named rather than to everything built on it, and the single most consequential
closure was never written down as one, which exempted it from every gate this project built.
**Struck-through items were actioned on 2026-08-28 and are now closed on evidence.**

- ~~**Mined / BM25 hard negatives — PREMATURE, and never recorded as a close at all.**~~
  **ACTIONED and now properly closed.** `hard_neg_k=0` had been hard-coded into every arm since the
  phase-2 screen on **one bge-era pair at lr 5e-5**, with no mechanism established.
  `program.phase4_negatives` ran the mandated teacher/BM25/mixed comparison from the candidate's own
  B checkpoint at its own A recipe; see the table row above and `m7_negatives_decision.json`.
- ~~**Three teacher-sweep survivors were never run through the adopted criterion.**~~
  **ACTIONED for two of the three; the third is closed on arithmetic.** `arctic-embed-m-v1.5`
  (0.3002) and `gte-base-en-v1.5` (0.2741) were probed on Dylan's M5 Mac against stella's 0.3439 and
  both lose CI-resolved — below even the teacher stella replaced. `gte-modernbert-base` and
  `granite-embedding-english-r2` cannot be probed at all: **the note that said granite's Gram "is
  ~10.2 GB and chunks into RAM" was wrong** — that is fp32, and `stage0_ridge.solve_ridge` builds it
  in **float64**, so 50,368² is **20.3 GB**, above this box's 18 GB budget and above a 24 GB machine
  outright. Changing the solver's numerics would break comparability with every candidate already
  measured. Their table would also be *larger* than stella's (38.7 MB int8 vs 31.3), inverting the
  tie-break's assumption that 768-d buys a smaller artifact. Reusable bound from the ten rows now
  measured: base out-approximates large in every family by only +0.04 to +0.07, so a family whose
  large variant scores below ~0.28 cannot reach stella by shrinking.
  `m7_learnability_report_mac.json`.
- **Bare (unprefixed) teacher vectors as the distillation target — UNDER-DIAGNOSED, still open.**
  Closed on the teacher's own +1.85 from its prefix, i.e. the ceiling→table inference this project
  disproved. No bare-target table was ever fitted, and none of the queued prefix ablations tests
  the *target*. Closed-form settle: ~1–1.5 h. (Note the mandatory prefix ablations since ran on the
  full suite and a query prefix **hurts** −0.0019 at RUNTIME — which is not the same question.)
- ~~**"dev cannot test long-query behaviour" — PREMATURE as a standing claim.**~~ **ACTIONED, and
  the first answer was wrong.** `longspan_probe.py` was written and immediately produced a
  confounded result (a fresh document sample per length bucket), which licensed a training chain
  that was already running when a review caught it. Re-measured as nested prefixes of the same
  documents, agreement is **flat from 16 to 256 words**. So there is no length gap to close, lever
  #7 is closed on its own pre-condition, and **ArguAna remains an unmeasured extrapolation** —
  agreement is not relevance quality.
- **"stella's approximability is unexplained, so there is no attribute to search on" —
  UNDER-DIAGNOSED as a search-stopper, still open.** No literature sweep on token-linear
  approximability of encoder query spaces exists in `research/`, and free correlational arithmetic
  over the cached candidates (anisotropy, effective rank, in-sample ridge R² vs table ratio) was
  never run. Cheap; would convert "stella is best" from a measurement into a mechanism.
- **The `phase3_hparams` sweep never ran**: `temp=0.02` and `n_neg=32768` have been fixed since
  phase 1 without one. Not a recorded close, but not a tested choice either.
- **Provenance debt**: lever #1's λ-sweep numbers live only in ledger prose and a commit message,
  with no committed JSON — the repo's own rule, violated in the inverse direction.
- ~~"no clean untouched-final member is available"~~ — ACTIONED: two unused CQADupStack subforums
  added pre-freeze by a rule fixed before the pick.
