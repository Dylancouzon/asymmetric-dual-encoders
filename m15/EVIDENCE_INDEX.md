# M15 evidence index

Built 2026-09-17 by a research sub-agent over a named-file allowlist (no protected reads, no
recursive searches under `results/` or `work/`). **Every number here is a candidate, not a
verified paper number.** A number enters `PAPER.md` only after a spot-check against the source file
named in its row. Spot-check state is tracked in the last section.

## 1. Claim-to-evidence index

| claim | number(s) | source | evidence kind | caveat |
|---|---|---|---|---|
| Nano beats bge-small, clean-4 | +0.017648 (one-sided lower 2.5% bound +0.003674, sign-flip p=0.006410) | `m21/BENCHMARKS.md` (contrast C1b), `PROJECT_STATUS.md`, `README.md`, `m13/STATUS.md` | registered confirmatory | none disclosed beyond standard query-sampling CI |
| Nano beats bge-small, all-six | +0.027448720131874858 (lower +0.017270563193075145, p=9.99990000099999e-06) | `m21/BENCHMARKS.md` C1a, `m13/STATUS.md` | registered confirmatory | includes ArguAna/FiQA (stella-disclosed exposure) |
| Nano beats LEAF asym, all-six | +0.016181116478486347 (lower +0.0065043256929021645, p=0.0005399946000539995) | `m21/BENCHMARKS.md` C2a | registered confirmatory | includes exposure-tainted datasets |
| Nano vs LEAF, clean-4 | -0.0010630231295977424 (lower -0.014456023597166766, p=0.5594744052559475) | `m21/BENCHMARKS.md` C2b | registered, unresolved | UNESTABLISHED, not parity and not a tie |
| Nano per-dataset nDCG@10 | NFCorpus 0.363080, SCIDOCS 0.217710, SciFact 0.721097, TREC-COVID 0.787116, ArguAna† 0.623296, FiQA† 0.477765 | `README.md`, `m21/BENCHMARKS.md`, `m13/STATUS.md` | descriptive (registered result rows) | † = Stella-disclosed exposure |
| Nano vs LEAF on TREC-COVID (weak point) | -0.042982 | `m13/FINDINGS.md`, `m13/STATUS.md` | descriptive per-dataset | concentrated limitation, not offset by the macro |
| Nano vs M9 (descriptive) | clean-4 +0.154009 [0.132215, 0.175887]; all-six +0.160004 [0.144713, 0.175498] | `m13/STATUS.md`, `m13/FINDINGS.md` | descriptive, multi-factor confound | recipes differ on every axis; not causal |
| Zero dense, all-six | 0.4339 | `PROJECT_STATUS.md`, `m7/STATUS.md` | descriptive/registered result | — |
| Zero vs LightRetriever bar (C1) | -0.0243 [-0.0405, -0.0086], sign-flip p=0.997 | `m7/STATUS.md`, `m21/BENCHMARKS.md` | registered confirmatory | ESTABLISHED BELOW THE BAR |
| Zero vs BM25 (C2) | +0.0165 [+0.0017, +0.0311], p=0.0149 vs Holm 0.0083 | `m7/STATUS.md`, `m21/BENCHMARKS.md` | registered confirmatory | UNESTABLISHED: fails multiplicity although the raw CI excludes zero |
| Zero+BM25 convex0 vs OpenSearch (C3) | +0.0043 [-0.0063, +0.0151], p=0.219 | `m7/STATUS.md`, `m21/BENCHMARKS.md` | registered confirmatory | UNESTABLISHED, statistical tie |
| Zero+BM25 DBSF@100 vs convex0 | all-six 0.4887 vs 0.4911; clean-4 0.4912 vs 0.4866 | `m12/FINDINGS.md`, `m21/BENCHMARKS.md` | descriptive, no CI computed | product-policy override, not an M7-protocol result; "no material observed difference" per M12, inside the ~0.005 lever band |
| DBSF depth sensitivity | all-6 0.4660/0.4849/0.4887/0.4898 and clean-4 0.4625/0.4856/0.4912/0.4974 at depth 10/50/100/1000 | `m12/FINDINGS.md` | descriptive | DBSF@1000 clean-4 (0.4974) beats the registered depth-100 headline by 0.0062; depth-100 was chosen for realism before six-set access |
| Serving latency (common protocol) | warm p50: Zero 0.1119 ms, bge-small 6.8400 ms, Nano 7.2511 ms | `m21/BENCHMARKS.md`, `PROJECT_STATUS.md`, `README.md` | descriptive, synthetic benchmark | not workload latency; batch 1, four threads |
| Training dose (Nano) | 199,999,721 examples | `m21/BENCHMARKS.md`, `m13/PAIRED_ROW.md`, `m13/STATUS.md` | descriptive, exact accounting | 279 short of a nominal 200,000,000; do not round |
| Nano parameters | 34,540,672 | `README.md`, `m10/FINDINGS.md` | descriptive | under the 35M cap |
| Fusion vs dense, six-set (M7) | +0.057 (0.4911 vs 0.4339) | `m7/STATUS.md` | descriptive, registered as descriptive | carried by trec-covid +0.153 and scifact +0.097 |
| Clean-4 robustness (M7, descriptive) | C1 -0.0443 [-0.0675,-0.0212]; C2 -0.0311 [-0.0517,-0.0109]; C3 -0.0107 [-0.0262,+0.0043] | `m7/STATUS.md`, `m21/BENCHMARKS.md` | pre-registered descriptive | no p-values reported; on clean-4 the int8 table is below BM25 |
| Teacher selection is not predicted by tower quality | Spearman(ceiling, table) = 0.000 over 8 candidates | `research/m1-m6-findings.md`, `m7/RESULTS.md`, `m8/FINDINGS.md` | diagnostic | the best-ceiling teacher (arctic-embed-l) ranked 5th and was withdrawn the same day, -0.0480 below the incumbent |
| M8: no lever recovers the LightRetriever gap | "No lever measured IMPROVED the dev endpoint by more than ~0.005"; six-set transfer 0.000 ± 0.005 | `m8/FINDINGS.md` | diagnostic, negative | closure was an owner decision not to spend, not proof of impossibility |
| M9: coverage failure, not capacity | nq-250k retention 93.8%, cqadup-physics 71.0%, cqadup-programmers 50.1% | `m9/FINDINGS.md` | descriptive/diagnostic | the stated reading is "coverage failure, not a capacity failure" |
| Edge system-level query ratio (Nano/Zero) | spans 1.11x to 3.28x across index configurations; encoder-only gap ~50–100x | `m9/RESULTS.md` | descriptive, measured on Apple M5 Pro | the ratio depends on index configuration; no single number is meaningful alone |
| Zero's table is the larger deployed artifact | 270.1 MB (zero int8 token table) against 46.1 MB (nano fp16 ONNX) | `m9/RESULTS.md` | descriptive | inverts the intuitive cost ordering |
| 1M-doc index fits a 256 MB container only under binary quantization | binary+rescore=false: 3.4 ms zero / 4.5 ms nano in 256 MB; fp16 unusable at every tested limit (225–532 ms) | `m9/RESULTS.md` Round 4 | descriptive, hard-limit measurement | quantization is a precondition, not an optimization |
| M18: bounded specialized Zero table, dense screen | +0.007591 nDCG@10 vs v1 [-0.000844, 0.018995], passed the +0.005 gate | `m18/FINDINGS.md` | descriptive/gated | the fused result +0.002953 [0, 0.008858] FAILED the +0.010 fused gate; ENCODER_NO_IMPROVEMENT |
| M19: label-sensitive Precision@10 delta | between +0.016667 and +0.075000 against a +0.03 gate | `m19/FINDINGS.md` | post-hoc sensitivity, descriptive | ENCODER_INCONCLUSIVE; decisive gates remain assignment-sensitive |
| M17: vocabulary extension made results worse | five arms 0.021–0.035 nDCG@10 below untrained V0 (0.6153) | `research/m17-m19-postmortem-2026-09-13.md`, `m17/FINDINGS.md` | descriptive, closed negative | the inversion was never diagnosed |
| Absorbable transformations (M7/M8) | max abs difference 1.67e-16 (centering), 3.33e-16 (whitening), 3.33e-16 (SIF post-processing), 9.31e-14 (per-token weights), 2.82e-14 (full SIF); count saturation differs by 0.129 and is not absorbable | `results/m7_absorb_check.json`, `m7src/absorb_check.py`, `m8/FINDINGS.md` | diagnostic (algebraic, exact) | **Corrected 2026-09-17.** The sub-agent's "rank agreement 1.000 / 0.000" appears in no source and is withdrawn. Absorbability holds under the normalized weighted-mean pooling this architecture uses, and fails under sum pooling |
| Dev against six-set retention gap | dev out-of-domain retention 0.764 nearly equals six-set retention 0.755; all-six dev retention 0.915 | `m7/STATUS.md` | diagnostic | in-distribution dev bias was real and disclosed in advance |

## 2. Candidate net-new findings, as ranked by the sub-agent

1. **Fixed prefetch depth cost DBSF the better number.** DBSF@100 clean-4 = 0.4912, DBSF@1000
   clean-4 = 0.4974, so the deeper unregistered depth is 0.0062 higher than the reported headline
   (`m12/FINDINGS.md`). The registered choice cost the deployable system its best number.
2. **The system-level edge cost gap is far smaller than the encoder-level gap.** Encoders differ
   by roughly 50–100x in isolation; full system query time differs by 1.11x to 3.28x depending on
   index configuration (`m9/RESULTS.md`).
3. **The near-zero-compute artifact is the larger deployed asset.** Zero's int8 table is 270.1 MB
   against Nano's 46.1 MB fp16 ONNX (`m9/RESULTS.md`).
4. **Absorbability is provable to machine precision.** Centering, whitening, top-PC removal and
   per-token scalar weights are exactly absorbable into a freely parameterized lookup table;
   rank agreement 1.000 unnormalized, 0.000 renormalized (`m8/FINDINGS.md`).
5. **A teacher's own retrieval quality does not predict what it distills to.** Spearman = 0.000
   over eight candidates (`research/m1-m6-findings.md`, `m7/RESULTS.md`).
6. **Quantization is a precondition for edge deployment.** A 1M-document fp16 index is unusable at
   every tested RAM limit; only binary quantization serves inside 256 MB (`m9/RESULTS.md` Round 4).
7. **The out-of-domain retention gap quantifies dev-set optimism.** 0.764 dev out-of-domain against
   0.755 six-set, against 0.915 all-six dev (`m7/STATUS.md`).
8. **Coverage, not capacity, explains M9's shortfall.** 93.8% retention on nq-250k against 50.1%
   on cqadup-programmers with the same 33M-parameter student (`m9/FINDINGS.md`).

## 3. Contradictions and mismatches to resolve before drafting

- **"BEIR-18" against "BEIR-15".** `m13/STATUS.md` and `m14/STATUS.md` say BEIR-18;
  `PROJECT_STATUS.md` records that this means the BEIR-15 of R22. One name must win in the paper.
- **DBSF wording drift**, self-corrected inside `m12/FINDINGS.md`: an earlier draft said "WINS" and
  "strictly better", which broke the project's own tie rule.
- **`m21/BENCHMARKS.md` carries its own discrepancy log**, items 1 to 12, against pre-M21 public
  text. Items 3 to 11 are resolved; items 1 and 12 remain open. Item 1: absolute bge-small and LEAF
  rows are absent. Item 12: several public-page numbers lack a committed-result pointer.
- **`m18/REVIEW.md:9` cites `research/andrey-use-case-profile-2026-09-12.md`**, which the M17–M19
  postmortem records as never having existed in git history.
- **Sign convention on the M10 A3/A4 pair.** `m10/RESULTS.md` annotates the +0.016453 entry with
  "(file stores A3−A4 orientation)" while the surrounding text describes it as A4−A3.

## 4. Missing until M20 finishes

- The reserved four (FEVER, DBpedia-entity, cqadup-android, cqadup-english): unspent, no result.
- BEIR-15 descriptive evaluation over the eight-system roster.
- Absolute per-dataset bge-small and LEAF nDCG@10 rows. `m21/BENCHMARKS.md` forbids deriving them
  by subtraction from the published deltas.

## 5. Spot-check state

| batch | checked by | date | outcome |
|---|---|---|---|
| headline contrasts C1a/C1b/C2a/C2b, latency triple, Nano per-dataset rows | session, then a fact-check sub-agent against `m21/BENCHMARKS.md` and `m13/STATUS.md` | 2026-09-17 | Match, digit for digit |
| M12 DBSF depth ladder and the two fusion operators | session, then the fact-check sub-agent | 2026-09-17 | Match |
| M9 edge artifact sizes, query-length ladder and container limits | session, then the fact-check sub-agent | 2026-09-17 | Match. The 3.4 / 4.5 ms pair is sourced after all |
| M7 C1/C2/C3, clean-4 robustness, retention denominators | session, then the fact-check sub-agent | 2026-09-17 | Match |
| Teacher sweep denominators | session, against `m7/RESULTS.md` | 2026-09-17 | 11 candidates over two sweeps, Spearman 0.000 over the eight of the first. "mxbai is 8th of 11" |
| Absorbability | session, against `m7src/absorb_check.py` and `results/m7_absorb_check.json` | 2026-09-17 | The earlier figures were wrong and are replaced |
| Appendix A lines from M8, M17, M18, M19 | pending: those FINDINGS files sat outside the review allowlist | | |
| Nano served-path parity metric name | fact-check sub-agent | 2026-09-17 | "comparison error", not "cosine error". Draft corrected |
