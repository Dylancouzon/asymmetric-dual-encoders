# Codex gpt-6-astra, read-only, evidence rating of the seven candidate findings

Date 2026-09-30. Access: audited: named result files only, no work/, no protected paths. Returned text below, verbatim. Dispositions are in `m15/LOG.md` (2026-09-30) and `m15/PLAN.md`.

---

## 1. Evidence ratings for the seven findings

1. **Retention versus compute — ADEQUATE.** Recomputing the 15 dataset means reproduces every headline score, retention ratio, 11/15 Zero wins, and both retention ranges. Sources: `results/m20_beir15_run.json:42`, `m20/STATUS.md:109–125`. Retention is **ratio of macro scores**, not mean per-dataset retention. The latency medians reproduce 0.111851 ms and 7.251130 ms; these are synthetic 20-word, batch-one measurements on an i7-10700KF, not representative workload latency (`results/m13_serving_costs.json:3–9`; `scripts/m13_serving_costs.py:118–146`). BEIR-15 is explicitly descriptive; disclose teacher exposure, reused reserved rows, and the twelve-forum aggregation (`m20/STATUS.md:127–132`). Without matched Stella timing, this establishes two measured query-cost points, not a complete teacher–student compute frontier.

2. **Tower-dependent distillability — WEAK, with a useful counterexample.** The denominator needs repair. The current report contains **11 candidate configurations, nine complete ceiling/table pairs**, with two missing ceilings (`results/m7_learnability_report.json:21–168`). The historical Spearman **0.000** reproduces over **eight configurations: seven distinct checkpoints plus Arctic-large’s alternative mean readout**, excluding later Mixedbread and the two missing-ceiling candidates. Their names are Stella-400M, BGE-base/large, E5-base/large, GTE-large, Arctic-large, and Arctic-large-mean. Arctic-large ranks fifth **within those eight**. Over the current nine complete configurations, rho is **0.0833**; over the original seven distinct checkpoints, **0.1429**.

   The **43.15%–71.56%** range divides each fitted table’s two-forum macro nDCG@10 by **its own teacher’s** two-forum macro, using that teacher’s document vectors—not a shared Stella index. The forums are programmers and physics; ridge λ is selected on this dev endpoint (`results/m7_learnability_report.json:2–5,129–168`). “Seven complete rows” in `research/constella-in-plain-english.md:141` is inaccurate as a description of the current report.

   The off-family check contains **only Stella and BGE-base**, on 1,790 SQuAD and 1,598 ESCI queries; macro difference +0.1018 [0.0930, 0.1107]. It supplies neither teacher-retention denominators nor an eight-tower correlation (`results/m7_offfamily_report.json:2–34`). “No disclosed exposure” cannot establish no exposure.

3. **Lookup-table ceiling — STRONG algebra; WEAK empirical ceiling.** Absorbability holds for freely parameterized normalized mean-pooled tables; it does **not** show that these transforms cannot improve a particular fitted or quantized table. Numerical differences reach **9.31e−14** for weights and **2.82e−14** for full SIF, rather than uniformly ~1e−16 (`results/m7_absorb_check.json:10–33`). More importantly, **4.73e−7 is teacher-target entropy; median KL is 1.08e−7**. The 99.75% figure covers **4,000 training queries against the recipe’s seeded random bank** (`m8/RESULTS.md:21`). “Moved <0.005” is false: several probes caused large losses; say **none improved by more than ~0.005** (`m8/FINDINGS.md:15–23`). Hard-candidate KL remained informative and untested as training (`m8/FINDINGS.md:53–56`).

4. **Distribution coverage — WEAK causally, ADEQUATE descriptively.** The 93.8% and 50.1% ratios check out (`m9/FINDINGS.md:20–24`); final FEVER scores are 0.6231 versus 0.6978 (`results/m20_beir15_run.json:331–333`). These comparisons do not isolate training coverage from capacity, objective, or other recipe differences. The 0.009 agreement and 0.160 overstatement belong to **M7’s table**, not the transformer: 0.764 versus 0.755, and 0.915 versus 0.755 (`m7/STATUS.md:18–21`). This is one retrospective agreement, not validated predictive calibration.

5. **System cost — ADEQUATE for encoding; UNSUPPORTED as general ANN dominance.** The ratio is **64.83×**. The system prototype uses synthetic documents/table and a different Nano architecture (`bench/edge_prototype_pair.py:6–11,41–42`). Its results vary with index configuration; they cannot establish production bottlenecks (`m9/RESULTS.md:171–180,249`).

6. **Silent swap-contract failures — ADEQUATE for documented implementation hazards; numerical examples UNVERIFIED.** I could not substantiate cosine **0.80/0.35** from the inspected permitted sources. Remove them pending exact fixtures and aggregation definitions. Float64 widening establishes output-memory/dtype consequences, not demonstrated relevance loss (`research/m22-dtype-issue-review-astra-2026-09-30.md:35–41`). Fusion depth materially changes quality, but constitutes a different retrieval configuration, not automatically a failed encoder swap (`m12/FINDINGS.md:84–99`).

7. **Registered comparisons — STRONG, with a denominator correction.** Clean-4 +0.017648 is supported (`research/constella-in-plain-english.md:93–100`). **+0.0032 [−0.0069,+0.0134] and −0.0389 apply to NDO-3, not all reserved four.** FEVER is separate. Disclose query-only uncertainty and weighting sensitivity: query-pooled Nano–BGE is −0.0121 with an interval excluding zero (`m20/FINDINGS.md:61–73`).

## 2. Lead and runner-up

**Lead with finding 1:** independently fitted query encoders provide substantially different encoding costs while retaining useful retrieval quality against one frozen document space. Its strongest research contribution is the measured breadth and tradeoff, not architectural firstness.

**Runner-up: finding 2**, framed as a concrete failure of selecting teachers by their own retrieval scores. The small, dev-selected screen supports a counterexample, not a law of zero predictiveness.

## 3. Use-case evidence and minimal valid measurements

Source searches found serving latency, synthetic system prototypes, and training/document throughput—not a deployed-query load test, autocomplete evaluation, or serving multicore scaling study.

| Use case | Existing evidence; minimal measurement; boundary |
|---|---|
| **High QPS** | Batch-one latency only. Measure **sustained completed requests/second under an arrival-rate sweep**, with representative inputs, concurrency/core counts, p95/p99 including queueing, errors, and a fixed latency SLO. Separate encoder-only and full retrieval throughput. **1/p50 is not throughput.** Encoder throughput cannot establish service capacity. |
| **Search-as-you-type** | No measured prefix relevance. BEIR truncation is valid only as **synthetic incomplete-query robustness against final-intent judgments**. It is not a validated typing proxy: intermediate intent, ambiguity, corrections, and relevance can differ. A product paragraph needs representative typing sessions, prefix-appropriate judgments, and event-to-result latency under realistic cadence/debouncing. |
| **Edge/on-device** | Synthetic million-vector feasibility exists. Measure released encoders plus real vectors on the target device, total memory/storage, cold/warm latency, quantization recall versus exact search, relevance, and sustained power/thermal behavior. An M5/Docker result cannot establish phone feasibility or battery life. |
| **Per-request tiering** | Shared-space quality is supported (`research/constella-in-plain-english.md:3–6`). Interleave both released encoders against one unchanged collection and verify retrieval/latency. That supports switching; a quality–cost advantage requires a fixed routing policy evaluated on held-out queries. |
| **New encoder for an existing index** | Nano demonstrates compatibility after **57.35 build hours**, not an end-to-end “in hours” recipe (`results/m13_build_record.json:857`). Time data preparation, teacher targets, fitting, export and validation separately; verify held-out quality against unchanged documents. One successful adaptation cannot establish arbitrary-index transfer or specialization gains. |

## 4. First hostile-review attack

**Finding 3’s claimed ceiling is the easiest attack:** algebraic capacity equivalence, an exhausted random-bank objective, and unsuccessful bounded probes do not establish an empirical optimum. Minimal fix: separate the exact invariance result from recipe-specific negative evidence, correct entropy/KL and improvement wording, and disclose unrun hard-candidate training.

The other essential fixes are the finding-2 denominators, NDO-3 labeling, and removing causal coverage claims without a controlled coverage ablation.
