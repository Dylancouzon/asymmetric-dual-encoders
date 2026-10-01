Found seven essential issues:

1. **Held-out contrasts have an unspecified, misleading partition.** [PAPER.md:138](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:138) labels the contrasts simply “held-out.” In `results/m13_reserved_run.json`, +0.0032 and −0.0389 are **NDO-3** estimates: DBpedia receives weight ½ and each CQADup forum ¼; FEVER is excluded. The −0.0121 query-pooled estimate also excludes FEVER. Name that partition and weighting explicitly; these are not aggregates over the displayed held-out four.

2. **Exploratory studies are implicitly promoted to registered evidence.** [PAPER.md:24](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:24) defines anything specified before observation as “registered.” But `MEASUREMENTS.md` and cards C16/C17 explicitly classify E19/E20 as **exploratory, with pre-specified analyses**. Sections 3–4 omit that distinction. The shared-binary selection at line 88 and query-precision experiment at line 102 likewise lack their exploratory labels, recorded in C10 and C15.

3. **The E20 presentation obscures the primary result and overstates the recovery summary.** [PAPER.md:98](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:98) says relevance is reported first, although recovery leads. It replaces TREC-COVID’s primary multiplier result with an explanation: the committed summary reports **12 spaces above 1, 13 at or below, median 0.25 among reached spaces, eight censored**. The agreed wording requires reporting that mixed result and treating the 50-query explanation as plausible. Also qualify the recovery median as **over reached spaces** and disclose recovery-specific censoring; Appendix C supplies only relevance-loss censoring.

4. **Registered contrasts are not shown in full.** [PAPER.md:134](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:134) omits Nano’s decision bounds for three contrasts: +0.0173, +0.0065, and −0.0145. Line 150 omits all three Zero clean-four intervals and their **descriptive sensitivity** classification. These are available in `m21/BENCHMARKS.md` and required by the project’s full-contrast disclosure rule.

5. **The shuffle analysis falsely claims only one correlate was measured.** [PAPER.md:42](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:42) contradicts C12, which also reports fertility (+0.101) and fragility (+0.066). C12 additionally discloses that the two correlated quantities share Stella’s full score, creating mathematical coupling; that limitation is absent from the manuscript.

6. **“Search dominates any encoder under 2 ms” is contradicted by the example itself.** [PAPER.md:88](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:88): Nano’s selected binary row has **0.869 ms search** within **2.484 ms encoding-plus-search**. Search therefore does not dominate this sub-2-ms encoder. Restrict the claim to Zero or the configurations that support it.

7. **The required MS MARCO licence-role disclosure is missing.** [PAPER.md:124](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:124) supplies validation-only use and backbone exposure, but no licence-role statement, explicitly required by `instructions-m15.md`, “Evidence and claims.”

No files changed or experiments run.

Codex session ID: 01a0f65b-3b91-7f33-8b0a-2389cdc37592
Resume in Codex: codex resume 01a0f65b-3b91-7f33-8b0a-2389cdc37592
