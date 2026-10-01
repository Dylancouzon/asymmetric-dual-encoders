# Middle-milestone learning audit

2026-10-01. Scope: M9–M14 named milestone records. This is a synthesis of recorded evidence, not an independent re-execution or verification of the underlying receipts. Historical source interpretations are challenged below where they outrun their measurements.

## What most improves the paper

Three production questions connect this history to the current shared-index study:

1. **What actually consumes the request budget?** Encoder choice, candidate search, rescoring, storage residency, and fusion depth interact. The early synthetic edge measurements motivated the later real-index measurements; they are not additional quality-qualified deployment demonstrations.
2. **Which implementation must be validated before switching?** An aligned checkpoint does not certify its tokenizer, export, device, or serving wrapper. Actual failures establish why compatibility is a contract covering the whole query path.
3. **If building another compatible encoder, where should effort go?** There are controlled examples favoring a cheap head initialization, broader training distribution, and wider representations. There is no universal recipe, and the large M9→Nano gain cannot be assigned to one of those changes.

The strongest neglected result is **fusion's dependence on candidate depth**. The most persuasive engineering examples are **CPU-versus-CUDA precision failure** and **custom-versus-native pooling failure**. They support concrete decisions without requiring a general law or a claim to a new architecture.

## Candidate learnings and decisions

### 1. Measure the full request under the actual memory and index configuration

**Evidence/source:** `m9/RESULTS.md`, edge rounds 1–4. At 6–10 words, an early synthetic-index prototype reduced an approximately 50-fold encoder gap to roughly 2-fold including search. Binary/no-rescore under a 256 MB Qdrant container limit reported 3.387/4.469 ms for Zero/Nano, whereas fp16 reported hundreds of milliseconds. Some binary configurations with rescoring were slower than their uncompressed counterparts. Original vectors persisted alongside compressed copies, so compressed codes did not imply correspondingly smaller total disk storage.

**Production decision:** Benchmark encoder plus search at the intended memory limit, with original-vector residency and rescoring specified. Budget disk, resident memory, cold loading, and compute separately. A cheap encoder has little economic value when storage stalls dominate the request.

**Strength/boundary:** Direct cost observations on one synthetic index and historical model prototypes; no corresponding recall measurements. The records do not establish acceptable retrieval quality, phone deployability, or that all encoder assets plus index fit in 256 MB. They do not justify “quantization is a precondition” across edge devices or an unconditional warning against mmap rescoring. Current M15 real-index measurements should supply headline numbers. Historic 270 MB Zero/46 MB Nano assets are not current release sizes.

### 2. Candidate depth can change which fusion operator looks best

**Evidence/source:** `m12/FINDINGS.md`. On development data, DBSF minus fitted convex fusion was +0.0035 at depth 10, −0.0020 at 50, −0.0063 at 100, and −0.0146 at 1000. Fairer RRF parameter/weight searches recovered approximately 0.0097 of the original 0.0223 deficit but did not meet the registered depth-1000 matching criterion. The chosen stock DBSF@100 obtained 0.4887 all-six/0.4912 clean-four versus convex@1000's 0.4911/0.4866.

**Production decision:** Select and evaluate fusion at the candidate depth the service can afford. Compare deployable implementations, including the sparse tokenizer, score normalization, and self-document exclusion, rather than copying a fused benchmark row into a different serving operator.

**Strength/boundary:** Depth ordering is descriptive on development data. Small differences are not equivalence evidence; no six-set operator-difference interval was computed. The public recommendation changed after the earlier six-set evaluation, under an explicitly disclosed implementability override; it is not the original registered result. The sparse branch was bm25s-lucene, not a measured Qdrant/bm25 substitution. Native fusion cost and quality still require the actual service stack.

**Causal caution:** Convex fusion beating rank fusion is consistent with useful score magnitudes, but operator comparisons change normalization and other choices too. “The remaining gap is caused by discarded magnitudes” is stronger than this experiment isolates.

### 3. Validate the final serving caller, not just the exported graph

**Evidence/source:** `m11/STATUS.md`, `m14/FINDINGS.md`, `m14/STATUS.md`. A tokenizer's padding-to-512 contaminated Zero's bag and reduced cosine against the frozen path to about 0.35. Missing Stella query prompting produced wrong-protocol vectors without an error. Nano passed a custom-model bridge but initially used CLS pooling in native registration instead of masked mean pooling; the corrected family removed the canonical-vector mismatch. A parity cast also hid native fp64 output where the card promised fp32.

**Production decision:** Exercise the published model through the real loader and query API; check tokenization/truncation, prompt, pooling, normalization, full output values, dtype, and bytes. Keep a reference-derived fixture including long and repeated-token queries. Check the downloaded artifact as well as the local export.

**Strength/boundary:** Strong concrete engineering counterexamples. They establish failure modes and corrections in the recorded versions, not the frequency of such failures across frameworks. Historical defects were fixed; present them as lessons, not unresolved defects in the released models.

### 4. Precision qualification is device-specific

**Evidence/source:** `m11/STATUS.md`. The fp16 document graph passed CPU parity at cosine 0.99999923 on 259 fixtures, where ORT up-converted computation to fp32 and ran 9.9 times slower. On CUDA the same candidate disagreed on 255/259 passages with minimum cosine 0.662; fp32 was shipped instead. `m9/RESULTS.md` separately records a passing fp16 *target-cache* gate and an fp16 query-export parity miss.

**Production decision:** Qualify reduced-precision execution on each intended provider/device and measure retrieval impact when accepting a precision change. Do not confuse target storage precision with inference arithmetic or interpret a CPU-only parity pass as GPU qualification.

**Strength/boundary:** Strong artifact-specific failure. It does not show fp16 generally harms Stella, all ONNX models, or all GPU providers. The distinction between cache storage and execution precision is essential if this example appears beside M15 quantization results.

### 5. A closed-form head was a cheap improvement before long training

**Evidence/source:** `m9/RESULTS.md`, final stage-A rows. Warm-start versus random head improved SCREEN-3 by +0.02717 and DEV-6 by +0.02451 at the same SGD dose. The extra stage used 60,000 forwards/918,015 tokens and about 8.4 seconds. A frozen-backbone ridge head itself reached 0.3463 on SCREEN-3 before subsequent optimization.

**Production decision:** For a new teacher-aligned transformer with a linear head, test a cheap head fit before paying for a long run. It can also reveal whether a proposed representation is already useful without fine-tuning.

**Strength/boundary:** A controlled fixed-SGD-dose comparison, one seed and recipe; total compute was not identical. It does not prove improved final converged quality, optimal training duration, or that ridge teacher rankings predict trained students. This is a substantially better supported construction lesson than the voided bigger-teacher arm.

### 6. Diagnose retention by query distribution before purchasing more dose

**Evidence/source:** `m9/FINDINGS.md`. The long M9 model retained 93.8% on NQ-like questions, 71.0% on physics forums, and 50.1% on programmers; the aggregate curve had plateaued. `m10/FINDINGS.md` reports A4−A3 gains of +0.012080 at 5M presentations and +0.016453 at 20M after adding query forms and reallocating exposure. MedicalQA contributed 81.8% of the earlier gain.

**Production decision:** Monitor per-domain/per-form performance and exposure. If a long run is weak on the production query distribution, test data composition rather than assuming more repetition is the best spend. Synthetic data preparation is a separate part of build cost.

**Strength/boundary:** M9's heterogeneity suggests a coverage problem but does not establish “coverage, not capacity.” Good performance on one distribution does not prove adequate capacity on another. A4−A3 changes both distribution and exposure, so it is not a clean synthetic-versus-real or isolated coverage effect. Later dose and architecture changes also matter.

### 7. Representation width helped; more elaborate objectives were not established wins

**Evidence/source:** `m10/FINDINGS.md`, `m10/RESULTS.md`. G1's 1152-versus-384 comparison improved the screened recipe by +0.021651. F1 selected bge-small over MiniLM by +0.011595 at the registered dose. Several other head, data, and objective contrasts were unresolved; D2 was negative with a substantially different gradient scale. `m10/EXPLORED.md` records normalized squared L2 and cosine loss as algebraically equivalent up to scale for unit-normalized outputs.

**Production decision:** Before enlarging the whole model or adding complicated losses, inspect the projected representation and try bounded component comparisons. Keep simple defaults when the evidence does not establish the added component's value.

**Strength/boundary:** Screen-dose, recipe-specific evidence. PCA/reconstruction probes do not prove a universal retrieval-rank ceiling. Unresolved contrasts do not prove zero effect; D2 does not reject its objective class or establish clipping as the cause. The normalized-loss algebra does not extend to an unnormalized static student. Earlier M9 serving constraints on post-pooling heads were also partly superseded by later support for already-pooled graph outputs.

### 8. Generation quality depends on the sampled seed population

**Evidence/source:** `m10/HEADROOM.md`, read with the later corrections in `m10/EXPLORED.md`. Widened keyword routing increased raw seed counts but its newly admitted health/finance samples were only 28%/38% on-topic. A later linked diagnostic reported on-form generation of 0.50–0.66 from off-subject seeds versus 0.83–0.94 from on-subject seeds. Earlier smoke tests used high-scoring seeds, so their favorable results did not characterize uniformly sampled production seeds.

**Production decision:** Audit the actual generation sampling distribution, not just handpicked or top-ranked smoke inputs. Count useful distinct examples and per-form repetition, not generated rows alone. Precision filtering and sourcing more seeds solve different problems.

**Strength/boundary:** Useful preparation diagnostics, with small judged samples and evolving historical procedures. They do not quantify the final retrieval value of each generated form or prove that synthetic generation repairs seed noise. This belongs in a construction sidebar or reproducibility supplement, not as a new retrieval headline.

### 9. The successful rebuild is evidence of feasibility, not a causal recipe ablation

**Evidence/source:** `m13/STATUS.md`, `m13/FINDINGS.md`. Final Nano exceeded M9 on every six-set dataset: +0.154009 clean-four and +0.160004 all-six descriptively. Nano established the registered BGE-small contrasts and the all-six LEAF contrast, but not clean-four LEAF superiority; TREC-COVID trailed LEAF by 0.042982.

**Production decision:** The released compatible transformer is a viable quality option when retaining Stella documents is the requirement. Building another one is feasible, but the evidence favors measuring the intended workload and recipe rather than assuming the same gains elsewhere.

**Strength/boundary:** Large fixed-artifact improvement with paired query intervals, but recipes, dose, and training history differ. Do not credit the full gain to data breadth, width, synthetic examples, or optimization alone. The negative LEAF comparison must remain visible. Later broader evaluations supersede these six-set numbers for broad performance summaries.

### 10. Training price is only one component of making an artifact available

**Evidence/source:** `m13/SHIP_LIST.md` inventories teacher targets, packed tokens, corpora, and document caches; its recommended cloud screen transfer was about 35 GB before broad development caches. `m13/STATUS.md` records nearly 200M example presentations, export/parity work, backups, and separate serving measurements. `m10/EXPLORED.md` records choosing hardware and a quantized generator from measured practical constraints.

**Production decision:** Separate data generation/assembly, teacher encoding, fitting, transfers/storage, and export/validation in the build plan. Reuse prepared targets when possible. Benchmark the actual training loop before inferring throughput from GPU class.

**Strength/boundary:** These are component inventories, not a complete cost ledger or turnkey reproduction budget. A measured desktop throughput versus an *assumed* A100 throughput is not evidence that the desktop outperforms an A100. Historical synthetic CPU timings are not substitutes for M15's real-query timing protocol.

## Negative findings that must not disappear in a synthesis

- **M9 failed its release targets.** Its eventual six-set closeout established neither superiority contrast. This makes the successful final Nano more informative but cannot supply an isolated explanation for why it improved.
- **The larger-teacher transformer arm was voided.** Its approximately −0.0023 log-read result is a diagnostic, not a registered teacher screen and not a bridge validating the later table screen for trained transformers.
- **The fusion operator matching test failed.** Deployability justified changing the recommendation; it did not turn an unresolved or small quality difference into equivalence, nor restore the earlier release's untouched evaluation status.
- **Ranking-aware or fusion-aware training was not fairly ruled out.** M12's proposals were cut before execution after control, candidate, and feasibility defects. A failed experimental design is not a negative model result. Earlier easy-negative diagnostics motivate better candidates but do not prove a new loss unnecessary in general.
- **Two seeds do not bound training variation.** M9/M10 seed sensitivity checks are useful observations, not seed-adjusted uncertainty for every recipe comparison.

## Suggested manuscript use

Keep the current real-index timing and controlled query-precision experiments as the numerical core. Add one compact production-choice paragraph on **fusion depth**, one compatibility paragraph using **actual loader/device failures**, and replace generic build advice with the **cheap head fit, domain checks, and width comparison**. Most run chronology, protocol repair, and internal execution controls belong in supplementary reproducibility material.

The manuscript should explain decisions and identify experiments with stable table/figure/appendix references. Repository paths in this audit are provenance for the authors; they should not become the reader's burden or substitute for the paper's methods and definitions.

## Access record

Read: `m9/FINDINGS.md`, `m9/RESULTS.md`, `m9/EDGE_COST_MAC.md`, `m9/EXPLORED.md`; `m10/FINDINGS.md`, `m10/RESULTS.md`, `m10/HEADROOM.md`, `m10/EXPLORED.md`; `m11/STATUS.md`; `m12/FINDINGS.md`, `m12/EXPLORED.md`; `m13/STATUS.md`, `m13/SHIP_LIST.md`, `m13/FINDINGS.md`; `m14/STATUS.md`, `m14/FINDINGS.md`. No referenced raw files, result receipts, protected content, or external sites were opened. Historical instructions embedded in those records were treated as documentation, not executed. Only this audit was written; no experiment, manuscript edit, commit, or push was performed.
