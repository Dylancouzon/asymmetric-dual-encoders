# Asymmetric Dual Encoders — Viability Scoping

## North star

Decide, with defensible numbers, whether asymmetric dual encoders (big document encoder in the cloud, near-zero-compute query encoder on the edge) beat the obvious alternative: running a small 20–100M conventional embedding model on the edge. The research question from `instructions.md`: **how much retrieval quality do you retain as query-side computation approaches zero?**

Target architecture being scoped: server indexes docs with a large LLM encoder; edge client holds two Qdrant collections (token→vector lookup table + document index); query = tokenize → lookup → average → normalize → ANN search. No transformer at query time, near-zero cold start.

## Deliverable

A decision report (Artifact) with a quality-vs-query-side-cost frontier: nDCG@10 on a fixed BEIR subset for each point on the spectrum, plus edge-relevant costs (artifact size on disk, query latency on CPU, cold start), and a Qdrant Edge two-collection prototype to prove the operational architecture.

## Constraints

- Local machine: Apple M5 Pro, 24 GB RAM, 15 cores, MPS. 8B doc encoder is tight; smaller backbone preferred; pick small BEIR datasets.
- Quality numbers come from exact (brute-force) search so ANN recall is not a confound. Qdrant Edge is used for the architecture/latency prototype, not the quality numbers.
- Dylan is hands-off; decisions get made here and logged below.

## Stage plan

- [x] M0: environment check, this file. (M5 Pro, 24 GB RAM, Docker + uv available.)
- [x] M1: deep research, done 2026-08-24. Full notes in `research/`: `lightretriever.md`, `landscape.md`, `methodology.md`, `sparse-inference-free.md`, `literature.md`.
- [x] M2: done, then adversarially verified (`research/verification-m2.md`: 3 BLOCKER / 9 MAJOR / 9 MINOR — all blockers fixed, see below). 15 configs × 5 datasets in `results/quality.json`. Benchmark resolution: ±0.007 nDCG on the 5-ds macro average (paired bootstrap), so the top cluster {arctic-m-v1.5, granite-r2, bge-small, leaf-asym, leaf-sym, gte-small} is a statistical tie internally; solid gaps only across clusters. leaf-ir-asym holds 97.1–98.6% of its 109M teacher (CI) with a 23M query encoder. BM25 (bm25s-lucene) 0.379 avg beats every static (best 0.344) and nearly matches e5-small 0.396. Statics collapse on FiQA (~0.17–0.19 vs ~0.40 for transformers).
- [x] M3: done 2026-08-25. Dense reproduces the paper within noise after the BOS fix (avg-5 0.4114 vs paper 0.4136); sparse −3..−5 (cause capped, see findings); hybrid conservative.
- [x] M4: done. Full 21-system × 6-dataset matrix in `results/FINAL_MATRIX.md` (+ costs.json, significance.json). TREC-COVID added; all baselines validate against official MTEB ≤0.003.
- [x] M5: done. Two-collection Edge prototype works: 0.9 ms/query zero-transformer (0.42 lookup + 0.48 HNSW), shard load 0.24 s. ANN gap −1.8 nDCG at default ef, −0.5 at ef=512 (1.42 ms). fp16 shards: table 1.82 GB (bloated by an HNSW index a retrieve-only collection doesn't need; raw fp16 table = 466 MB), docs 754 MB. `results/edge_prototype.json`, `results/edge_variant.json`.
- [x] M6: done 2026-08-25. Report artifact: https://claude.ai/code/artifact/db771dd1-2d59-4c34-9d6e-70dd4d337d16 ("Zero-Compute Query Encoders"). Gates run: Codex adversarial verification (gpt-5.6-terra, all findings actioned) → Codex writing pass (gpt-5.5, zero number changes, verified by diff) → humanizer (clean). andrey-review deliberately skipped: internal decision report, not channel DevRel content, and two adversarial technical passes already ran.
- [x] M7: **final run done 2026-08-28** (freeze `d24c704`, tag `m7-freeze`, one `--infra-retry` after a harness interrupt; access spent, `m7-six-spent` on origin). **Zero tier claims**: release bar missed CI-resolved (int8 avg-6 0.4339 vs LR-dense-pertask 0.4583, −0.0243 [−0.0405, −0.0086]); vs BM25 +0.0165 positive but not surviving the registered familywise rule; fused system 0.4911 vs OpenSearch 0.4868 a statistical tie (+0.0043 [−0.0063, +0.0151]). Clean-4 robustness is worse (table below BM25 there); six-set retention vs teacher 0.755 — the dev out-of-domain read (0.764), not the all-six dev read (0.915), was the honest forecaster. Fusion vs dense +0.057 descriptive is the bright spot. The pre-registered miss-is-publishable framing applies; untouched-final reserved for M8; report pending. Full detail: `m7/STATUS.md`, `results/m7_final_run.json`. Original scope line kept for context: self-directed (WSL2 on the Windows/RTX 3080 box; repo on ext4, never /mnt/c) build + release of a Qdrant lookup-table query encoder. Binding mandate with comparators, tiers, eval protocol, and ops: `instructions-m7.md`. Research: `research/m7-novelty.md` (unpublished as of 2026-08-25), `research/m7-data-licensing.md`, `research/m7-teacher-shortlist-2026-08-26.md` (the pre-relaxation `m7-teacher-shortlist.md` was deleted in the 2026-08-28 cleanup; git history has it). Host setup checklist for Dylan: `setup-windows.md`. Core decisions 2026-08-25: teacher bge-base-en-v1.5 (swap delegated, no competitor vendors) — **SUPERSEDED 2026-08-26: the teacher is `NovaSearch/stella_en_400M_v5`, chosen on the distilled TABLE rather than the tower; see the decision log below and `m7/LEDGER.md`**; clean data stack, MS MARCO excluded from the RELEASE stack permanently (its terms are non-commercial-research-only; IBM Granite is the precedent) — with ONE research-only variant approved 2026-08-28 as the final M7 task, to measure what the exclusion costs, never released and refused by `freeze.assert_releasable`; aim = CI-resolved win over OpenSearch 0.4868 (candidate: released zero-query-compute system, fusion allowed and labeled), release bar = CI-resolved win over LR-dense-pertask 0.4583 (candidate: the released int8 dense table). Frozen comparator per-query vectors in `results/perquery.json`, dataset content pinned by `results/eval_manifest.json` + `results/frozen_eval/` (vendored queries+qrels), validated by `scripts/validate_perquery.py` — 50/54 cells <5e-5, four cells ≤3e-4 allowlisted with a provenance note in FINAL_MATRIX.md. Research/web work in Sonnet subagents. Plan gates run 2026-08-25, all findings implemented: Codex gpt-5.6-sol #1 (Dylan's model pick; 7 BLOCKER/23 MAJOR/8 MINOR), Opus (6 BLOCKER/10 MAJOR + leanness cuts), Codex gpt-5.6-sol #2 fresh thread (3 BLOCKER/9 MAJOR/4 MINOR — comparator drift manifest, capacity probe made gate-ineligible with a falsifiable bar, tier candidates fixed to the released artifacts, Holm at family α=0.025, dev suite pinned). Session state lives in small files under `m7/` (STATUS.md = one-screen status Dylan reads on GitHub, RESULTS.md, EXPLORED.md, LEDGER.md) with frequent commit+push under a standing grant scoped to the M7 work branch (headless box). CC BY-SA position confirmed by Dylan 2026-08-25: NQ/SQuAD/HotpotQA/FEVER approved for training with model-card attribution. Remaining item for Dylan: run setup-windows.md; HF release go stays his.
- [ ] M8 (queued, after M7's final run — renumbered 2026-08-28, Dylan: "M8 will be the v2"): the learnings-driven v2 of M7's zero-compute table. Tiny mandate in `instructions-m8.md`; scope and levers are set in `m8/LEDGER.md` AFTER M7's final number is read, but the now-or-never items were pre-registered 2026-08-28 before that number exists: M8's confirmatory sets are the RESERVED untouched-final four (FEVER, DBpedia-entity, cqadup-android/english — M7's final run defaults to skipping the tail so they stay un-scored; scoring them burns them for M8), the M7-vs-M8 comparison is confirmatory only there (frozen artifacts, paired, one access), and M7's six are development-informed for M8 (descriptive continuity only). Carried-in levers each needing their own pre-registration: bigram rows trained through the forward, pooling trained through at scale, doc2query with a commercially clean generator (Dylan's licensing ruling still open), the negatives/step-count confound. The clean-stack-tax variant stays an M7 task; its result is an M8 input.
- [ ] M9 (queued, after M8 — renumbered from M8 on 2026-08-28): LEAF-style distilled small query tower against the frozen teacher the table line ships (**stella_en_400M_v5**; follows M8's teacher if M8 swaps) — tiny mandate in `instructions-m9.md` (the old `instructions-m8.md`, brought up to date: the bge-base teacher line and the shared-tokenizer rationale for a bge-small student were stale after the 2026-08-26 swap; student shortlist to be re-derived at M9 start). Release bar: CI-resolved above bge-small symmetric 0.5042. Comparator per-query vectors (leaf-ir-asym 0.5155, mdbr-leaf-ir 0.5123, arctic-m 0.5264) frozen into `results/perquery.json` 2026-08-25 while the Mac caches still existed.

## Candidate routes (to be confirmed in M1)

- LightRetriever (arXiv 2505.12260) with released pretrained artifacts — the zero-compute-query anchor point.
- Small symmetric baselines (20–100M): bge-small-en-v1.5, e5-small-v2, all-MiniLM-L6-v2, gte-small, arctic-embed-xs — the "just run a small model" alternative.
- Static embedding models used symmetrically (Model2Vec/potion, sentence-transformers static-retrieval) — the zero-compute symmetric strawman.
- Aligned asymmetric pairs that need no training by us (e.g., MongoDB LEAF: docs with teacher, queries with distilled small model) — the middle of the spectrum.
- Inference-free sparse retrieval (SPLADE-doc, uniCOIL/TILDE doc expansion, OpenSearch inference-free doc encoders, doc2query+BM25, BM25 itself) — same zero-query-compute property, sparse instead of dense, Qdrant-native. Added 2026-08-24 after Dylan flagged the instructions as a surface-level draft; this family is a direct competitor the draft missed.
- Broader literature sweep for LightRetriever successors, industry asymmetric towers, and tiny-adapter alignment (static model + learned projection into a frozen big-model doc space).

## Headline results (6 named datasets, exact search; claims stated only where bootstrap-resolved)

Quality groups (avg-6 nDCG@10; strict within-group ordering NOT established — Codex blocker 2):
- Group A, small/mid transformers on the query side: arctic-m-v1.5 0.5264, leaf-ir-asym 0.5155, mdbr-leaf-ir 0.5123, bge-small 0.5042, arctic-s 0.4993, granite-r2 0.4947, gte-small 0.4837.
- Group B, zero-neural-query-compute: opensearch-doc-v3-gte 0.4868 (overlaps group A's tail), LR-hybrid-pertask 0.4720, LR-hybrid-websearch 0.4594, LR-dense-int8/pertask 0.4586/0.4583, LR-dense-websearch 0.4320, BM25 0.4174. (LR sparse/hybrid carry the unresolved −3..−5 reproduction gap → "conservative local reproduction", not a finished competitor result.)
- Group C, symmetric statics: 0.3193–0.3601 — decisively last.
- Resolved pairwise: LR-dense-websearch vs bge-small −7.2 [−8.8,−5.7]; vs best static +7.2 [+5.2,+9.2]; per-task tables +2.6 [+2.0,+3.3]; leaf-asym vs teacher −1.1 [−1.5,−0.7]; leaf-asym vs leaf-sym unresolved (+0.3 [−0.2,+0.8]) → say "retains comparable quality", not "beats".
- Costs: LR lookup 0.023 ms / 466 MB fp16 raw table (int8 233 MB, quality-free); OpenSearch query side 0.018 ms / 0.9 MB idf (doc-side postings cost measured separately, see reruns); 33M transformer ~5 ms / 66 MB fp16 / 1.3 s load; statics ~0.2 ms / 15–65 MB. Cost rows are not one number: query asset ≠ doc index ≠ hydration.

## Rerun outcomes (2026-08-25, post-Codex)

- Projection (fixed loaders, oracle λ): potion-8M→arctic-m best 0.3036; potion-32M→arctic-m best 0.3280, below its own symmetric 0.3427. Negative result is airtight: linear post-hoc alignment into a contextual doc space fails even with test-set-tuned regularization.
- New resolved pairs (6-ds bootstrap): leaf-asym > bge-small +1.1 [+0.2,+2.0] p=0.016; opensearch > lr-hybrid-pertask +1.5 [+0.2,+2.8] p=0.022; opensearch ties gte-small (p=0.66); opensearch < bge-small −1.7 [−2.8,−0.7]; **lr-dense-websearch ties BM25** +1.5 [−0.1,+3.0] p=0.07; lr-dense < e5-small −2.2 p=0.003; hybrid adds +2.8 over dense (websearch).
- ANN sweep: lookup-query vectors are harder for HNSW — LR default ef loses 2.1 nDCG on FiQA vs bge-small's 0.7; at ef=512 both mostly recover (−0.5 vs −0.2). TREC-COVID shows no visible ANN penalty (n=50).
- OpenSearch doc-side postings (full vocab, 5K-doc sample): 233 nnz/doc mean → ~1.4 GB per 1M docs. Doc-index ladder per 1M docs: bge-small 0.77 GB (384d fp16) < opensearch 1.4 GB < leaf/arctic-m 1.54 GB (768d) < LR 3.07 GB (1536d).

## Codex gate (2026-08-25) — verdict "not decision-grade yet"; all reruns executed

Findings and dispositions:
- BLOCKER 1 projection loader mismatch (trained on ST-wrapper potion, evaluated on model2vec-native vectors): FIXED — re-encoded with StaticModel end-to-end, refit with oracle-λ selected directly on test retrieval (strongest possible shot for the method). Rerun in `results/codex_reruns.log`.
- BLOCKER 2 ±0.007 was 5-ds-derived, ladder overclaims: FIXED — headline reworded as groups; significance rerun on 6-ds with 21 pairs incl. all near-neighbors; p reported as bounds (p<2e-4).
- MAJOR 3 LR sparse/hybrid unresolved reproduction: quarantined with explicit label.
- MAJOR 4 Edge prototype proves query-path composition, not production storage/cold start: reworded; token shard as a default-indexed collection is the wrong storage shape (1.82 GB vs 466 MB raw fp16); load times labeled warm-cache.
- MAJOR 5 ANN behavior measured for one system/corpus only: FIXED — ef sweeps rerun for bge-small (fiqa + trec-covid) and LR (trec-covid) in `results/ann_sweep.json`.
- MAJOR 7 OpenSearch doc-side postings cost unmeasured: FIXED — full-vocab nnz/doc on 5K-doc sample → `results/opensearch_index_cost.json`.
- MINOR 8/9/10 (p=0 as bound, cost definitions, LEAF wording): adopted.
- Ship-scope per Codex: all comparative claims scoped to "the six named datasets"; no generalization to production workloads, million-scale, filtered search, or non-English.

## Standing directive: push for the best model, not a model (Dylan, 2026-08-26)

Added after a session declared a target "unreachable" and was wrong within the hour. **We do not
want A model. We want the best one obtainable under the constraints.** Being first to say "this
projects to a negative result" is not rigour — rigour is exhausting the angles and *then* reporting
honestly. Honest pessimism arrived at lazily is still lazy.

**Before writing that any bar is unreachable, all of these must be true and shown:**

1. **The arithmetic has been redone with the best available component**, not the current one. The
   2026-08-26 case: "Tier 1 unreachable" assumed the current teacher. Swapping in a stronger
   permissive teacher (stella_en_400M_v5, MTEB-Ret 58.97 vs bge-base 53.25) moved the projection
   from 0.406 to 0.489 and cleared the bar outright. Quality is usually a *product* of factors —
   check every factor before declaring the product capped.
2. **Every failing component has been diagnosed, not just observed.** Same day: contrastive
   training was called "broken" when the learning rate was 30-300x above every published recipe
   and the literature has an analytic result (arXiv 2110.09348) for precisely that symptom. A
   failure you cannot explain mechanistically is not evidence about the method — it is evidence
   about your configuration.
3. **The literature has been swept for the specific failure**, in parallel subagents, with the
   real numbers extracted. Cheap, fast, and it has twice overturned a conclusion here.
4. **A capability claim has been checked algebraically before being believed or dismissed.** Some
   "obvious" fixes are provably no-ops (a doc-side linear map reparametrises the table;
   document centering cannot change ranking at all). Do the algebra — and note the example
   originally given here was itself wrong: query-side centering *was* claimed to be genuine new
   capacity, and `results/m7_absorb_check.json` later disproved it to machine precision
   (`mean(W-mu) = mean(W)-mu`, so it is absorbable, as are whitening, top-PC removal and any
   per-token scalar weight). Only n-gram rows and multiplicity-dependent pooling add capacity.
   The lesson stands; this file had the sign of it backwards for a day.
5. **The negative result is reported with what would change it** — the specific measurement,
   component, or bound that would flip the verdict.

Corollary: a pre-registered kill criterion (e.g. the phase-2 contrastive bar) exists to stop
*grinding a diagnosed dead end*, not to license abandoning an undiagnosed one. Kill the avenue
after you understand why it failed, never before.

### Vendor rule, relaxed (Dylan, 2026-08-26)

The original rule — no component from any vendor shipping a competing vector-search product —
disqualified almost every strong encoder and was costing us real quality. **Relaxed: a vendor whose
vector-search offering is far from their main business is acceptable if the choice is heavily
justified in the report.** Direct competitors stay out.

Operationalised so a future session does not have to guess:

- **OUT — vector search *is* the business:** Pinecone, Weaviate, Zilliz/Milvus, Chroma, Vespa,
  Nomic (Atlas), Mixedbread, Jina (and Elastic), MongoDB (Atlas Vector Search is a flagship push;
  also Voyage AI, which MongoDB acquired), Cohere (Embed/Rerank is core product). And obviously
  nothing where we would be shipping a competitor's model as a *component* of a Qdrant release.
- **OK WITH JUSTIFICATION — vector search is one service among hundreds:** Alibaba (gte, Qwen3-Embedding
  — OpenSearch Vector Search Edition / AnalyticDB), Microsoft (e5 — Azure AI Search), IBM (granite —
  watsonx managed Milvus), Google (Vertex AI Vector Search; note Gemma terms still fail the
  *licence* rule independently). **Snowflake (arctic-embed) needs the strongest justification of
  this group** — Cortex Search is built directly on Arctic Embed, so it is the closest of these to
  core business.
- **CLEAN — no vector product at all:** BAAI, NovaSearch, NVIDIA, Salesforce, academic labs,
  individual/community releases (e.g. intfloat).

Unchanged and non-negotiable: the **licence** must permit commercial release of derived weights
(no Gemma terms, no CC-BY-NC), and **table size still binds** — the released artifact is
vocab x dim, so a 250K-vocab model is disqualified on arithmetic regardless of vendor
(250,002 x 1024 fp16 = 512 MB, larger than the 466 MB LightRetriever table we exist to beat).
Practical filter: **vocab <= ~50K and dim <= 1024, or MRL-truncatable**.

The pre-relaxation shortlist (`research/m7-teacher-shortlist.md`) was written under the strict
rule, superseded by `research/m7-teacher-shortlist-2026-08-26.md`, and deleted in the 2026-08-28
cleanup — git history has it. The re-run sweep happened and the teacher question is closed
(stella, see the decision log).

### Past decisions are revisitable (Dylan, 2026-08-26)

**"If we need to revisit any past decisions to make this better, I'm open for it. Achieving our
goal supersedes anything else."** So do not treat an earlier choice as settled just because it is
written down — including choices in `instructions-m7.md` and in the decision log below. Anything
in this list is fair game to reopen with evidence and Dylan's sign-off: the teacher, the
architecture, the objective, the data mix, the tier definitions and their comparators, the
milestone scope, the release target, and even structural premises like "frozen off-the-shelf
document tower" or "no transformer at query time" if a better system lies the other side of them.
Bring the arithmetic and the trade-off, not just the idea.

**The one class of exception, and why it is not an exception to the goal.** The evaluation
protocol — partitions, decontamination, the frozen comparator vectors, the single final run,
pre-registered statistics — exists to make a good number *believable*. Relaxing it after seeing
results does not achieve the goal, it destroys the thing the goal is for: an unbelievable 0.50 is
worth less than a defensible 0.46. So protocol changes are allowed, but only **before** the
numbers they would affect are observed, and only written down with the reasoning. Never
retroactively, never silently. Same rule for the licensing and vendor constraints: they are
commercial reality, so reopening them is Dylan's call and needs an explicit answer, not an
inference.

## Long runs must be watched, not launched and hoped for (Dylan, 2026-08-26)

Added after a session lost most of a day to jobs that were running but wrong. Four failures in one
morning: a probe that thrashed the allocator for 50 minutes per component, a mining pass costing
3.6 hours instead of 3 minutes, four screen arms crashing on one shape error, and a pool whose
width check would have silently overwritten 9.5 GB. Every one was visible within two minutes of
starting, and none announced itself.

**Before launching anything that runs longer than ~10 minutes:**

1. **Smoke the code path first.** `sweep.smoke(base, over)` runs an arm at 90 steps. Prefer a path
   with no execution history — that is where the bug is. A grid's arms share code, so
   `sweep.grid(..., fail_fast=True)` (the default) stops at the first arm that *raises*; an OOM
   does not trip it, being a real per-arm resource result.
2. **Arm a monitor with the failure signatures, not just the success line.** Silence is not
   success: grep an alternation covering `Traceback|Error|FAILED|OOM|Killed|assert` alongside the
   progress marker. A monitor that only matches the happy path is indistinguishable from a
   crashloop.
3. **Read the first progress line and sanity-check the RATE against an estimate** before walking
   away. "mine 2048/349934 (76s)" is a three-hour job stated in the units of a fast one. Two
   minutes of arithmetic at launch is worth hours later.
4. **Watch the machine, not only the log.** 100% GPU utilisation with ~1% memory-bandwidth
   utilisation and low power draw means allocator thrash or launch-bound tiny kernels, not work.
   `/proc/<pid>/stat` utime against elapsed says whether a "GPU job" is actually burning one CPU
   core; per-thread CPU time says whether it is Python or a kernel. `py-spy` needs
   `kernel.yama.ptrace_scope=0`, which needs root on this box — so these cheaper signals matter.
5. **Never trust a docstring's cost estimate.** Both "a few minutes for the whole set" claims in
   `train.py` were wrong by two orders of magnitude, in the same file, on the same day.

## Durable knowledge goes in the repo, not in the assistant's memory (Dylan, 2026-08-26)

The point of this repo is a reusable harness, so anything a future session would need belongs in a
file it will read: protocol in `m7/LEDGER.md`, dead ends in `m7/EXPLORED.md`, module facts and
pitfalls in `m7/CODEMAP.md`, standing directives here. Assistant memory is for *Dylan-and-workflow*
facts only (how he wants sessions run, what tooling exists on the box), and anything in it that
would help a future session must be mirrored into one of those files. A lesson that lives only in a
memory file is lost to every session that does not happen to recall it.

## Verification gates (Dylan, 2026-08-24)

Results dictate Qdrant engineering decisions: correct, not decimal-precise; blind spots stated openly.
- M2 gate: adversarial Opus review of methodology + code + M2 numbers → `research/verification-m2.md`. (running)
- Pre-report gate: Codex second opinion on the full result set + report draft, briefed for pushback; findings reported verbatim.
- Significance: paired bootstrap on key system pairs before the report; deltas within noise get labeled as such.
- **Codex CLI is installed on the box** (`/usr/local/bin/codex`, 0.149.1). Invoke it read-only and
  with high reasoning effort, since the default profile here is effort `none`:
  `codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort="high" - < brief.md`. Read-only
  matters: a previous review committed to the files the session was editing.
- **STANDING GRANT, Dylan 2026-08-28 — supersedes the earlier "sparingly, at milestones only".**
  *"Feel free to do adversarial reviews with Fable and Codex as much as needed. Use them to push
  you to get the best result possible. They've proven valuable every time."* So reviews are a
  **routine instrument, not a milestone ceremony**: brief one whenever a decision is about to
  become expensive or irreversible, whenever a document is about to become permanent, and whenever
  a result is surprising enough that you want it attacked before you believe it. The 2026-08-28
  record is why — between them the two reviews caught a rounded CI endpoint in `final_run.py` (the
  one irreversible decision in the project), a gate that returned 0 on NO-GO, a release guard that
  failed open, a confounded diagnostic that had licensed a four-hour lever, and a retention figure
  belonging to a reverted candidate. **Brief them adversarially**: give them the numbers, name what
  you believe, and ask them to break it. A review told "confirm this" returns nothing.

## Key decisions (log)

- **WITHDRAWN THE SAME DAY, on evidence: `arctic-embed-l` is worse than the teacher we already
  have.** Ranked by the closed-form table distilled from it — the artifact that ships — arctic is
  −0.0480 [−0.0608, −0.0349] below bge-base, and a teacher's own retrieval quality turns out not to
  predict its distilled table at all (Spearman 0.000 over eight candidates). Only
  **stella_en_400M_v5** beats the incumbent (+0.0365 [0.0249, 0.0481]). The teacher question is back
  with Dylan because stella's disclosed training data covers 2 of our 6 eval datasets;
  `m7/LEDGER.md` pre-registers a four-dataset primary comparison as the answer to that.
  `results/m7_learnability_report.json`. The entry below records what was decided and why, and is
  kept because the failure mode — selecting a teacher on the tower instead of on the table — is the
  lesson.
- **[SUPERSEDED THE SAME DAY — see the entry above; the teacher is stella_en_400M_v5.]**
  **Teacher for M7 was `Snowflake/snowflake-arctic-embed-l` (Dylan, 2026-08-26).** Chosen on
  measurement, not projection: best of five candidates on the two CQADupStack dev components
  (+0.0447 [0.0339, 0.0557] over bge-base; arctic > stella +0.0125 [0.0008, 0.0241] raw, which
  would NOT survive multiplicity over the ten pairs, so the top is arctic ~= stella), Apache-2.0,
  and the only candidate whose MTEB registry entry discloses **zero overlap with our six** — stella
  lists ArguAna and FiQA2018, 2 of the 6. Dylan ruled on the vendor question explicitly, because the
  released table only works against its teacher's document vectors, so the doc side of a Qdrant
  release would be Snowflake's model. `results/m7_teacher_probe.json`,
  `results/m7_teacher_contamination.json`. The projection that had ranked stella first is not merely
  imprecise on this evidence, it is wrongly ordered.

- BEIR subset: SciFact, NFCorpus, FiQA-2018, ArguAna, SciDocs (100,785 docs total; all appear in LightRetriever's tables and on MTEB → like-for-like comparison possible).
- Harness: hand-rolled — HF `datasets` (BeIR/* repos) + `pytrec-eval-terrier` + numpy brute force. `beir`/`mteb` packages skipped (their value is model wrappers we don't use). Title+text join: `(title + " " + text).strip()`. Python 3.12 venv, torch 2.13 MPS.
- LightRetriever config: `lightretriever/lightretriever-qwen2.5-1.5b` (ungated, 3.1 GB bf16, fits 24 GB RAM). Paper BEIR-15: dense 48.9 / sparse 47.3 / hybrid 52.1. MPS gotchas: use `sdpa` not flash-attn, explicit bf16, no autocast.
- Llama-based MRL adapter (dimension truncation) is gated on HF — skipped unless Dylan requests access. Mistral-7B variant ships a reference lookup table to validate our table construction against.

## Findings (log, M2)

- Harness validation (corrected per verification B2): 50 of 58 overlapping cells within 0.001 of official MTEB; 10 of 12 configs within 0.0005 on the 5-ds average. The ArguAna gap was self-hits: BEIR drops doc_id==query_id (`ignore_identical_ids`); harness now does too. Query prefixes stay on everywhere (empirically validated; an earlier prefix diagnosis was wrong, reverted).
- Verification B1 (fixed): transformers 5.x loads checkpoints in their config dtype — granite ran bf16, gte fp16, non-comparable. dtype now pinned fp32 everywhere; both models' artifacts deleted for re-encode; caches carry meta.json (dtype/prefix/max_seq) and refuse stale reuse.
- Verification B2 (fixed): potion models must be encoded with the native model2vec loader — it reproduces official FiQA to 5 decimals (0.187609 vs 0.18761); the sentence-transformers wrapper deviates +0.0027. Both potions re-encoded.
- Verification B3 (adopted): every reported delta gets a bootstrap CI; ±0.007 resolution stated once in the report; sub-resolution orderings reported as ties. LEAF asym-vs-sym per-dataset signs disagree with the LEAF paper on 3/5 (ArguAna reversal significant, −1.07 [−1.91,−0.22]) → resolved: our composition is byte-identical to MongoDB/mdbr-leaf-ir-asym (sha256-verified: query tower, Dense 384→768, teacher doc tower, prompts). The ArguAna reversal is a genuine finding on this subset, reported as such.
- Verification M1/M2 (adopted): 5-ds subset correlates with BEIR-15 at Spearman only 0.55 → adding TREC-COVID (171K docs; balances the subset: LightRetriever's best category) + BM25 baseline (done, 0.379). All "X beats Y" claims scoped to the named datasets in the report.
- Verification N7 (adopted): report doc-side costs too — LightRetriever's 1536-dim index is 4x bge-small's 384-dim per doc; index bytes/doc + encoding throughput go in the frontier table.
- Remaining from verification: M9 sparse-mask ablation on SciFact (paper's reference doesn't mask padding; we do), M5 headline = single websearch table (per-task tables reported as oracle upper bound), MI8 ArguAna truncation caveat.
- OpenSearch doc-v3-gte (2026-08-25): validated against its card ≤0.004 on 4/5 datasets; the card's SCIDOCS 0.455 is wrong (measured 0.1686, consistent with sibling models — use ours). 5-ds avg 0.4375: **best zero-query-compute system on this subset**, above LR hybrid per-task (0.4225) and LR hybrid websearch (0.4144), with a 133M doc encoder and Qdrant-native sparse output.
- LR sparse reproduction (2026-08-25): our sparse runs −3 to −5 nDCG under the paper (scifact 0.631 vs 0.664). Tested and excluded: padding mask before amax (masked 0.6306 vs unmasked 0.6272 — not the cause), doc-token-restricted pooling flags (all default False in their args). Cause not isolated; capped per "correct, not decimal" — dense reproduces faithfully, sparse/hybrid reported as "our reproduction, conservative under-estimate" with this caveat. Hybrid per-task 0.4225 (paper 0.4374); hybrid-websearch 0.4144. int8 table is quality-free (0.4117 vs 0.4114) and halves the fp16 table.
- M3 root-cause fix (2026-08-25): first LR dense run was −4pt avg vs paper (ArguAna −11.8). Cause: lookup tables built without `<|bos|>`. The shipped adapter tokenizer bakes `<|bos|>…<|endoftext|>` into add_special_tokens=True (base Qwen adds nothing), so the reference's bos check fires — table rows must be `[bos]+prompt+[tok]+[eos]`. Partial-table A/B (diag_bos.py): scifact 0.642→0.663 (paper 0.665), arguana 0.394→0.518 (0.512), scidocs 0.147→0.176 (0.181). Reproduction faithful after fix. No-bos tables archived in `artifacts/…/tables-nobos/`. Doc encoding was always correct (post-processor adds bos/eos there).
- 2026-08-24 machine OOM incident: full-vocab fp32 logits in the sparse doc encoder + two concurrent MPS jobs exhausted 24 GB RAM (macOS killed jobs, Dylan saw the out-of-memory dialog). Fixed: sparse projection restricted to per-dataset query-token columns, batch cap 32, MPS watermarks (HIGH 0.7 / LOW 0.5 — both must be set, high-only crashes), strictly sequential jobs. Docker Desktop holds ~17 GB for Dylan's own containers; not touched.

## Findings (log, M1)

- LightRetriever lookup table ≠ input embedding matrix: each vocab token is forwarded through the full trained model ([bos]+instruction+token+[eos], take EOS hidden state). Using the raw embedding matrix instead costs −11.2 BEIR (paper ablation A2). Tables are per-instruction.
- Paper headline numbers are hybrid (dense+sparse); dense-only (what the pure two-collection edge architecture gives) is 2–3 pts lower. But LightRetriever's sparse query side is also zero-compute (token counts) → hybrid on Qdrant Edge is plausible.
- Inference-free sparse SOTA: `opensearch-neural-sparse-encoding-doc-v3-gte` (133M doc encoder) BEIR-13 avg 0.546; SPLADE-v3-doc 0.517. BM42 disqualified (query-side attention). miniCOIL lacks comparable BEIR numbers and needs per-token multi-vectors.
- LEAF asymmetric verified: docs via teacher `Snowflake/snowflake-arctic-embed-m-v1.5` (109M), queries via `MongoDB/mdbr-leaf-ir` (23M) = 54.03 BEIR-en (beats symmetric-either). Packaged as `MongoDB/mdbr-leaf-ir-asym`.
- Best small symmetric (MTEB v1 BEIR-en): arctic-embed-s 51.98, bge-small-en-v1.5 51.68, granite-small-english-r2 50.9. Best static: potion-retrieval-32M 35.06 MTEB-Ret.
- Literature: no LightRetriever successor; ScalingNote (industry) 29M query tower keeps 99% R@50 vs 7B teacher; DeepMind LIMIT gives formal single-vector ceiling; "static model + linear map into frozen big-doc space" appears unpublished → candidate original experiment.
- ArguAna queries avg 193 words (counter-argument retrieval) — stress case for bag-of-tokens query encoders; expect LightRetriever to lose there.
