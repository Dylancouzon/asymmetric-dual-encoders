# Asymmetric Dual Encoders — Viability Scoping

## North star

Decide, with defensible numbers, whether asymmetric dual encoders (big document encoder in the cloud, near-zero-compute query encoder on the edge) beat the obvious alternative: running a small 20–100M conventional embedding model on the edge. The research question: **how much retrieval quality do you retain as query-side computation approaches zero?**

Architecture: the server indexes documents once with a large frozen encoder (**stella_en_400M_v5**, 1024d, since M7); the edge client holds the document index plus a query path. Two query paths exist: **zero** (M7, a token→vector lookup table, no transformer) and **nano** (M9/M10, a ≤35M distilled transformer). Both serve the SAME index — the product is the pair, two points on a quality-vs-query-cost frontier.

## Deliverable

A decision report (Artifact) with a quality-vs-query-side-cost frontier: nDCG@10 on the six named datasets for each point, plus edge costs (query asset, document index, hydration, CPU latency), a Qdrant Edge prototype of the pair, and the released models. Every headline claim is paired on frozen comparator vectors with pre-registered statistics; the miss-is-publishable framing applies to every milestone.

## Constraints

- Mac: Apple M5 Pro, 24 GB RAM, MPS — probes and CPU cost rows only. The Windows/RTX 3080 box ran M7–M9 and holds their gitignored `work/` artifacts; **M10 runs on a rented cloud GPU under an approved budget or not at all (Dylan, 2026-09-01; `instructions-m10.md` §Compute)**. Headless either way: Dylan follows on GitHub.
- Quality numbers come from exact search so ANN recall is not a confound; Qdrant Edge is for architecture and latency.
- Dylan is hands-off; decisions get made here and logged. The evaluation protocol (partitions, decontamination, frozen comparators, single final run, pre-registered statistics) may change only BEFORE the numbers it affects are observed, never retroactively.

## Stage plan

- [x] M0–M6 (2026-08-24/25): environment, research (`research/landscape.md`, `literature.md`, `methodology.md`, `sparse-inference-free.md`, `lightretriever.md`), the 21-system × 6-dataset matrix (`results/FINAL_MATRIX.md`, costs, significance), the two-collection Edge prototype, and the M6 report artifact (https://claude.ai/code/artifact/db771dd1-2d59-4c34-9d6e-70dd4d337d16). Headline numbers, rerun outcomes, the M2 Codex gate and the M1/M2 findings logs are archived verbatim in `research/m1-m6-findings.md`.
- [x] M7 (closed 2026-08-29): **zero**, the int8 lookup table distilled from stella. Final run avg-6 **0.4339** vs `LR-dense-pertask` 0.4583 (CI-resolved miss); fused with BM25 0.4911, a statistical tie with OpenSearch 0.4868. Frozen and verified releasable (`m7/FREEZE.json`); **published 2026-09-03, private** — https://huggingface.co/DylanCouzon/zero-query-encoder-v1, see `m11/STATUS.md`. Mandate `instructions-m7.md`, detail `m7/STATUS.md`, transferable lessons `m7/FINDINGS.md`. Teacher decision: stella_en_400M_v5, chosen on the distilled TABLE, not the tower (see Key decisions).
- [x] M8 (closed 2026-08-30 as a measurement): twelve probes, no lever moved the table more than 0.005; no confirmatory access spent. `m8/FINDINGS.md`, `m8/EXPLORED.md`.
- [x] M9 (closed 2026-09-01 as a measurement, six-set close-out pending): **nano**, bge-small distilled into stella's query space by L2 regression. Build stopped by the plateau rule at 3.74B tokens: SCREEN-3 0.5606 = **82.2% retention**, but **93.8% on NQ against 50–71% on the two CQADupStack components** — a coverage failure, compounded by a 384-wide linear head that L2 regression cannot push past ~90–93% once queries are diverse (`m10/PLANNING.md` §9); not a parameter-count failure. Aim (leaf-ir-asym 0.5155) and release bar (bge-small 0.5042) both need ≥88% on avg-6; projected miss. **Read `m9/FINDINGS.md`.** Mandate `instructions-m9.md`; recipe `m9/M92_LOCK.md`; runs `m9/RESULTS.md`; closed avenues `m9/EXPLORED.md`. The registered six-set transaction (`m9/FINAL_LOCK.md`) is executed on the frozen candidate as M9's close-out measurement only after M10's recipe lock is pushed (so it cannot inform an M10 decision), six-only (its reserved conditional struck by a ratified amendment), and once Dylan ratifies it; LoTTE stays unread for M10.
- [ ] M10 (**PAUSED 2026-09-03 pending the cloud GPU budget below** — Dylan is on another project until it lands; resume from `m10/STATUS.md`. Retry of nano, planned 2026-09-01): same pair, same bars, new recipe built around the coverage finding: millions of synthetic queries in 12 forms seeded from Wikipedia and the approved corpora (FineWeb ruled out of M10), a wider linear output from pooling three layers, LEAF's small-batch cyclic schedule, a registered ranking-aware phase-2 loss, warm start from the M9 candidate. **Compute: one rented A100 under a $1,000 ceiling (expected $400–715, up to $895 if generation is hosted), or M10 does not run; LEAF's dose of 200M examples.** Mandate `instructions-m10.md`; evidence `m10/PLANNING.md`; status `m10/STATUS.md`.
- [ ] M11 (**deliverable 1 half-done**: `zero` is released, `nano` waits on M10 — `m11/STATUS.md`): release the pair, ONNX port including the document model, fastembed integration, whitepaper. `instructions-m11.md` (was M10 before the 2026-09-01 renumbering).
- [ ] M12 (noted, not scoped — Dylan, 2026-08-30): an IMAGE model. Edge workloads are largely vision, so the asymmetric premise has a bigger market there. Open: whether the query side is a lookup table at all, image→image vs text→image, which frozen document tower. Do not inherit the text architecture's assumptions; scope it when M11 lands.
- [ ] M14 (**parked, long-term, may never run** — Dylan, 2026-09-03): a better `zero`. Idea register only, no compute committed: `instructions-m14.md`. The ranked lead is document-side co-adaptation (M8's `E14-LORA`, authorised and unrun), which reopens the frozen-tower premise and costs the drop-in-to-stella's-index property. The cheap lead is cosine-space distillation, priced at ~+0.035 by a pyNIFE retention comparison on fiqa (they retain 73.4% of a weaker teacher against our 67.1%).

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

## Markdown files must be TIGHT (Dylan, 2026-08-29)

*"Information is wayyyy too verbose in there. We're diluting context for next session."* Said of an
M8 ledger that had reached 2,694 lines, ~40% of it one session's amendment prose.

**Every line in a `.md` file costs a future session context it could have spent on the work.** A
file a session is *told* to read before deciding — `CLAUDE.md`, `STATUS.md`, `LEDGER.md` — is loaded
whether or not the reader needs the paragraph you enjoyed writing. Verbosity there is not
thoroughness; it is a tax levied on every session that follows.

**The rule:** write the decision, the number a rule reads, and the pointer. Nothing else.

- **One fact, one home.** Numbers live in the result JSON, bars in `registry.json`, runs in
  `RESULTS.md`, closed avenues in `EXPLORED.md`, long-form reviews in `research/*`. A `.md` that
  restates any of them is duplication that will go stale in exactly one direction — the wrong one.
- **An amendment is: what changed, why, and the pointer.** Not the reasoning that got you there.
  If the reasoning matters, it belongs in the archived review, cited by path.
- **Prefer a table to prose, a clause to a sentence, a pointer to a summary.** Cut every sentence
  that only restates the previous one with more emphasis.
- **Withdrawn claims and owner rulings are the exception — always keep them**, because a future
  session that re-derives a withdrawn claim wastes far more than the lines cost. Keep them *short*.
- **When you add to a long file, budget for it**: if an entry runs past ~10 lines, compress an old
  one or move detail out. Files grow by default; only deliberate effort shrinks them.
- **Check the size when you touch it.** `wc -l` on the file you just edited. If a protocol file has
  grown past ~1,500 lines, compressing it is part of the task, not a separate one.

## Verification gates (Dylan, 2026-08-24)

Results dictate Qdrant engineering decisions: correct, not decimal-precise; blind spots stated openly.
- M2 gate: adversarial Opus review of methodology + code + M2 numbers → `research/verification-m2.md` (done).
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
- **EVERY brief must carry a READ-EXCLUSION, and the log must be audited afterwards (2026-08-29
  incident).** An external reviewer is a separate process with ordinary read access to the whole
  repo, so `paths_guard`/G2 — an in-process Python bulkhead — is structurally incapable of
  constraining it. A repo-wide grep dumped two RESERVED confirmatory sets, queries **and qrels**,
  complete, into a reviewer's context. Nothing was scored and no decision read them, but the
  reviewer's recommendations were thereby potentially informed by held-out data. So: state in the
  brief that `results/frozen_eval/untouched-*`, the reserved qrels caches and `work/m9reserve` must
  not be read; prefer naming the files to read over inviting a repo-wide search; and **grep the
  review log for reserved-set reads before reading its findings.** Quarantine anything that draws
  on them.

## Key decisions (log)

- **Teacher = `NovaSearch/stella_en_400M_v5` (2026-08-26), chosen on the distilled table, not the tower.** `arctic-embed-l` had been picked the same morning on its own dev retrieval quality (+0.0447 over bge-base) and was **WITHDRAWN the same day**: ranked by the table distilled from it, arctic sits −0.0480 [−0.0608, −0.0349] below bge-base, and a teacher's own quality does not predict its table (Spearman 0.000 over eight candidates). Only stella beat the incumbent (+0.0365 [0.0249, 0.0481]). Stella discloses ArguAna and FiQA (2 of the six) and FEVER (1 of the reserved four) in its training data; every stella-based claim carries that qualification and the NDO-4 rows. `results/m7_learnability_report.json`, `results/m7_teacher_contamination.json`. Kept because the failure mode — selecting a teacher on the tower — is the lesson. M9 re-confirmed it for towers: stella-1.5B distils WORSE than stella-400M (−0.0023 at equal dose).
- **Clean data stack (2026-08-25):** MS MARCO permanently excluded from anything released (non-commercial terms; priced at +0.0058 [−0.0015, +0.0131] on the six — not what the miss is made of). CC BY-SA sources (NQ, SQuAD, HotpotQA, FEVER, MIRACL, Mr.TyDi) approved with model-card attribution. FineWeb was approved 2026-08-30 as document-side regression text but is **out of M10 in every role** (ruled 2026-09-01 under delegated authority: no reserved-set document fingerprints exist, and seeding from it would add a rights review for no measured gain). `research/m7-data-licensing.md`.
- **Reserved four** (FEVER, DBpedia-entity, cqadup-android, cqadup-english): one confirmatory access, unspent through M9. **`results/perquery.json` is irreplaceable** — frozen comparator vectors regenerated from caches that no longer exist. Never overwrite it.
- **`zero` published private, 2026-09-03 (Dylan's ask).** The M7 artifact unchanged, on Dylan's own HF account, so the architecture can be tested as a PoC. Model card leads with the miss and the stella-contamination caveat. **Licence: MIT, ruled by Dylan 2026-09-03 and valid for a public release** (matching stella), with CC BY-SA attribution for NQ/SQuAD/HotpotQA/FEVER/Mr.TyDi. `m11/STATUS.md`.
- **Renumbering 2026-09-01 (Dylan):** M10 becomes the nano retry; release/port/whitepaper moves to M11; the image model to M12.
- **Student cap 35M, hard (Dylan, 2026-09-01):** "109M is not an option. This isn't low compute anymore. 33M was already in the upper bound of what I think is acceptable." No larger student in any role; the cheaper student gets the benefit of a tie.
- **M10 compute (Dylan, 2026-09-01):** "M10 won't be done on a 3080. M10 will be done on a GPU budget, if allowed, or not at all." The box is not an execution target for M10; doses and screen sizes are set from the budget, not the box (`m10/PLANNING.md` §5–6 and the §8 amendment block). Budget approval is decision 2; refusal closes M10 unstarted.
- **M9 close-out (2026-09-01):** LoTTE read #1 withdrawn unexecuted so M10 inherits a fresh surface; the six-set transaction runs six-only (reserved conditional struck, ratification pending) after M10's recipe lock; the `m9-status` branch deleted. M10 execution work happens on branch `m10-work` under the headless commit-and-push contract.
- Harness, BEIR subset, LightRetriever reproduction details and the M1/M2 findings: `research/m1-m6-findings.md`.
