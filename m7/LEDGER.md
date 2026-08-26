# M7 protocol ledger

The load-bearing record: partitions, licence evidence, every six-set access, decontamination
counts, gate results, freeze record, incidents. Detail lives in `results/m7_*.json` and is
pointed at, never restated.

> **Compacted three times on 2026-08-26** — after the gate, after the research session, and after
> the Codex gate. Every protocol-required fact is kept verbatim; settled justification is one line
> each and review finding-lists are counts plus dispositions. It sits at ~5.1K tokens against a ~4K
> budget, and the honest reason is that two whole sections are live rather than settled: the Codex
> gate's OPEN list is a to-do list, and Stage 0 / the GO correction still carry the projection the
> current plan is trying to beat. **Next compaction: retire the Stage-0 and GO sections once the
> arctic-teacher numbers replace them, not by shaving prose.** Full narrative in
> `git log -p m7/LEDGER.md`.

## Environment

- Box: RTX 3080, **10 GB VRAM**, 25 GB RAM (peak budget 18 GB), 16 cores, ext4, nvcc 12.6.
- Stack: Python 3.12.14, torch 2.8.0+cu126, transformers 4.57.6, datasets 5.0.1,
  pytrec-eval-terrier 0.5.10, qdrant-edge-py 0.8.0, Qdrant server v1.19.0. Lock:
  `m7/requirements.lock.txt`.
- Teacher: **BAAI/bge-base-en-v1.5 @ a5beb1e3e68b9ab74eb54cfd186867f64f240e1a**.
- Doc-encode dtype: **fp16 for dev + training, fp32 compute for the final run; fp16 at rest
  everywhere**, matching the M4 convention the frozen comparators were produced under. Evidence:
  cosine 1.000000 vs fp32 on 10K docs, |Δ nDCG| ≤ 3e-4 on both CQADupStack components.

## Verification

- `scripts/validate_perquery.py` OK, 54 cells (4 allowlisted per FINAL_MATRIX.md).
- `scripts/verify_manifest.py`: all six datasets re-downloaded and hash-matched to
  `results/eval_manifest.json`; `results/frozen_eval/` matched the fresh download. Frozen
  comparator pairing is valid.
- **SIX-SET ACCESS, class (a) harness validation** (2026-08-25, `results/m7_harness_validation.json`):
  bge-small ArguAna 0.6038 (want 0.6034), SciFact 0.7127 (0.0000), bm25 FiQA 0.2532 (−0.0000).
  All within 0.003. No new-model number was scored against six-set qrels in this access.
- Conformance suite 30/30 (`m7src/test_conformance.py`), including the real save→load→encode path.

## Partitions

**TRAIN** — approved sources only (`research/m7-data-licensing.md`). Final count after all
decontamination: **349,934 pairs** + 221,395 query-text-only rows for objective B. Per-source
fields, rights, positive construction and counts: `results/m7_field_table.md`.

**DEV** (pinned; hashes in `results/m7_dev_manifest.json`, frozen before any candidate result):
nq-250k 250,000/3,452 · hotpotqa 5,233,329/7,405 · cqadup-programmers 32,176/876 ·
cqadup-physics 38,316/1,039 · heldout-train (corpus = the full 6,169,142-doc pool)/7,325 ·
heldout-longq (same corpus)/**55**. Banned: Touché (args.me is ArguAna's source family), Quora
(no licence). BM25 and potion have no row on the two held-out slices (their corpora are pool row
indices carrying no document text), so those comparisons run on the four text-backed components.

**KNOWN-TEST** — the six, development-informed. Content pinned by `results/eval_manifest.json` +
`results/frozen_eval/`.

**UNTOUCHED-FINAL** — BEIR FEVER and DBpedia-entity. **Climate-FEVER dropped: fails the
affirmative-licence standard** (no statement at climatefever.ai, in arXiv:2012.00614 incl.
appendices, or the GitHub repo; only HF mirrors assert CC-BY-SA-4.0, and a wrapper tag is not
evidence here — the same rule that excluded Quora).

### Source-level licence evidence (eval-use standard)
- **NQ** CC BY-SA 3.0 — first-party but **not on the live README**: declared in merged PR #11
  (2019-06-10, commit `c307fa7030`) and silently dropped Aug 2019. Cite the commit. Repo LICENSE
  is Apache-2.0, code only.
- **HotpotQA** CC BY-SA 4.0, dataset and Wikipedia corpus, hotpotqa.github.io.
- **CQADupStack** CC BY-SA 3.0, verbatim in the ADCS 2015 paper (the 2014 Stack Exchange dump,
  predating the 2024 no-LLM-training clickwrap). Eval-only here. HF wrapper tags contradict each
  other (BeIR cc-by-sa-4.0 vs mteb apache-2.0) — why tags aren't evidence.
- **FEVER** CC BY-SA, fever.ai's own licence page.
- **ESCI** Apache 2.0 at repo root. Caveat: unanswered issue #21 asks whether it covers the data.
- **MIRACL, Mr. TyDi** Apache 2.0, LICENSE files confirmed.
- **DBpedia-entity** test collection MIT (iai-group/DBpedia-Entity); abstracts CC BY-SA 3.0 + GFDL.
- **BEIR itself is not a licence authority** — its Apache-2.0 covers packaging only.

## Decontamination

Rules (narrowed deliberately; reasoning kept because it is part of the protocol):
- **R1 remove** on query overlap, all partitions. Query overlap is the leakage that decides scores.
- **R2 remove** on positive-document overlap with the six — the contamination map enforced at
  fingerprint level rather than by source name.
- **R3 measure and disclose, do not remove**, for DEV and UNTOUCHED-FINAL documents. Removal there
  would delete the sources rather than decontaminate them (`hotpotqa-corpus` *is* the dev HotpotQA
  corpus), i.e. forbid training on Wikipedia while evaluating on Wikipedia benchmarks. What removal
  protects — test queries and qrels — is enforced by R1 and the final-scorer ledger, and every M4
  comparator has the same property, so the comparison stays like-for-like.

Method: blake2b-64 word hashes, polynomial rolling word-8-grams, bottom-32 sketch, ≥8/32 shared
(est. Jaccard ≥ 0.25). Index built over the TRAIN side, protected corpora streamed against it, so
peak RAM is ~0.4 GB regardless of corpus size.

Results (`results/m7_decontam.json`, `..._querytext.json`, `..._heldout.json`):
- R1: 1,329 pairs. Plus nq-open −213, TriviaQA −155.
- R2: 45 pairs, from 23 of 855,324 unique positives — **3e-05** against the six. The source-level
  map was already doing the work.
- TRAIN↔held-out: **2,211 further pairs** (fever-train 1,847 — FEVER contains many near-identical
  claims that straddled the mod-50 split). Without this pass, `heldout-train` would have scored
  models on paraphrases of their own training queries.
- R3 overlap: six 3e-05 · cqadupstack-dev ~0 · nq-250k-dev 0.46% · **DBpedia-entity 9.32%** ·
  **FEVER 11.3%**. The two untouched-final sets are the two most overlapped — both are Wikipedia,
  and so is most of TRAIN.

**Consequence: the untouched-final partition has no clean member.** Both rows are reported with
their overlap rate attached; neither is presented as an uncontaminated generalisation number.

Held-out slice rule: mod-50 applied at **query** granularity, not pair — strictly stronger than
the mandate's literal wording (per-pair holdout would leave a held-out query's text in TRAIN via
its other positives). Disclose: `heldout-train` is a *seen-document/unseen-query* slice (SQuAD
gives ~5 questions per context), so it rewards document-anchored memorisation during dev selection.

## Stage 0

**0.1 closed-form ridge** (`results/m7_stage0_ridge.json`, full suite): the global optimum of flat
distillation under squared loss. 571,329 TRAIN queries, vocab coverage 0.895, best lambda=1e-2 —
an **interior** optimum, so the binding constraint is representational, not statistical. `train_cos`
0.9110 is in-sample; the honest figure is overlap@10 **0.490** vs teacher 0.5722, i.e. half the
teacher's top-10 is not recovered. At lambda=10 the rows barely move and score ~0.20, so the
teacher-derived init alone is a poor table, useful only as a regularisation anchor. **Codex 2026-08-26:
"structural upper bound" is unearned — it bounds that penalised-MSE problem, not objective B or
retrieval.**

**0.2 capacity probe** — PASS at ~1.0000, d=+0.5917, and **near-vacuous**: 23.4M parameters against
~3,500 dev queries. It falsifies only "good retrieval is inexpressible here", which it does
decisively: the frozen-tower tax is a **generalisation** gap, not an expressivity limit. The
load-bearing Stage-0 evidence is the ridge probe.

**Objective grid** (curves in `m7/RESULTS.md`): distillation works; contrastive was destructive from
two initialisations; `reg_init` exonerated. Both named suspects are bounded small
(`results/m7_diag_scores.json`) but were measured in the *teacher's* score geometry, and the suspect
list never included Adam-on-sparse-rows dynamics or cross-query row interference. The learning rate
was the leading untested hypothesis; the phase-2 screen is its test.

**This contradicts a mandate premise** — `instructions-m7.md` says large negative pools are nearly
free, "exploit that first". Scale without hardness wasted the objective, so phase 2 carries the
mandate's full BM25-mined / teacher-mined / mixed comparison (`train.mine_bm25_negatives`, mined
within each query's own doc store).

## GO/NO-GO GATE: **GO** (2026-08-26 03:03)

`results/m7_gate_p1-objB.json`. Full six-component dev suite; BM25/potion on the four text-backed.

All four conditions PASS (G1 Stage-0 > potion, G2 capacity probe > BM25, G3 candidate > BM25,
G4 int8 equivalence within a 0.005 bar); per-condition deltas and CIs are in the JSON. G2 is
**near-vacuous** for the reason given above. Text-backed candidate macro **0.4795** vs teacher
0.6106 → **retention 0.7853** text-backed / 0.8073 all six, agreeing with the ridge probe's 0.777
from a different method. `p1-objC` fails G3 as expected.

### The GO is one component wide (correction, 2026-08-26, post-review)

G3's per-dataset breakdown was in the committed JSON from the start and in none of the prose until
an adversarial review flagged it: **nq-250k +0.1445, cqadup-physics +0.0152, cqadup-programmers
−0.0203, hotpotqa −0.0316 (CI-resolved loss)**. The +0.0270 macro is carried entirely by nq-250k — the component whose query distribution is most
represented in TRAIN (86K nq-open rows feed objective B, guarded only by an R1 near-dup test that
degenerates to exact match under 8 words). The candidate **loses to BM25 CI-resolved on
HotpotQA** and directionally on cqadup-programmers, i.e. it wins on the in-distribution component
and loses on the ones most like the six.

**Projection to the six: ~0.41** = 0.785 text-backed retention x a plausible bge-base six-set row
of ~0.52 (bge-small measured 0.5042). That is below BM25's 0.4174 on the six and far below the
0.4583 release bar; substituting the 64% cqadup retention for FiQA-like sets lowers it further.
**On today's evidence the best candidate projects to Tier 4.** The gate answers "is the program
alive" (yes); it does not support "on track", and nothing in the record claimed otherwise before
this entry, which is the omission.

**Checkpoint substitution, logged in advance:** the driver named `p1-objC` before any result
existed; gating a knowingly-inferior checkpoint would be a false negative, so `p1-objB` was also
gated and both are reported. Selecting on dev is within the protocol — the gate is a dev-stage
decision.

## Strategy pivot (Dylan, 2026-08-26)

Trigger: the corrected projection put the best candidate at ~0.41 on the six (Tier 4). Direction:
research properly, revise the plan, aim for Tier 1, restart the model work if needed. What is
explicitly NOT what went wrong, and survives any model-side restart: the eval protocol, this
partition ledger, decontamination, the pinned dev suite, the frozen comparator vectors, the
freeze/final-run machinery, and every review finding. Only architecture and training recipe change.
The live plan is `m7/STATUS.md`.

## Incidents

- **2026-08-25 ~23:10 WSL OOM (self-inflicted).** Three memory-heavy jobs concurrently hit 24 of
  25 GB; kernel killed a process and WSL went down. Repeated the M4 lesson already in CLAUDE.md.
  Nothing scored yet, so no results lost; encode caches survived (shard-resumable). Fixes:
  decontamination rewritten TRAIN-side-indexed, pool index per-store and lazy, `encode_cached`
  returns a memmap, hashes streamed, jobs strictly sequential via the `run_stage0*.sh` drivers.
  Peak-RAM budget now explicit at 18 GB. Recurring-mistake list: `m7/CODEMAP.md`.
- **2026-08-26 05:52 reboot — Windows Update, not a crash.** Event 1074 `TrustedInstaller.exe`,
  reason "Operating System: Upgrade"; last shutdown success true, no bugcheck/thermal/power event.
  Box was idle; nothing lost. **Host action for Dylan: stop Windows Update rebooting mid-run.**
- A cosmetic bug crashed the gate's *printer* after its JSON was written (G4 is an equivalence
  bound with no `ci95` key). Verdicts never at risk; each field is now guarded independently.

## Protocol decisions, 2026-08-26 research session (logged BEFORE the numbers they affect)

- **The single-anchor MTEB→six projection is withdrawn.** Its one anchor (bge-small, ratio 0.976)
  is the 3rd-highest of the nine models we have measured both ways, so it biased every teacher
  estimate high. `results/m7_calibration.json` is authoritative: ratio spread 0.926–1.001,
  affine r=0.950, **residual sd 0.0102**. Retention Tier 1 (0.4868) would demand: 95.8% bge-base,
  94.2% bge-large, 88.9% gte-large, 87.5% stella.
- **Teachers are ranked by measurement, not by that projection**, whose residual exceeds the
  1.06-MTEB (~0.009) gap between the top two candidates. `m7src/teacher_probe.py` measures each
  candidate's symmetric ceiling on the **two CQADupStack dev components**: the only dev components
  on no candidate's disclosed training list (the registry lists NQ, HotpotQA, FEVER, MSMARCO,
  ArguAna, FiQA2018 as in-domain for stella — 2 of our 4 text-backed dev components), and already
  flagged here as the nearest dev analogue to FiQA. Dev-only; not a gate input; the six are unread.
- **Disclosure obligation if stella becomes the teacher:** ArguAna and FiQA2018 are recorded as its
  in-domain training data (MTEB registry, community-maintained, not an author disclosure — the card
  promised training details and never delivered). That is **2 of the 6 final eval datasets**, and
  our comparators do not carry the same flag. Must be labelled at the dataset row, not buried.
- **Fusion (dense table + BM25) selected on dev for checkpoint `p1-objB`.** One family, one
  parameter, no per-dataset weights or routing, at `fusion.DEPTH`=1000 for selection and
  application alike, fitted against the **int8** table because that is the released artifact. This
  is the sanctioned dev-stage selection. **If the checkpoint changes, the fusion must be
  re-selected** — a parameter frozen on one checkpoint is not valid for another.

## Codex gate, 2026-08-26 (gpt-5.6-sol, read-only, high effort) — 6 BLOCKER / 9 MAJOR / 2 MINOR

Full text: `research/m7-codex-gate-2026-08-26.md`. Its own "fix before any more compute" was the
teacher-swap boundary. Dispositions, honestly labelled:

**FIXED this session** (detail in `git log`, verbatim findings in
`research/m7-codex-gate-2026-08-26.md`)
- B1 teacher-swap boundary: pool width/identity, init-cache keying, and `teacher_rows`' hardcoded
  CLS read-out — all three would have crossed the arctic swap silently. `test_init_rows.py` is the
  standing check.
- B4 final scoring not bound to the frozen teacher: `FREEZE.json` now carries the full Spec
  fingerprint, `load_and_verify` refuses per field, `test_freeze_guard.py` covers the
  same-repo/different-pooling case no hash can catch.
- M-screen: the screen could not isolate the lr; redesigned to A-only arms from one checkpoint.

**OPEN, and each one blocks a specific later step, not the current compute**
- B2 **decontamination covers positives only (~855K docs) while training touches the full 6.17M-doc
  pool** as random/mined/KL negatives. "TRAIN↔KNOWN-TEST decontaminated" is therefore false for the
  actual training surface. Blocks the release claim, not the teacher swap. Fix = fingerprint every
  pool row eligible as a negative against the six, then mask matches from bank, mining and KL sets.
- B3 **the bootstrap p-values are percentile tail probabilities, not null-distribution p-values**, so
  Holm does not control family error over them. Blocks every confirmatory claim. Fix = paired
  label-swap randomisation test for the macro statistic, type-I error verified by simulation, with
  percentile/BCa intervals kept for *intervals* only.
- B5 **the frozen fusion function differs between dev selection and final scoring** (selection drops
  BM25 `score <= 0`, final keeps them; convex fusion min-max normalises over what is returned, so
  the minimum and every normalised score move). The Tier-1 system would not be the function selected
  on dev. Fix = one shared run builder, asserted byte-identical across cached/uncached/selection/
  final paths.
- B6 **the "two six-set accesses" rule is already breached**: `bench_throughput.py` called
  `load_beir("fiqa")`, which parses FiQA test qrels, and that was neither logged harness validation
  nor the final run. Recorded here as the required ledger entry. The rule is convention-based, not
  enforced — any script can read committed plaintext qrels without `final_run.py` noticing — and the
  report must say so rather than claim enforcement.
- M-calibration **the prose PI half-widths disagree with the JSON** (recomputation gives 0.02818 /
  0.03294 / 0.03446 vs the 0.024 / 0.030 / 0.035 in STATUS), and **"bge-base cannot clear Tier 2 at
  any retention" is false** — its teacher-only lower bound clears above ~0.955 retention. Also, the
  table macro is `mean_i(r_i x teacher_i)`, not `dev_ratio x mean_i(teacher_i)`, so multiplying a
  dev-macro retention by projected teacher PI endpoints does not compose the two uncertainties.
- M-probe **the probe ranks symmetric tower quality, not table learnability**, on two subforums of
  one family, taking the max of five candidates with no selection correction (winner's curse). The
  proper form is a cheap closed-form table per candidate, ranked on held-out dev. Our arctic choice
  additionally rests on contamination evidence, which is independent of this critique.
- M-probe-cache **probe cache files are keyed on name+tag only** — no revision, corpus hash, prompt,
  pooling, Dense, dtype, or remote-code commit — so a pre-Dense-fix stella cache would be reused.
  Also true that `trust_remote_code` weights revisions do not pin the remote code.
- M-perquery **`validate_perquery.py` validates each vector's MEAN**, so a permutation of scores
  across qids passes while destroying the pairing every CI depends on; `boot._align` intersects
  silently and nDCG drops missing queries. Blocks trust in the frozen comparators.
- M-decontam-short **the 8-word fingerprint rule degenerates to normalised exact match for short
  queries**, which is the dominant NQ/FEVER regime and exactly where the dev win is training-adjacent.
- M-ridge **"structural upper bound" is unearned**: the ridge solves penalised unnormalised MSE at a
  dev-selected lambda, while objective B is normalised cosine + KL and the endpoint is retrieval.
  Claim must be restricted to that MSE problem.
- M-stella-ship, MINOR-int8-weights (the released int8 artifact still multiplies an unbounded fp32
  weight vector; fold weights into rows before quantisation and re-run G4), MINOR-doc-transform
  (the absorbability algebra omits re-normalisation): recorded, not yet actioned.

## Phase-2 screen redesign, 2026-08-26 (logged before any arm's A-phase result was read)

The screen's arms were objective C at a matched step budget across a 60x lr range, which cannot
isolate the contrastive lr because the B phase runs at that same lr and is nowhere near converged at
the low end (B reaches 0.2731 at 4k steps at lr 5e-5 vs 0.4449 at 3e-3). Arms are now **objective A
only from one fixed p1-objB checkpoint** (init `run:p1-objB`, restoring rows from `rows_fp16` and the
trained token weights), varying only the contrastive lr, with a zero-step arm pinning the start.
That zero-step arm reproduces **0.4548 exactly**, so the checkpoint path is faithful.
`CONTRASTIVE_KILL_BAR` and `KILL_REQUIRES` are unchanged and still satisfied by the 1e-5/5e-5/1e-4
arms. Codex reached the same conclusion about the old design independently.

Also recorded: the **collapse diagnostics must be read against the init, not against zero.** The
untrained teacher-init table sits at mean pairwise cos 0.954 / effective rank 21.8 and dev 0.0061;
the 0.4548 table sits at 0.342 / 270.4. A session watching those numbers mid-training will otherwise
read the starting geometry as a collapse, as this one did for an hour.

## Teacher selection criterion changed, 2026-08-26 (logged BEFORE any six-set access)

The teacher was to be chosen by measured **symmetric ceiling** (`m7_teacher_probe.json`). That
criterion is now refuted: Spearman(ceiling, distilled-table) = 0.000 over eight candidates, and
arctic-embed-l — approved on the ceiling — produces a table 0.0480 BELOW the incumbent's,
CI-resolved. **The criterion is now the closed-form distilled table's dev score**
(`m7_learnability_report.json`), which is the artifact that ships. Dev-only; the six stay unread.

Consequence: **Dylan's arctic-embed-l ruling is withdrawn on evidence, not overruled** — the
question goes back to him because the only candidate that beats the incumbent is stella, whose
disclosed training data includes ArguAna and FiQA2018, 2 of the 6 confirmatory datasets.

If stella is chosen, the option pre-registered here (and legal only because it is written before any
six-set number is observed): make the **primary** comparison the four datasets with no known teacher
exposure — SciFact, NFCorpus, SCIDOCS, TREC-COVID — recomputing every comparator on the same four
from the frozen per-query vectors in `results/perquery.json`, and report the six-set number as
secondary with the exposure labelled at the dataset row. Codex's M-stella-ship says labelling alone
does not remove the bias, and this is the answer to it. The tier bars are defined on the six, so a
four-set primary claim needs its own bars computed the same way, from the same frozen vectors.

