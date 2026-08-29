# Codex adversarial gate, 2026-08-26

`codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort="high"`, briefed for pushback and
given read-only access to the whole repo at commit `1bcf99b`-ish (the brief is reproduced at the end
of this file). Verbatim findings, unedited. Dispositions live in `m7/LEDGER.md` under "Codex gate";
do not restate them here.

## BLOCKER

BLOCKER | The planned teacher swap cannot produce a trustworthy training run.

where: `m7src/pool.py:27,80-90`; `m7src/train.py:259`; `m7src/init_table.py:23-36,49-64`

why it is wrong or unearned: The pool is hard-coded to 768 dimensions and its cache metadata omits model/revision. Training calls `poolmod.build()` without the active dimension, so 1024-d stella/gte runs either fail or reuse the old bge-base pool; same-dimensional swaps silently reuse the wrong teacher. Init caches likewise omit teacher identity. Worse, `teacher_rows()` always CLS-pools and omits `Spec.post_dense`, so stella’s mandated mean→Dense pipeline is not used. Random init also defaults to 768.

what would settle it: Before any corpus encode, run an end-to-end smoke test for every candidate asserting pool/table/query/target shapes equal `Spec.dim`, pool metadata matches the exact Spec fingerprint, and token-init rows match `teacher.encode()` on singleton-token sequences. Cache paths must change when any encoder identity field changes.

BLOCKER | Decontamination excludes only positive documents, while millions of documents actually used for training are unchecked.

where: `m7src/decontam.py:263-287`; `m7src/pool.py:73-83`; `m7src/train.py:318-321,393-405,417-430`

why it is wrong or unearned: R2 indexes roughly 855K positive documents, but objectives A and B/KL train against the full 6.17M-document pool as random, mined, and provided negatives. Protected six-set documents can therefore participate in gradients without ever entering the contamination index. “TRAIN↔KNOWN-TEST decontaminated” is false for the actual training surface.

what would settle it: Fingerprint every pool row eligible as a negative against the six and remove or mask all matches from banks, mining, and KL candidates. At minimum, prove by ID and fingerprint that no six-set relevant document occurs anywhere in the training pool.

BLOCKER | The confirmatory “p-values” are not null-distribution p-values, so Holm does not control family error.

where: `m7src/boot.py:29-63,83-94`; `m7src/final_run.py:201-218`

why it is wrong or unearned: Replicates are drawn around the observed paired difference, then `mean(delta* <= 0)` is called a p-value. That is a percentile-bootstrap tail probability, not a bootstrap under H0; it is not guaranteed super-uniform, which Holm requires. Zero hits are also reported as `p<1/B`, although their Monte Carlo uncertainty permits a larger p.

what would settle it: Use an exact paired label-swap/randomization test or a properly centered null bootstrap for the macro statistic, verify type-I error by simulation, then apply Holm to those p-values. Keep percentile/BCa intervals separate from testing.

BLOCKER | Final scoring is not bound to the teacher frozen with the artifact.

where: `m7src/freeze.py:51-58,66-99`; `m7src/final_run.py:31-38,141-156`; `m7src/encoders.py:132-137`

why it is wrong or unearned: `FREEZE.json` records teacher and revision, but verification never compares them with the process’s active `M7_ENCODER`. Final document vectors, teacher queries, and the table tokenizer come from the environment-selected encoder. A stella table can therefore be evaluated accidentally against default bge-base while every hash guard passes.

what would settle it: Freeze the complete `Spec` fingerprint and require it to equal `teacher.SPEC` before any label access. A dry-run test setting the wrong `M7_ENCODER` against a frozen manifest must refuse.

BLOCKER | The frozen fusion function changes between dev selection and the final run.

where: `m7src/select_fusion.py:28-52`; `m7src/final_run.py:129-138`; `m7src/fusion.py:27-38`

why it is wrong or unearned: Dev selection removes BM25 results with `score <= 0`; final scoring retains them. Convex fusion min-max normalizes over the returned scores, so including zero-padding changes the minimum and therefore every normalized positive score. The Tier-1 system is not the function selected on dev.

what would settle it: Use one shared BM25 run builder and assert byte-identical run dictionaries and fused per-query scores between cached, uncached, selection, and final paths on every dev component.

BLOCKER | The hard “only two six-set accesses” rule has already been violated without a ledger entry.

where: `m7src/bench_throughput.py:13-17`; `bench/core.py:21-28`; `m7/LEDGER.md:28-34`; `results/m7_throughput.json`

why it is wrong or unearned: The executed throughput benchmark calls `load_beir("fiqa")`, which loads and parses FiQA test qrels even though it only needs documents. This is neither logged harness validation nor the final run. More generally, committed plaintext qrels can be read or scored by any script without `final_run.py` noticing; its ledger marker enforces only cooperative use of that script.

what would settle it: Record the incident explicitly. For an enforceable one-shot claim, keep labels outside the working repo behind an audited scorer. Otherwise the report must concede that access was convention-based and already breached.

## MAJOR

MAJOR | Multiplying a dev-macro retention ratio by a projected six-set teacher macro is not a table-score prediction interval.

where: `results/m7_calibration.json` keys `retention_today_text_backed`, `candidates.*.at_retention.*.pi95_teacher_only`; `m7/STATUS.md:21-40`

why it is wrong or unearned: The table macro is `mean_i(r_i × teacher_i)`, not `dev_macro_ratio × mean_i(teacher_i)`. Component retention ranges widely and covaries with domain and teacher score. Treating retention as fixed hides both its sampling uncertainty and dataset-transfer variance; multiplying the teacher-only PI endpoints does not compose those uncertainties. The prose table also disagrees with JSON: recomputation gives half-widths 0.02818 for bge-base, 0.03294 for gte-large, and 0.03446 for stella—not 0.024/0.030/0.035. “bge-base cannot clear Tier 2 at any retention” is false: its teacher-only lower bound clears at retention above about 0.955.

what would settle it: Estimate per-component joint teacher/table outcomes on independent representative dev datasets and bootstrap the entire projection—including teacher calibration, candidate selection, and retention—at dataset and query levels. Until then these are scenarios, not clearance probabilities.

MAJOR | The hotpotqa-excluded `+0.049` is post-hoc selection, not a defensible transfer estimate.

where: `results/m7_fusion_p1-objB.json` key `grid`; `results/m7_fusion_report_p1-objB.json` keys `per_component`, `macro_gain_fused_over_dense`; `m7/STATUS.md:42-57`

why it is wrong or unearned: The fusion family and weight were selected on these same four components, then the component with the largest observed gain was excluded after inspecting the breakdown. The remaining mean has neither a pre-registered target population nor selection-adjusted uncertainty. CQADup similarity to FiQA does not establish transfer to all six.

what would settle it: Select the fusion on one dev split and estimate gain once on disjoint components, or use nested/leave-one-component-out selection with a dataset-level interval.

MAJOR | The teacher probe optimizes symmetric tower quality, not the quantity that determines the shipped system.

where: `m7src/teacher_probe.py:1-18,68-105`; `m7/STATUS.md:17-19`

why it is wrong or unearned: The final score depends on teacher ceiling multiplied by table learnability/retention. A tower can score higher symmetrically while being less additively predictable by a token table. Two subforums from one dataset family cannot establish six-domain superiority, and choosing the maximum of five candidates without selection correction adds winner’s curse.

what would settle it: For each candidate, fit the same cheap closed-form or short distilled table and rank the resulting student on held-out dev, with multiplicity-adjusted comparisons. Symmetric ceiling should be reported only as a secondary diagnostic.

MAJOR | The live teacher probe’s cache and remote-code provenance are not reproducible.

where: `m7src/teacher_probe.py:48-65`; `m7src/teacher.py:72-91,166-188`; `m7/CODEMAP.md:119-120`

why it is wrong or unearned: Probe caches are named only by candidate name and component tag; they omit revision, corpus hash, prompt, pooling, Dense head, configuration, dtype, and code version. A cache created before the stella Dense fix would be reused. For `trust_remote_code` models, the weights revision also does not pin separately hosted remote code, and that code identity is absent from validation and encode keys.

what would settle it: Key probe files on the full Spec, corpus hash, dtype, and remote-code commit; verify stored metadata before reuse; pin or vendor remote code.

MAJOR | Shipping stella would make two of six confirmatory rows training-set evaluations.

where: `m7src/encoders.py:101-109`; `m7/LEDGER.md:267-276`; `m7/STATUS.md:35-40`

why it is wrong or unearned: Recorded stella training exposure includes ArguAna and FiQA2018. Labelling those rows does not remove their upward bias or make the six-set macro comparable to systems without the same known exposure. The result can still describe this benchmark, but it cannot credibly establish zero-shot release quality.

what would settle it: Obtain auditable corpus/dedup evidence that the benchmark examples were absent, choose a teacher without known exposure, or make a pre-registered clean partition excluding contaminated rows the primary comparison.

MAJOR | The planned contrastive screen cannot isolate learning rate, and its kill guard is not enforcement.

where: `m7src/program.py:20-53,68-106`; `m7src/train.py:445-499`; `m7src/train.py:328-331,468`; `m7/EXPLORED.md:16-17`

why it is wrong or unearned: Every “sane” C arm retrains phase B using its overridden learning rate, warmup, and weight LR; it does not load one common B checkpoint. Arms are then compared with an 8K-step B-only bar after only 4K B + 2K A, confounding starting quality and training budget. `may_invoke_contrastive_kill` has no call site and checks only that a qualifying configuration exists—not its score or whether another arm passed. Training also samples globally rather than dataset-aware, while dense Adam continues momentum updates on intermittently touched rows. The diagnostics in JSON are in teacher/random-negative geometry and cannot exonerate these student/mined-negative mechanisms.

what would settle it: Load one byte-identical B checkpoint for every A-only arm; vary one factor at a time; log start/end scores, row drift, student score geometry, and source composition. The kill function must consume committed results and permit closure only if every qualifying arm fails the frozen bar.

MAJOR | Frozen comparator validation does not validate the pairing on which every CI depends.

where: `scripts/validate_perquery.py:20-28`; `scripts/dump_perquery.py:39-43`; `m7src/boot.py:17-26`; `m7src/evalkit.py:47-49`

why it is wrong or unearned: Validation checks only each vector’s mean against `quality.json`; an arbitrary permutation of scores across qids passes while destroying paired inference. Bootstrap alignment silently intersects datasets and qids, and nDCG evaluation silently drops missing queries. The vectors are also rounded to six decimals despite the mandate specifying unrounded scores.

what would settle it: Assert exact dataset/qid equality at every comparison and validate every per-query score against a frozen run/qrels artifact, not its mean. Missing queries must be fatal, not intersected away.

MAJOR | The fingerprint rule does not provide the claimed near-duplicate protection for short queries.

where: `m7src/decontam.py:73-96,171-218`; `m7/LEDGER.md:242-244`

why it is wrong or unearned: Texts shorter than eight words receive one whole-query gram, so “near duplicate” degenerates to normalized exact match for the dominant NQ/FEVER-style regime. Minor insertions, deletions, paraphrases, and reordered words evade R1. This is precisely where the dev win is training-adjacent.

what would settle it: Measure recall on a blinded mutation/paraphrase set stratified by query length, then use shorter word/character shingles or semantic candidate retrieval with manual adjudication until a predeclared recall bar is met.

MAJOR | The “closed-form structural upper bound” is not an upper bound on objective B or retrieval.

where: `m7src/stage0_ridge.py:1-15,51-68`; `m7src/ridge_full_eval.py:35-45`; `instructions-m7.md:50-52`

why it is wrong or unearned: The solve optimizes regularized, unnormalized squared error for a dev-selected λ. Objective B is normalized cosine plus ranking KL, and the reported endpoint is retrieval after normalization. Another flat table can have worse squared training loss but better nDCG, so “no training run can beat it at that objective” and “structural upper bound” are unearned.

what would settle it: Restrict the claim to the exact penalized MSE problem, or optimize/bound the actual normalized cosine+KL objective and demonstrate retrieval saturation independently.

## MINOR

MINOR | The released “int8 table alone” still depends on an unbounded fp32 weight vector.

where: `m7src/table.py:71-99,183-206`; `instructions-m7.md:41-44`

why it is wrong or unearned: Softplus is positive but not bounded. `save_table()` quantizes rows while storing token weights in fp32, and runtime multiplies both. That is not literally a single int8 vocab×dim lookup artifact, and large learned weights can change quantization behavior when absorbed.

what would settle it: Report the learned-weight range and fold weights into rows before int8 quantization; rerun the G4 equivalence test on that actual single-table artifact.

MINOR | The absolute document-transform algebra in the prose omits re-normalization.

where: `m7/LEDGER.md:182-186`; `CLAUDE.md:89-95`

why it is wrong or unearned: `q·Ad=(Aᵀq)·d` and the constant-shift argument hold if transformed documents are left unnormalized. With the project’s normalized-embedding retrieval, `Ad/||Ad||` or `(d−μ)/||d−μ||` introduces a document-dependent denominator and can change rankings.

what would settle it: State the algebra with normalization explicit and test both normalized and unnormalized document transforms.

## IF I HAD TO PICK ONE THING TO FIX BEFORE ANY MORE COMPUTE IS SPENT

Fix and test the teacher-swap boundary end to end: one immutable Spec fingerprint must determine dimensions, pooling/Dense, tokenizer, every init, every encode cache, the document pool, and the final freeze. Right now the expensive stella/gte path can reuse bge artifacts or encode the wrong model, making all subsequent compute worthless.

---

## The brief it was given

    You are running an ADVERSARIAL REVIEW of a research repo at a milestone, before any training or
    final evaluation is redone. Be hostile to the work. Your job is to find what is wrong, unearned,
    or unbelievable — not to summarise, not to praise, and not to propose a rewrite.
    
    HARD CONSTRAINTS
    - READ ONLY. Do not edit, create, or delete any file in the repo. Do not git commit, stash, or
      checkout. If you want a change, describe it in your findings.
    - Do NOT run anything GPU-bound, training, or a corpus encode. A GPU job is running right now
      (teacher_probe.py) and a second one has previously taken this machine down. Cheap CPU commands
      (git log, grep, python arithmetic over the committed results/*.json) are fine.
    
    CONTEXT, read in this order
    - CLAUDE.md — the project, the standing directives, and the decision log.
    - instructions-m7.md — the binding mandate for the current milestone: comparators, tiers, eval
      protocol, ops.
    - m7/STATUS.md, m7/LEDGER.md (protocol), m7/EXPLORED.md (closed avenues), m7/RESULTS.md,
      m7/CODEMAP.md (module map).
    - results/m7_*.json — every committed number. Prose in the .md files must never restate a number
      these files hold, so treat the JSON as authoritative and the prose as a claim to be checked.
    - m7src/ — the code. bench/ and scripts/ are the reused M1-M6 harness.
    
    WHAT THE PROJECT IS TRYING TO DO
    Ship a query encoder with NO transformer at query time: a vocab x dim lookup table (tokenize ->
    look up rows -> average -> normalize) distilled against a frozen off-the-shelf document tower. The
    release bar ("Tier 2") is a CI-resolved win over LightRetriever-dense-pertask 0.4583 on six named
    BEIR datasets, achieved by the DENSE TABLE ALONE. An easier "Tier 1" bar (0.4868, OpenSearch
    inference-free doc encoder) permits labelled fusion with BM25.
    
    ATTACK THESE SPECIFICALLY. For each, say whether the claim is earned, and if not, what measurement
    or argument would settle it.
    
    1. CAN THE RELEASE BAR BE EARNED AT ALL, honestly? m7/STATUS.md carries a teacher-projection
       prediction interval (results/m7_calibration.json) and a retention figure (~0.7853). Re-derive
       the arithmetic yourself from the JSON. Is the interval constructed correctly (regression sigma,
       df, extrapolation widening)? Is applying a dev-macro retention factor to a projected six-set
       score a valid composition of two uncertainties, or is it a category error that hides variance?
       Is the fusion "defensible transfer estimate" of +0.049 (hotpotqa-excluded) defensible, or is it
       still selection on dev?
    
    2. EVAL PROTOCOL INTEGRITY. This is the thing that makes a good number believable, and it is the
       one part the project says may not be relaxed after seeing results. Check: the TRAIN / dev /
       six-set / untouched-final partitions; the decontamination (m7src/decontam*.py — is the
       fingerprint rule strong enough, is the index built on the right side, what does it miss?); the
       frozen comparator per-query vectors (results/perquery.json, validated by
       scripts/validate_perquery.py, with four allowlisted cells); the pre-registered statistics
       (m7src/boot.py paired bootstrap, one-sided tests, Holm at family alpha 0.025); and
       m7src/final_run.py's guarantee that the final run happens once. Where could a number be
       selected on the test suite without anyone noticing?
    
    3. THE CODE THAT WILL PRODUCE THE HEADLINE NUMBER. Read for correctness, not style:
       m7src/table.py (the released artifact, its ONE frozen preprocessing rule, int8 symmetric
       per-row absmax, unseen-token policy), m7src/evalkit.py (chunked brute force, per-query nDCG,
       two-axis tiling), m7src/pool.py, m7src/train.py, m7src/fusion.py, m7src/gate.py. A bug here is
       worth more than any of the arguments above.
    
    4. THE ALGEBRAIC CLAIMS. results/m7_absorb_check.json asserts that query-side centering,
       whitening, top-PC removal and per-token scalar weighting are ALL absorbable into the table and
       therefore add no capacity, while n-gram rows and multiplicity-dependent pooling do add capacity.
       Check the algebra independently. The project already had the sign of one of these backwards for
       a day, so do not take it on trust.
    
    5. THE TEACHER DECISION, which is live right now. m7src/teacher_probe.py ranks candidate document
       towers by measured retrieval on two CQADupStack dev components, because the MTEB->six-set
       projection cannot separate the front-runners. Is that a sound basis for a choice that costs a
       full corpus re-encode? The leading candidate (NovaSearch/stella_en_400M_v5) lists ArguAna and
       FiQA2018 on its recorded training data, and those are 2 of our 6 final eval datasets — what
       does shipping it do to the credibility of the final number, and is the proposed labelling
       enough? Also check m7src/encoders.py + m7src/teacher.py for loader fidelity (encode cache key
       coverage, pooling, post-pooling Dense, config overrides) — results/m7_encoder_validation.json
       claims all five candidates reproduce sentence-transformers.
    
    6. THE CONTRASTIVE FAILURE. Training on the contrastive objective degraded the table. The current
       leading hypothesis is a learning rate 30-300x above published recipes, and
       m7src/program.py:phase2_screen is the planned decisive test, with a kill criterion enforced in
       code (program.may_invoke_contrastive_kill). Is that screen actually decisive, or is it
       confounded? Are there suspects the diagnosis has never considered
       (results/m7_diag_scores.json bounds two of them, in the TEACHER's geometry, not the student's)?
    
    7. ANYTHING IN THE PROSE THAT OVERCLAIMS relative to the JSON, and anything in EXPLORED.md that
       was closed on weaker evidence than it states.
    
    OUTPUT FORMAT
    Findings only, ordered BLOCKER, then MAJOR, then MINOR. For each:
      SEVERITY | one-line claim
      where: file:line (or the JSON key / the .md line)
      why it is wrong or unearned: the mechanism, concretely
      what would settle it: the specific measurement, command, or algebra
    Then a final section: "IF I HAD TO PICK ONE THING TO FIX BEFORE ANY MORE COMPUTE IS SPENT".
    Do not pad. A short list of real findings beats a long list of style notes. If you believe
    something is fine, say nothing about it.
