# M17 ledger

## 2026-09-11 — owner scope

Dylan requested a new branch and **planning only** for Constella Zero v1.1: better vocabulary,
especially S3/k8s, modest performance improvement, a future 72-hour training allowance, the same
document tower, and simple approaches. Small checks on the local RTX 3080 are authorized.
Research uses Luna agents, at most four concurrently (this session uses three plus the parent).
Branch: `m17-zero-v1.1-planning`, based on `3006eab`.

This opens M17 planning; it does not reopen M7/M8 decisions, authorize a long training run,
change release policy or grant access to M13's evaluation surfaces. Existing rules remain.
No `CLAUDE.md` exception has been requested or adopted.

## P0 — local feasibility checks, recorded before observation

Purpose: establish tokenization and resource facts useful for planning. No model selection,
retrieval-quality measurement, corpus download, real-data training or checkpoint export.

- Read the local M11 bundle, M7 freeze and training-checkpoint metadata; preserve their bytes.
- Use only hand-authored technical terms and boundary/repetition fixtures. Inspect original
  segmentation and an in-memory `AddedToken(single_word=True, normalized=True)` extension.
- Time the released CPU query path at two batch sizes after warm-up. These fixtures are not a
  production latency distribution and this Linux box is not the Mac reference edge device.
- Check compositional sum initialization on unique and repeated/shared-piece fixtures under the
  existing sqrt-count pooling. Do not infer semantic accuracy from vector agreement.
- Measure two synthetic cached-target GPU training shapes: 128 and 256 queries, 64 candidates,
  24 tokens/query, 33,594 rows by 1,024 dimensions, fp32 table/Adam and fp16 document vectors.
  A short warm-up plus 20 measured steps per shape is allowed. Synthetic loss updates disposable
  random rows only; rates exclude data preparation, teacher encoding, mining and evaluation.
- Outputs: `results/m17_planning_probe.json`; reproducible diagnostic in
  `m17src/planning_probe.py`. Stop on error/OOM rather than alter a real recipe.

Excluded reads: `results/frozen_eval/untouched-*`, reserved qrels caches, `work/m9reserve`, and
six-set/LoTTE evaluation payloads. No evaluation or training driver is invoked. Research access
logs are recorded in each `research/m17-*-2026-09-11.md` note.

The future experiment design and decision thresholds will be explicitly marked **draft**;
planning measurements do not ratify a training protocol or establish a quality gain.

## 2026-09-11 — scope clarification and P0b

Dylan confirmed the local RTX 3080 and requested broad query-domain coverage in addition to
cloud/software terms. Nano/M13 remains unchanged. A read-only local tokenizer comparison in
response to Dylan's question found Nano's cached bge-small vocabulary map equal to Zero v1's.

P0's repetition fixtures did not actually share constituent IDs across different lexical units
(`3` and `##3` are distinct). They cannot establish general sum-initialization parity. P0b will
check actual shared-piece fixtures (`s3 s`, `k8s eks`, `kubernetes kubectl`), illustrative terms
from other domains, and persist the Nano tokenizer comparison. No quality inference, data
access or training follows. Output: `results/m17_tokenizer_followup.json`. P0's original code
and observation are preserved in commit `2bca40d`; this is a supplemental diagnostic.

## 2026-09-11 — A1: add the three follow-up ideas to the plan

Dylan: **“These look like great ideas, add them to the plan, commit and push.”** This accepts
alias consistency, compatible late-checkpoint averaging and retaining int8 rows in memory as
planned comparisons. Their inclusion is authorized and does not need to be asked again.

This pre-observation amendment adds one matched alias arm to the unrun screen, fixes a single
checkpoint-averaging window and selection rule, and defines an int8 loader comparison on the
same model bytes. `m17/registry.json` owns the loss/sample/window constants, comparison/read
counts and rebalanced allocation. Total local time and the recovery reserve stay unchanged.
The earlier four-arm draft and idea shortlist remain in git at `fe68910`.

All arms see the same admitted alias views; the new arm alone enables the equivalence loss.
The finalist's endpoint/average choice is fixed before the audit, with its control using the
same form. Loader choice depends on fixture parity and measured RAM/latency, not new quality
selection. No snapshot-window sweep, runtime query expansion or inference ensemble is added.

The session is still **planning only**. No model training, corpus access, quality measurement,
new data rights, teacher/index change, M13 change, release-policy change or publication follows
from this amendment. Input/panel manifests, real timing, implementation reviews and execution
authorization remain future work. Prior diagnostic artifacts retain their original provenance.

## 2026-09-11 — A2: plan review, six recommendations accepted

A Fable review of the plan verified the registry arithmetic (parameter cap, byte sizes,
candidate mixes, alias slots, allocation total, read counts) and confirmed the listwise arm is
new relative to M7's uniform-distractor KL. It raised six concerns; Dylan: **“Go with all your
recommendations.”** Dylan also clarified that the S3/k8s vocabulary is an internal request, not
a headline, and that new rows must stay broad rather than overly specific.

Pre-observation changes, all in `m17/registry.json` and `PLANNING.md`, previous values at `b9d355e`:

1. Screen routing moves to the pinned development suite; the M17 panel is descriptive, with its
   expected standard error registered.
2. The judged panel is built and sealed before the 72-hour clock; a pre-clock row is added.
3. **New-source ruling in principle:** official Kubernetes documentation (CC BY 4.0) is admitted
   as a small technical slice, capped at 10% of training queries; one further CC BY / Apache
   project-documentation source may be named at the manifest stage. Licence evidence must be
   recorded in `research/m7-data-licensing.md` before download. AWS pages remain out.
4. Final dose cut from 16,000 to 6,000 steps; query cap raised to 600k and bank to 262,144
   documents; a train/held-out divergence check is logged.
5. Anchor documented as inherited and inert; averaging snapshots moved to 4,500/5,250/6,000.
6. Candidate entropy recorded at cache build in the B2 format.

Two additions from the same review: an untrained sum-init export V0 read once, and a per-domain
cap of 1,024 new rows so no field dominates the vocabulary.

Dylan then accepted three further ideas, with one exception: a tiny pre-clock end-to-end
rehearsal, a held-out judged alias test set of about 200 pairs, and preferring whole words over
abbreviations when support is thin. **`k8s` is exempt and pinned**: it is a direct CTO request
and receives a row regardless of support, with that support recorded honestly.

Dylan then asked for a full Codex Astra review of the plan. Its five P1, eight P2 and one P3
findings and their dispositions are in `REVIEW.md`; all were adopted at the planning level and
none required an owner ruling beyond the ones already recorded. The Kubernetes source remains a
proposal until its licensing row is completed. Still planning only.

Still planning only. Nothing trained, downloaded, scored or published; no protected access.

## Step 2a — Kubernetes documentation licensing row and acquisition (2026-09-11)

Pre-clock, no GPU, no protected access. The licensing row in `research/m7-data-licensing.md` is
complete, so the source is **admitted for download**; the earlier draft's "code samples Apache
2.0" claim was wrong and is corrected in the row — a recursive listing of the pinned tree
(15,566 paths) holds exactly one licence file, the root CC BY 4.0 `LICENSE`, and `content/en/
examples` carries no separate grant. Licence text and the publisher's own footer statement are
quoted with URLs; the CC BY 4.0 trademark and no-endorsement carve-outs are quoted verbatim.
Revision pinned by `git ls-remote`: `17133089068629ec12ca15c1bdf36a60d2671a74`. Attribution
artifact to ship with derived weights: `research/m17-k8s-attribution.md`. Acquisition-terms
finding: none restricting training — an anonymous `git clone` presents no clickwrap and no
separate download agreement, unlike the 2024 StackExchange dump terms.

Acquired with `git clone --filter=blob:none` then checkout of that SHA into gitignored
`work/m17/sources/kubernetes-website` (submodules not fetched). English `content/en/docs` only
was extracted to `work/m17/sources/k8s_docs_en.jsonl`; counts, bytes and hashes are in
`results/m17_k8s_source_manifest.json`. This admits nothing to training: every document must
still pass the executor's protected screen and the registry's new-source share and per-domain
caps.

Decontamination: screened against the development suite through the approved fingerprint
interface, in `m10src/cov_screen.py::screen`'s direction and at its threshold. Over 5,553,821
dev documents and 12,772 dev queries: 0 exact and 5 near-matched k8s documents (0.302%), all on
the document side; the flagged paths are named in the manifest and must be dropped or held out
before training. One of them is the repository's own `content/en/docs/test.md`. The six, the
reserved four and LoTTE are **deferred, not waived** — every interface reaching them
materializes protected payloads in-process, which pre-clock development may not do; that screen
belongs inside the M17 executor via `protected10.build()` + `protected10.hits`. Recorded as a
blocker in the manifest.

## Step 3 — the one M17 driver, its companions and the synthetic rehearsal (2026-09-11)

Pre-clock, no protected access, no registered observation. `m17src/` now holds `common.py`
(paths, registry/freeze, hashes, the `require_executable` status gate), `cache.py`, `vocab.py`,
`train.py`, `export.py`, `loader_np.py`, `evaluate.py`, `rehearse17.py` and six pytest modules.
Constants are read from `registry.json` rather than restated, so the lock in step 6 binds the
code as well as the plan. Reuse is limited to the primitives the codemap marks safe —
`QueryTable`, ragged bags, `occurrence_weights`, `quantize_int8`, `save_table`, `qfusion` —
and no old driver is invoked: `m7src/train.py` and `sweep.one` are never called.

Refusals implemented, each with a test that breaks the bundle or checkpoint it exists to catch:
a real arm while the registry says `DRAFT_NOT_EXECUTABLE` (only `--rehearsal` bypasses, and it
prints that it did); a folded release offered as the warm start, or a warm start from another
teacher/revision or vocabulary size; a resume across a different tokenizer, table size or step
budget; averaging a wrong snapshot window or mixing runs, tokenizers or shapes; a staged
`model.npz` whose bytes differ from provenance; a drifted `encoder_spec` or `preproc`; an
unsanitised padding setting; a loader that stops reproducing the torch query path; and, by name,
any path containing `frozen_eval/untouched-`, `m9reserve`, reserved qrels or LoTTE. The M17
gates are a separate set: M11's stay bound to `m7/FREEZE.json` and are untouched.

The tiny synthetic rehearsal (`work/m17/rehearsal`, gitignored; record in
`results/m17_rehearsal.json`) ran the production modules end to end on the RTX 3080 in 2.3 s:
vocabulary discovery and selection, count-weighted initialization, the extended tokenizer,
a 112-query candidate cache with its entropy block, 20 VL-A steps with the three step-bound
snapshots and a resume, both export forms through all five gates, loader parity, and evaluation
on synthetic fixtures only. Loader parity was exact (0.0 against a 1e-6 bound). The P0b probe
reproduced the hazard it was written for: cosine 1.0 where the new term shares no piece with the
rest of the query, 0.993 where it does. Fixture-scale constants (minima, K, batch, steps,
snapshot window) are recorded field by field in the result under `scaled_constants`.

This certifies that the modules run and refuse; it certifies nothing about quality, throughput
or the real data path, which is still unwritten. Step 4 (two independent implementation reviews)
and step 5 (measured allocation at two sizes) remain the entry conditions for the lock.

## Step 2c — the judged panel and the held-out alias test, sealed provisional (2026-09-11)

Pre-clock, no GPU, no protected access, nothing scored. `m17src/panel_build.py` and
`m17src/alias_test_build.py` built **553 panel queries in 542 families** across the six declared
domains — cloud-software 120, general 100, legal 100, finance 85, medicine 75,
science-engineering 73 — split by family into **279 selection and 274 audit** queries over a
**14,198-document** declared corpus (550 golds, 12,000 seeded stratified distractors, all 1,648
admitted Kubernetes documents). Manifest, hashes and every limitation:
`results/m17_panel_manifest.json`, marked **PROVISIONAL**. The selection partition is mirrored to
`results/m17_panel_selection.jsonl` (182 KB, under the 2 MB rule); the audit partition's text
stays in `work/m17/panel/audit.jsonl` with only its hash published.

Queries are drawn **held-out only** — `m7src/trainmix.heldout`'s registered mod-50 rule — from
SQuAD, HotpotQA and Mr. TyDi English, whose licences are affirmative in
`research/m7-data-licensing.md`, plus the step-2a Kubernetes slice. Two admitted sources were
deliberately not drawn from and the reasons are in the manifest: **`fever-train`** because FEVER
is one of the reserved four *and* a disclosed stella exposure, so its training claims sit on the
reserved evaluation's own Wikipedia surface (licensing permits it; protocol caution does not —
reversible by an owner ruling, which would also owe those families the executor's protected
screen), and **`esci-us`** because e-commerce product relevance maps to none of the six domains.

Domain labels are **heuristic**: the source map gives `general`/`cloud-software`, and a keyword
classifier over query plus gold text re-labels the four specialist domains. The threshold was
measured before it was chosen — over the 3,576 held-out candidates, thresholds 2/3/4/5/6 yield
{science, medicine, finance, legal} = {142,133,149,215} / {75,75,85,101} / {34,56,38,61} /
{17,35,24,40} / {8,24,12,25}. Threshold 3 was taken as the loosest that still needs a strong term
plus corroboration; four domains therefore sit **short of the ~100 target**, recorded as a gap
rather than filled from the training slice or an unapproved source. The yield table is in the
manifest.

**No judgment was fabricated and no teacher ranking was used as a label.** The 433 non-k8s
queries carry their datasets' own annotator qrels (`DATASET_QRELS`), which judge each dataset's
own corpus and are therefore incomplete on this mixed one — an unjudged relevant distractor
depresses every system's nDCG@10 together. All 120 cloud-software queries are `PENDING_HUMAN`
with candidate documents from a **lexical tf-idf** neighbourhood over titles and lead text:
**360 (query, candidate) rows** in `results/m17_panel_pending_judgments.jsonl`, alongside the
**59 ambiguous alias pairs**, 419 review rows in one sheet with empty `relevant_yes_no`/`judge`
fields. **Who judges the technical queries, and to what instructions, is Dylan's decision**; the
panel cannot be read as a six-domain result until it is made.

The alias test is **200 pairs in 164 families** — 100 Kubernetes, 80 Mr. TyDi, 20 SQuAD; 171
acronym/expansion and 29 alias/canonical; **141 VERIFIED_BY_SOURCE** (the source document states
the equivalence, quoted verbatim in the record's citation, acronym initials checked against the
expansion's words) and **59 PENDING_HUMAN** where the short form has several expansions in the
admitted text and the intended sense is a judgment — for example `PVC` as *persistent volume
claim* or *posterior vegetal cytoplasm*. `work/m17/manifest/alias_test_families.json` carries the
706 family ids and 940 normalized-text keys (the ledger first said 708/942; the file holds 706/940, corrected 2026-09-11 step 4) the training-pair builder must exclude, written in
step 2b's `group_id` convention so the two scripts interoperate without importing each other.

**Ancestry screen** (`work/m17/panel/ancestry_screen.json`), through the approved fingerprint
interface in step 2a's direction — `decontam.query_grams` + `Inverted(...).match`, candidate-side
index, ancestor text streamed — at `min_share=4` rather than 8, because the candidate side here is
query text and an 8-gram vote is unreachable for a six-word question. Streams: M7 TRAIN queries
for all five pair sources (the superset the M9 pool and M10's `m9-pool` segment derive from),
`nqopen`/`triviaqa` query text, the `pseudoq-2000000-0` pool named in the warm start's own run id,
M10 harvest and M10 generated queries — **3.5 million ancestor texts**. Every stream applies the
same mod-50 rule, added after a first pass reported 299 self-matches against squad-train: an
ancestor stream that includes the held-out slice makes every panel query match itself.

Result: **433 exposure-known-exposed, 120 exposure-known-clean, 0 exposure-unknown**, with the
reason recorded per query — 378 have a *training query on the same gold document*, 14 matched
ancestor query text (4 exact, 10 near), 41 are exposed only through the ancestor document pool,
and the 120 Kubernetes queries screened clean against every available manifest. **Sealed is not
unseen, and clean is not unseen either**: stella's own pretraining is not screenable here and is
claimed neither way, and the six, the reserved four and LoTTE stay **deferred to the executor**,
exactly as `results/m17_k8s_source_manifest.json` records.

Registered expected standard error, stated as an assumption because no observation exists: the
paired per-domain nDCG@10 SE is `sd/sqrt(n_families)` with the per-query paired difference's sd
assumed between 0.15 and 0.25 and **families, not queries, as the independent unit** — 0.019-0.032
(cloud-software, 60 selection families) to 0.025-0.042 (medicine, 36). That brackets the registry's
registered 0.02-0.04 and sits far above the screening bands, so the panel stays descriptive and
audit evidence: it routes no arm selection. A single system's own nDCG@10 SE is larger still and
is recorded beside it.

Checks: `.venv/bin/python -m pytest -q m17src` — 137 passed, 24 of them new in
`m17src/test_panel.py`, on synthetic fixtures only. The sealing step refuses a family that
straddles the two partitions, a judged document missing from the declared corpus, and a corpus
whose size has moved under the manifest. `panel.jsonl` is byte-identical across repeated draft
runs at the same seed. Nothing here starts the clock: the manifest is provisional, and resolving
the pending judgments re-seals it with a new panel hash.

## Step 2b — admitted-source support manifest and alias training-pair pool (2026-09-11)

Pre-clock, no GPU, no protected access. `m17src/support_manifest.py` counted every currently
admitted commercial-training source plus the step-2a Kubernetes slice, and `m17src/alias_pairs.py`
built the bulk alias pool; `results/m17_support_manifest.json` carries the numbers and every file
hash, with the group and family artifacts in gitignored `work/m17/manifest/`. Documents are
grouped by exact-normalized text sha; query families are union-find over identical normalized
query text and shared positive document groups, with a 50-query hub cutoff so one multi-hop
corpus cannot collapse into a single family. MS MARCO is on this box in both
`work/train/stores/` and `work/decontam/kept.json`; it is refused by name, validation only.

Measured: 6,165,306 deduplicated documents and 496,327 deduplicated training queries after
decontamination and the mod-50 held-out rule, minus the two queries step 2c holds out. Against
the registered source-to-domain map only two panel domains are populated — **general** (496,327
queries, 6,163,658 documents) and **cloud-software** (0 queries, 1,648 documents). Science and
engineering, medicine, finance and legal are **recorded gaps**: under
`vocabulary_ranking.breadth_completion` the vocabulary outcome cannot reach 'broad' with the
current sources, and closing that needs an owner ruling on a new source, not a manifest change.

Dose rule, applied as `pre_lock_rule` requires and written into the registry's `data` block: at
the registered 6000x256 the general bucket is 2.32 passes and coverage 0.03, but alias pair draws
would be 6.17 passes over 15,573 distinct pairs. The alias share therefore shrinks from 16 to 10
pairs per batch (3.85 passes), the 24 freed slots go to general replay (2.47 passes, re-checked),
coverage is untouched and **steps stay at 6,000** — no reduction was needed. The 10% new-source
ceiling is not the binding constraint on Kubernetes: 1,648 documents would need 93 passes to fill
a 153,600-view share, so the honest allocation is the four-pass figure, 6,592 views, 0.43%.

Two of the four registered alias rules have no material and are reported rather than substituted:
no admitted source ships a paraphrase field, and the admitted Wikipedia-derived datasets carry
passages, not the redirect graph. The pool is therefore the Kubernetes glossary's own `aka`
statements (414 pairs from 11 alias forms) and initial-matched `Expansion (ABBR)` definitions
evidenced in the same admitted document (15,159), each substituted into a carrier sentence so a
pair is two query *views*, not two bare terms. 11,669 ambiguous abbreviations and 122 ambiguous
expansions were dropped rather than expanded globally; ESCI is deliberately not mined. Step 2c's
exclusion file is applied by normalized-text sha, its stated interoperable key, since its family
ids are rooted in its own union-find. A seeded 2% sample, 311 pairs, is in
`results/m17_alias_spotcheck_sample.jsonl` for Dylan's human check; the pool is unusable until
that check happens.

Two measurements that corrected a diagnosis. The first alias pass ran at ~1,950 documents/second,
45 minutes for HotpotQA alone; anchoring the pattern on the parenthesis and slicing a fixed lead
window made it ~886,000/second, and the mined-document count rose rather than fell. FEVER then
mined zero definitions — not an absence but a pre-tokenized store writing `( EP )`, and tolerating
the inner spaces recovered 1,146 documents. Both are recorded in `CODEMAP.md`.

Open for Dylan, neither actioned here: the four unpopulated panel domains, and the fact that the
alias pool is 94% general Wikipedia with only 432 Kubernetes pairs. Checks:
`.venv/bin/python -m pytest -q m17src` — 141 passed, 51 of them new in
`m17src/test_support_manifest.py` and `m17src/test_alias_pairs.py`, synthetic fixtures only.

## A3 — checkpoint 1 rulings (Dylan, 2026-09-11)

Three rulings recorded after steps 2 and 3, before any review or observation:

1. **Panel judging.** Dylan judges the 60 selection-partition cloud-software queries (about 180
   rows of `results/m17_panel_pending_judgments.jsonl`) before the screen; a second engineer
   double-judges 20 of those queries and agreement is reported. The 60 audit-partition queries
   may be judged later without touching the clock. Dylan also judges the 59 ambiguous alias senses.
2. **Alias spot check.** Dylan checks the 311-pair 2% sample. More than 5% wrong tightens the
   abbreviation filter and rebuilds the pool before lock.
3. **Domain map.** Documents from sources mapped to `general` are sub-assigned per document by the
   panel builder's heuristic keyword classifier, same threshold, so the four empty domains can
   populate for vocabulary breadth. No new source. Labels remain heuristic and recorded as such.
   The panel's exclusion of FEVER-train and ESCI is confirmed. The step 4 Astra brief asks whether
   this map could bias term selection or leak panel construction into training.

Registry fields changed: `vocabulary_ranking.domain_assignment`, `data.source_domain_map_note`,
`data.panel_exclusions_confirmed`, `data.panel_judging_plan`, `data.alias_spotcheck_rule`.
Implementation of ruling 3 in `m17src/vocab.py`/`support_manifest.py` is step 4 work.

## Step 4a — ruling A3-3 implemented and the support manifest re-run (2026-09-11)

`support_manifest.document_domain(source, doc_text)` returns the source-map domain except for
`general`-mapped sources, whose documents are classified one at a time by
`panel_build.classify` on the document text only (same `KEYWORDS`, same `MIN_SCORE`); the
document pass records `documents_by_domain` per source and `build()` reports the per-document
roll-up as `per_domain` beside the old `per_domain_source_level`. `vocab.discover` now counts a
term's domains once per distinct supporting document rather than per query occurrence, and the
query record's `domain` is defined as its source document's domain. Queries in the manifest stay
on the source map (the manifest has no query-to-document join for the two query-text-only
sources); a term's domain is decided from documents in `vocab.py`, as the ruling says.
Nine tests added (150 pass).

The real document pass re-ran on the RTX 3080 box (`work/m17/logs/support_manifest_a3.log`).
Per-document counts of deduplicated general-source documents: science-engineering 20,227,
medicine 42,157, finance 10,655, legal 22,782; 6,067,837 stay `general`. All six panel domains
now have supporting documents, so `breadth_completion` is arithmetically reachable; whether three
domains supply 64 selected rows each is decided by vocabulary selection. Every count, family
statistic, dose outcome and text hash in `results/m17_support_manifest.json` is unchanged from
the step-2b record (git `930f198`); only the new domain blocks, `script_sha256` and the
`*.tsv.gz` hashes moved, the latter because gzip stores a write timestamp in its header.
No panel, dev-suite or protected read; no text published.

## Step 4b–4d — Astra implementation review, fixes and artifact regeneration (2026-09-11)

Codex Astra reviewed all of `m17src/` read-only (brief and dispositions in `REVIEW.md`; 29
findings). Fixes landed in two Opus passes under Dylan's ruling of the day, "make sure we don't
over-engineer, this is supposed to be a fairly easy re-training": smallest fix per real bug,
adversarial-only findings dropped and listed. Tests 150 → 206. Commits `e2db77e`, `a2a08b9`.

Four fixes invalidated provisional step-2b/2c artifacts, which were regenerated in one chain
(`work/m17/logs/regen_chain.log`), before any human judgment existed, in this order:
alias test → panel screen and seal → alias pool → support manifest.

- Alias test: 200 pairs, 120 `VERIFIED_BY_SOURCE` / 80 `PENDING_HUMAN` (was 141/59); the 21 new
  pending pairs are extractions that hit the phrase cap or started mid-phrase. The exclusion
  file now carries normalized held-out terms and evidence document groups.
- Panel: same 553 queries and split; the ancestry screen is now bound to the panel and alias file
  hashes; new manifest hash. The pending sheet holds 440 rows: 360 Kubernetes candidates plus 80
  alias senses (was 419). Dylan's A3 judging plan is unchanged in substance.
- Alias pool: 15,393 pairs, 14,469 families, **180 pairs removed by the held-out exclusion**
  (the step-2b build removed 0, which was the defect). Spot-check sample regenerated: 308 pairs.
- Support manifest: per-document domains at the panel's pinned threshold 3 (step 4a used the
  module default 4): science-engineering 60,002, medicine 63,476, finance 24,557, legal 43,579
  documents; two training queries dropped as bare held-out alias terms; general population
  496,327 unchanged. Dose rule outcome unchanged at 204/32/10 with alias passes 3.898 (was 3.853);
  registry `measured_bucket_populations.alias_pairs` and the passes figure updated.

No development-suite, panel-quality or protected read. Next: Codex Sol review (2 of 2).

## Step 4e–4f — Sol review, fixes, re-check and second regeneration (2026-09-11)

Codex Sol reviewed the fixed tree independently (22 findings, `REVIEW.md`). Fixes kept to the
accident-prone bugs under Dylan's ruling: exact search kept whole cutoff ties per document block
(the per-block partial selection was block-size dependent for ties wider than k); evaluation
requires identical keys across dense, BM25, baseline and alias views; export checks the supplied
tokenizer against the snapshot and requires identity for both forms; fixture status is an
explicit flag set only by the rehearsal; the seal requires the exact ten ancestor streams; every
arm requires the alias supply; alias-test families are source-qualified at the pinned threshold
3; a thin abbreviation's expansion is admitted at its rank; uniform candidate draws are without
replacement. The lock-receipt, prepared-manifest and universal-admission items were dropped or
deferred to the step-5 data builder and recorded in `execution_entry_missing`. The single P1
re-check (Sol) confirmed 25 P1 fixes and found three small remaining ones, fixed directly.
Tests 206 → 223. Commits `cc88fc1` and the step-4f commit.

Second regeneration chain (`work/m17/logs/regen_chain2.log`): alias test (same 120/80 split;
source-qualified families; threshold 3 recorded), panel screen and seal (ten streams present by
exact name; script and registry hashes recorded; 440 pending rows unchanged), alias pool
(unchanged: 15,393 pairs, 180 excluded; spot-check sample unchanged), support manifest
(document-only method string; counts unchanged). The chain's rehearsal stage correctly refused
to delete the step-3 output directory, which predates the marker; that directory was removed by
hand and the rehearsal re-run after the re-check fixes: `results/m17_rehearsal_step4.json`,
all gates pass, resume exercised. The step-3 record `results/m17_rehearsal.json` is retained.

Step 4 is closed. No development-suite, panel-quality or protected read occurred.

## Step 5a–5d — the prepared-data builder, two-size timings and the cache speed-up (2026-09-11)

Pre-clock, no protected access, no development-suite or panel read, no quality number.
`m17src/prepare_data.py` builds the directories `train.py --data` consumes (pool, domain join,
deferred protected screen, teacher cache, bank, v1 vectors, old-vocabulary parity, vocabulary,
candidate cache, manifests) at a requested size; `results/m17_prepare_timing.json` holds the
measured cost at 2,000 and 10,000 real queries. The first extrapolation was 20.4 h for the full
574,327-query pool, almost all in `cache.build`'s per-query mat-vec and full sort; commit `6fc3b6a`
blocked the scoring and partial-ordered the walk (order provably unchanged; teacher scores move by
at most 4.2e-7 from the summation order — one v1 near-tie flipped in 1,949 queries) and the rebuilt
caches extrapolate to 1.6 h. Old-vocabulary parity: max deviation 9.76e-4 against the released table
(tol 5e-3) under the real FREEZE hash. Warm-start KL median 0.26 nats, ~80 % of queries above 0.1
(reported only). Codex Astra reviewed the builder (`research/m17-astra-step5-review-2026-09-11.log`,
21 findings, 10 P1); a parallel Sol pass was aborted at Dylan's instruction (reviewers alternate;
`...-aborted-parallel.log`) and Sol reviews the fixed code instead.

## A4 — step-5 owner rulings (Dylan, 2026-09-11)

Dylan: "agree on your recommendations. Just on number 4 the Kubernetes vocabulary is just an
addition that was requested as a nice to have, this shouldn't be that much of a thing."

1. **Labeled positive budget** — `training.labeled_positive_budget_share = 0.5`; the held-out
   divergence slice is excluded from the budget. `positive_bank_policy` unchanged.
2. **Kubernetes documents in the bank** — reserved before the uniform stratified sample, inside
   the cap (`candidate_construction.bank_sampling_amendment_a4`).
3. **Documentless sources** — nqopen/triviaqa cast no distinct-document vote in term discovery
   (`data.documentless_sources_vote = "none"`); contexts and residuals still count.
4. **Pre-clock smoke on unscreened Kubernetes text** — allowed for `--rehearsal --data` only,
   with the driver printing the unscreened row counts; real runs refuse a deferred protected
   screen. Nothing more is built around this: the slice is a nice-to-have.

Registered as `accepted_plan_revision_a4`. No bar, cap, teacher, licence or protocol change.

## A5 — pool cap-fill priority (Dylan, 2026-09-11)

Codex Sol's step-5 review noted the registry says "use as many distinct admitted queries as
exist up to the cap" but not which bucket takes the remainder once general is exhausted
(496,229 distinct general queries; pool 574,229 of 600,000). Ruling: the alias bucket takes all
distinct admitted pairs (15,393, not the 15,000 four-pass minimum) and unpaired-coverage views
fill the rest of the cap. Registered as `data.cap_fill_priority` and `accepted_plan_revision_a5`.
Deterministic; the dose rule states the resulting passes. No bar, cap or protocol change.

## A6 — model-judged descriptive surfaces (Dylan, 2026-09-11)

Dylan: "1 - Is that even necessary? The Kubernetes was just a mention from a coworker who tested
it, a nice to have, I'm not sure if we need dedicated tests. 2 and 3 Can this be done by Astra?"
then "agree with A6, register it and run Astra on 2 and 3. Could Astra do 1 too while at it?"

Ruling: the 360 Kubernetes panel rows, the 80 pending alias senses and the 308-pair alias spot
check are judged by Codex gpt-6-astra (read-only, from the row text alone), ingested by a script
that stamps `judge = codex-gpt-6-astra`. The cloud-software panel domain is labelled
MODEL-JUDGED in the manifest and the paper; the panel remains descriptive and never routes
selection. Dylan re-judges a seeded slice (20 rows, 10 senses, 20 pairs) whenever convenient and
agreement is reported; that slice is not a clock dependency. The 5 % spot-check threshold and its
consequence (tighten the abbreviation filter, rebuild the pool before lock) stand. Registered as
`data.judging_amendment_a6` and `accepted_plan_revision_a6`; A3's human-judging plan is
superseded and retained in the registry text. Briefs: `research/m17-astra-judge-*-brief-2026-09-11.md`;
logs: `research/m17-astra-judge-*-2026-09-11.log`.

## A6 executed — the three sheets model-judged and ingested (2026-09-11)

Codex gpt-6-astra, read-only, from the row text alone (`research/m17-astra-judge-*-2026-09-11.log`).
Panel rows: 205 yes / 155 no over 360 (selection 102/78, audit 103/77; 26 `unsure` notes, all
answered `no`); 118 of 120 Kubernetes queries have at least one relevant candidate. Alias
senses: 57 verified / 23 rejected (18 of the 21 truncated extractions rejected). Spot check:
2 of 308 training pairs wrong (0.65 %), both incomplete R2 expansions; under the 5 % rule, the
pool stands. `m17src/judgments_ingest.py` wrote the answers into the sheet
(`judge = codex-gpt-6-astra`), the panel qrels (`MODEL_JUDGED`) and the alias statuses
(`VERIFIED_BY_MODEL` / `REJECTED_BY_MODEL`), rebound the ancestry-screen receipt with a
field-level proof that only judgment fields changed, and re-sealed the panel:
`results/m17_panel_manifest.json` is **FINAL**, panel sha256 `d1b017a8a7a770ac…`, with the
model-judged disclosure. Dylan's seeded double-check slice (20 rows, 10 senses, 20 pairs, model
answers hidden) is `results/m17_human_doublecheck_slice.jsonl`; not a clock dependency.
Record: `results/m17_judgments_ingest.json`. No development, protected or vector read.

## Step 6 — the full-pool build and the lock code (2026-09-11)

Pre-clock, no protected access, no development-suite, panel or quality read. The full pool
(`work/m17/prepared/full`, no `--size`, seed 0) was built in two invocations of the same output
directory: attempt 1 built pool/domain (146.2 s of stage time: pool 24.3 s, domain 121.9 s,
protected deferred 0 s; `work/m17/logs/prepare_full_crash1.log`) and died at 120,096 of 549,931
teacher texts with `torch.AcceleratorError: CUDA error: an illegal memory access was
encountered`, losing every encoded vector because `TextVectorCache.get` flushed only at the end.
The flush was made chunked (25,000 texts) and atomic, with recovery of an uncommitted vector
suffix; attempt 2 resumed the cached stages and ran teacher→manifests in 5,240.2 s without a
fault, passing the crash point at the same text order (the fault is recorded as transient on the
WSL box, not as a data-dependent bug: nothing about the batch changed). Summed wall clock
**5,386.4 s = 1.496 h** for 600,000 queries; RSS high-water 12.1 GiB (anonymous 9.5 GiB peak at
`student_tokenize`, 8.7 GiB at `cache_build`), GPU peak 3.0 GiB. Stage costs: teacher encode
1,360 s for 549,931 cold texts (404/s), cache build 3,483 s, everything else under 3 min. Pool
plan exactly A5 (496,229 / 72,985 / 15,393 pairs = 600,000; 2,000 held out). Discovery: 210,447
candidates, 445 selected (441 general, 3 cloud-software, 1 legal; 199,763 below the support
minima, 10,239 abbreviations dropped, no cap touched). Protected screen deferred to the clock
with 629 pool rows and 1,648 bank documents from `k8s-docs-en` unscreened. Vocabulary term list
and every downstream identity from this build are a PRICE, not the executed artifact: the
on-clock screen re-derives them (see the lock design below). No quality surface was read.

**The lock (`m17src/lock.py`) is two dated halves**, because `protected_screen` is part of every
stage marker's identity and the screened rebuild re-runs teacher→manifests: the pre half binds
protocol/recipe/seeds/fixed manifests/invariant inputs and the measured allocation and sets
`EXECUTABLE` (preparation only); the executed half, on the clock and BEFORE the V0 read or any
development read, binds the executed vocabulary/tokenizer/new-rows/cache/bank/teacher identities
for both forms, exports V0 through the five gates and sets `LOCKED_EXECUTABLE`, which real
training requires. Two reviews (Codex Astra 11 findings → fixes; Codex Sol: all confirmed,
nothing remaining; `REVIEW.md`). Astra's verdict on the sequencing: defensible under the
registry's lock text and PLANNING's observation boundary; ×2 is a chosen ceiling, not a bound.
The pre half was dry-run against the full build (`--prior-seconds 146.211 --prior-stages
pool,domain,protected`): measured 1.496 h, phase ceiling 3 h (placeholder 16), allocation total
59 h, 13 h unallocated and recorded; registry untouched. **Not yet committed as the lock:** the
checkpoint before the lock and clock start is Dylan's.

**Owner ruling on the clock (Dylan, 2026-09-11, closing Astra P1-8):** "The clock isn't a hard
requirement, this is fine." The 72 h stays the registered ceiling and stop policy for the
phases, not a deadline being raced; `clock_started` is recorded at the first
`--protected-screen` invocation on the full pool, pre-clock preparation and review work stay
exempt, and the two-half lock sequence stands as designed. Recorded here so no session blocks
on a start definition.

## Step 6a–6b — the pre half committed and the clock started (2026-09-11)

The pre half of the lock ran for real and was committed as `e0a1a40` ("M17 lock: pre half");
the registry is `EXECUTABLE`. The clock then started under the ruling above.

**`clock_started = 2026-09-11T23:41:24Z`** (`work/m17/logs/clock_started.txt`), the first
`--protected-screen` invocation on the full pool — recorded conservatively even though that
invocation was REFUSED immediately: "the cached `pool` stage was built under different inputs
(['protected_screen'])". The protected flag is part of every stage identity, so screening an
existing unscreened directory rebuilds every requested stage and needs `--force` (all ten stages
are requested by default; the teacher and doc caches stay warm). The STATUS 6b command omitted
it. Log: `work/m17/logs/prepare_full_screen_refused.log`.

The `--force` relaunch started at 2026-09-11T23:48:37Z via `work/m17/logs/run_screen.sh`
(log `work/m17/logs/prepare_full_screen.log`). The launcher exists because the Claude Code
auto-mode classifier refuses any command containing `--force`; Dylan authorized the flag
explicitly and asked for a standing permission rule, written to `.claude/settings.local.json`
(untracked). No protocol, bar or recipe change.

## Step 6b — the screened rebuild, an import-order crash and a dated invariant amendment (2026-09-12)

The `--force` screened build (23:48:37Z) died at the `parity` stage:
`AttributeError: module 'train' has no attribute 'load_warm_start'`. Importing `m10src/protected10`
inserts m7src at `sys.path[0]`, so the later lazy `import train` in `stage_parity` resolved to the
legacy M7 driver. The unscreened build never imported protected10 and the synthetic rehearsal never
runs the real screen, so the path had never been exercised. Log:
`work/m17/logs/prepare_full_screen_crash1.log`.

Fix `d8f2b43`: `common.reassert_path_order()` (the module-load path-ordering loop, made callable)
is called immediately after `import protected10` in `stage_protected`. `bfa7644` added the two
Codex Astra P2 fixes (evict a cached m7src `train`; a call-site regression test). `9dff737` updated
five tests that assumed the real registry was still `DRAFT_NOT_EXECUTABLE` to build a draft copy;
307 → 308 tests pass.

**Amendment (dated, not a re-decision).** `prepare_data.py`'s sha is an invariant build input bound
by the pre half, so `lock.invariant_build_inputs_sha256.prepare_data_py` was amended BY HAND from
`f4010c28…` to `aefcf585…`, with a dated `lock.amendments` entry recording old, new, reason and the
pre-half commit `e0a1a40`; the original block is preserved in git at `e0a1a40`. It is an amendment
and not a re-decision because the change is import order only — no recipe, constant, protocol or
identity changed — and it governs no observation: the crash happened before vocabulary, cache and
manifests, and no quality surface was read.

**Resume** at 2026-09-12T00:00:56Z WITHOUT `--force`: pool, domain, protected, teacher, bank and v1
were reused (`stage_identity` does not include the builder source), `parity` → `manifests` ran, and
`prepared -> work/m17/prepared/full` finished in 3779.4 s this invocation, RSS high-water 10.95 GiB,
zero `Traceback|Error|FAILED|REFUSED` lines. Screen state `complete` with receipt, 1,512 rows
dropped and refilled to 600,000; seed 0; size null; `registry_status_at_build` EXECUTABLE; manifest
`hashes.prepare_data_py = aefcf585…`, matching the amendment. Log:
`work/m17/logs/prepare_full_screen.log`. On-clock preparation so far ≈ 1 h 25 m across the three
invocations (refused, crash, resume), inside the 3 h ceiling.

**Reviews.** Codex Astra reviewed the fix (brief/report
`research/m17-astra-6b-fix-brief-2026-09-11.md`, `…-review-…md`): two P2s, both fixed in `bfa7644`.
A re-check with a wider read list is running; its verdict is **pending** and gates 6c.

## Step 6c — executed half committed (2026-09-12)

The Codex Astra re-check (`research/m17-astra-6b-recheck-review-2026-09-11.md`) cleared 6c: the
reused stage markers are sound, no live comparison holds the old builder hash, the dated amendment
is sufficient, nothing downstream refuses the partial-rebuild record. Verdict: proceed.

Command (exit 0, log `work/m17/logs/lock_executed.log`):

```
.venv/bin/python m17src/lock.py --phase executed --build work/m17/prepared/full \
  --v0-out work/m17/bundles/V0 --clock-started 2026-09-11T23:41:24Z
```

All five V0 gates PASS (files, artifact, encoder_spec, tokenizer, conformance). 446 vocabulary
terms after the protected screen (445 unscreened). V0 exported with `read: false`; registry status
`LOCKED_EXECUTABLE`. Committed and pushed as `badb048` ("M17 lock: executed half"). **V0 has not
been read**; the one declared descriptive read on the development suite is still pending and needs
the dev-suite reader wired in `m17src/evaluate.py` first.

**On-clock preparation cost (both invocations).** The crashed `--force` invocation's stage timers
sum to **339.9 s**, a lower bound only — `protected10.build()` runs outside those timers — plus
**3,779.4 s** for the resume. Wall clock from clock start (2026-09-11T23:41:24Z) to build
completion ≈ **1 h 25 m**, inside the 3 h preparation ceiling.

**Owner-level note — deferred Astra P2.** On a protected-index cache miss, `protected10.build()`
lazily imports `m10src/cov_screen.py`, which re-inserts m7src at the front of `sys.path` after our
repair. Not fixed now: the fix touches `prepare_data.py`, whose sha the lock binds, and the finished
build is what trains. It is a **required amendment before any future screened rebuild** — reassert
the path order after `build()` as well, recorded as a dated `lock.amendments` entry.

## 2026-09-12 — dev-suite reader wired and reviewed; two owner rulings pending before the V0 read

Reader wired (`2c5321c`), Astra review (six P1 / three P2) fixed in `4824f84`, Sol review of the fix
(seven P1 / three P2, one dropped as spurious) fixed in `ef718a1`; dispositions in `REVIEW.md`.
`pytest m17src/test_evaluate.py` 32 passed. The V0 read is refused at preflight on two points:

**Ruling 1 — trust-on-first-use teacher caches.** All four M7 stella document caches behind the
text-backed dev components (nq-250k, hotpotqa, cqadup-programmers, cqadup-physics) carry shard
digests adopted from their own bytes (5/5, 105/105, 1/1, 1/1 shards, plus both `combined.f16`
stitches). `encode_cached(verify=True)` refuses them. Evidence gathered (no protocol change,
`work/m17/scratch/teacher_spotcheck/summary.json`, seed 20260911, 400 sampled docs per cache
re-encoded with the pinned teacher on the 3080): cosine min 0.999996 across all four, max |Δ| ≤
3.2e-4, zero rows below 0.999, i.e. fp16 rounding only. Every cache `meta.json` pins
`NovaSearch/stella_en_400M_v5` @ `ffeb2b7e…`, mean-l2, `2_Dense_1024`, 1024d, max_length 512, and
each `corpus_sha256` matches the streamed `work/dev/<name>.json` texts with exact row counts. A full
re-encode is ≈ 6.8 h GPU (hotpotqa 6.1 h) and would produce digests that are again self-derived.
Orchestrator recommendation: accept with a dated disclosure keyed to this spot-check; reader
accepts a TOFU cache only when such a disclosure entry exists for it.

**Ruling 2 — text-component query texts unpinned.** `results/m7_dev_manifest.json` pins ids and
qrels but no ordered (qid, text) digest for the four text-backed components; `devsuite` records no
build provenance. The reader records the digest in the receipt and result. Pinning now would be a
manifest amendment binding the digest to itself. Orchestrator recommendation: recorded-only,
disclosed. Both rulings are Dylan's (protocol / evidence integrity); the night stops here.

**Rulings (Dylan, 2026-09-12).** Ruling 1: accept the four trust-on-first-use stella caches with a
dated disclosure keyed to the spot-check above; the reader accepts a TOFU cache only when its
disclosure entry exists. Ruling 2: text-component query texts stay recorded-only, disclosed in the
receipt and result; no manifest amendment. Both are recorded here before the V0 read.
