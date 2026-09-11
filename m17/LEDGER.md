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
