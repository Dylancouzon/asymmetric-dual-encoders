# M18 ledger

## 2026-09-12 — E0 execution authorization and initial lock

- Dylan authorized executing `instructions-m18.md`, including local collection, training and the
  fresh M18 evaluation surfaces. M7–M13 protected/spent evaluation content remains forbidden.
- Pinned `qdrant/qdrant` default branch `master` at commit
  `5e32ea89cb5ea9a6827a68b27291d2e2251e1b1c`; GitHub export cutoff
  `2026-09-13T00:36:26Z` (UTC).
- The staged released Zero v1 is accepted only because its `model.npz`, tokenizer and config hashes
  are pinned in the registry and the model hash equals `m7/FREEZE.json:table_sha256`.
- The M7 unfolded checkpoint is the only training initialization. M17 endpoints are historical
  negative evidence and are not M18 inputs.
- GitHub collection uses repository-wide REST list endpoints for issues/PR openings, issue
  comments, review comments and issue events. This preserves explicit page receipts without one
  request per thread. Cross-reference/closing links are taken from event/link fields; unsupported
  timeline payloads are not silently inferred.
- Evaluation target is 250 queries: development 35 and confirmation 15 per realized stratum. The
  mandated minima govern short strata. Split assignment happens on deduplicated artifact families.
- The T0 diagnostic uses a 4,000-step scheduler with `stop_after=500`; a separate 500-step schedule
  would change the optimization trajectory and make the later resume incomparable.
- The primary evaluation is held-out question retrieval over the complete pinned snapshot, not a
  claim that only documents predating each question are searchable. Query-bearing openings remain
  distractors and are never relevant solely because they share a thread; qrels point to distinct
  answer spans and the self-hit rate is reported.
- API cutoff means objects/events created no later than the cutoff, with mutable bodies captured as
  observed during the recorded acquisition interval. M18 does not claim GitHub reconstructs body
  versions at the exact cutoff instant; post-cutoff updates are counted in reconciliation.
- No M18 result has been observed at this entry.

## 2026-09-12 — E1 minimum fork and synthetic rehearsal

- Copied from M17 at `4213920`: `common.py` (`caa1e927…`), `vocab.py` (`9ab9a30a…`),
  `cache.py` (`4d1e0ca4…`), `train.py` (`78f7ab12…`), `export.py` (`c3a377b5…`),
  `loader_np.py` (`140cdf1b…`) and the rehearsal structure (`7bbe396f…`). Evaluation copied only
  the generic exact-search/metrics/bootstrap ideas; none of M17's dev/panel readers came across.
- Semantic differences: fresh M18 paths/guards; project-only support rather than M17 domain caps
  and pins; colon compound extraction; only appended rows are Parameters; no learned-scalar
  optimizer; fixed 4,000-step identity plus execution-only stop; T1/T2 serving preprocessing;
  atomic cache/snapshot/bundle publication; no checkpoint averaging or parameter cap; fresh
  M18 structural surfaces and dense/BM25/DBSF evaluator.
- The canonical synthetic rehearsal passed. It parsed 42 document units in 21 artifacts, created
  10 development and 5 sealed-confirmation questions across all five strata, resumed steps 2→4
  to bit-identical rows versus an uninterrupted run, kept inherited rows unchanged, moved a new
  row, passed int8 loader gates, and refused a second confirmation claim. Receipt:
  `results/m18_rehearsal.json`. These fixture metrics are not quality observations.
- A planning Astra review found the schedule/freeze/atomicity/protocol issues above. It reviewed
  no implementation and does not count as either pre-expensive-run implementation review.

## 2026-09-13 — E2 first implementation review and remediation

- The first Astra implementation review was a no-go for live acquisition/GPU work. It found that
  title-only splitting could separate one artifact family and conceal duplicate-answer leakage;
  broad answer cues admitted information requests; confirmation reads bypassed the one-read rule;
  live payload edits broke byte-level REST reconciliation; and training/export did not bind every
  realized array, tokenizer and table identity.
- Remediation connected families across artifact IDs, exact answer digests, title/backport shapes
  and explicit Qdrant thread links. Split reports now compute actual artifact/family/answer-digest
  intersections. Generic log/detail requests are rejected, targets must be later answer-bearing
  chunks, discussion objects are chunked before qrels, and verbatim documentation self-pairs were
  removed.
- REST reconciliation now compares the cutoff-admitted canonical identity set while recording
  mutable payload changes and a growing post-cutoff tail; the first-pass bytes remain immutable.
  Corpus parsing requires the combined source manifest and all page receipts in real execution,
  and repository bytes are read from the pinned Git object rather than the worktree.
- Confirmation can only be loaded inside a decision-locked, hash-verified, exclusive one-shot
  transaction. Training resume binds ordered query IDs/student IDs, teacher vectors, document
  bank, candidates, scores, aliases and initial rows. Snapshot load reconstructs and verifies the
  concatenated table. Pair-aware batches co-locate supported alias views. Export binds variant,
  tokenizer, preprocessing and float-table hashes and compares the actual torch training forward
  path to serving.
- The revised synthetic rehearsal passes 26 network-free tests. It now reports zero artifact,
  connected-family, answer-digest and development/confirmation overlap; confirmation is exercised
  only through the one-shot transaction. A second independent implementation review is pending.

## 2026-09-13 — E3 second review and live pagination correction

- The second independent review approved live raw acquisition but held development reads, full
  Stella encoding and training behind stale-cache, label, index and execution-lock fixes. Those
  focused blockers were corrected and covered by 28 network-free tests; no broad framework was
  introduced.
- Live GitHub acquisition reached issue page 100 and received HTTP 422: GitHub now requires the
  opaque `after` cursor from its Link header for large collections. The adapter was corrected to
  carry that cursor while retaining explicit page numbers and immutable receipts. The first 99
  validated pages resumed in place because their endpoint, order, page and payload identities are
  unchanged; page 100 onward records the cursor in each request receipt.

## 2026-09-13 — E4 realized corpus and protocol review

- The reconciled pinned snapshot contains 71,937 canonical GitHub objects. Parsing the Git tree and
  discussion snapshot yielded 105,025 records across 11,574 artifacts: 79,269 indexable document
  units and 25,756 non-indexable relationship events. The terminal issue-event page was full at the
  server's 30,000-object cap; this is disclosed auxiliary metadata and never supplies a retrieval
  document or sole qrel justification.
- The first realized protocol contained 1,594 training views, 175 development questions and 75
  sealed confirmation questions. It had zero measured artifact, connected-family, answer-digest or
  development/confirmation leakage. Confirmation content remained unread.
- Astra's deterministic 40-development-qrel audit found 24 clear answers, eight partial/diagnostic
  targets and eight wrong targets. It also found that 174/175 query-bearing source documents contain
  the query verbatim, making their unjudged presence a systematic evaluation confound. Corpus scale,
  provenance, deduplication and leakage controls were accepted; scoring was not.
- The prospective v3 qrel rule no longer admits thread closure plus generic words as an answer. It
  requires substantive resolution, linked explanation or closing-event plus strong explanation;
  rejects information/clarification and unrelated CI/codespell/rebase/push status; and retains the
  cleaner explicit review-reply relation. Evaluation removes, but never marks relevant, each
  query's own source document and records its pre-exclusion top-10 rate.
- Task-style PR titles no longer fill the concept/how-to audit stratum. The rebuilt development
  composition is 25 review questions and 10 issue openings there. Overall, the v3 protocol has 921
  structural candidates, 1,043 training views, 175 development questions and 75 sealed confirmation
  questions; all five strata remain at 35/15 and all measured leakage intersections remain zero.
  Full Stella encoding remains paused until Astra re-audits this rebuilt development set.

## 2026-09-13 — E5 final narrow qrel hardening

- A focused diff review found that bare “need/want logs” requests and interrogative or negated
  resolution language could still pass the v3 predicates. The v4 rule rejects those forms in both
  opening/comment and explicit review-reply candidates, with falsifying fixtures for each case.
- Generic review questions without project-specific error, configuration, identifier or jargon
  context were also removed from the concept/how-to audit stratum. They depended on an unseen code
  diff even when their replies were substantively correct. That stratum now consists of 35 issue
  openings; no synthetic replacements were made.
- The rebuilt v4 protocol has 733 structural candidates, 901 training views, 175 development
  questions and 75 still-sealed confirmation questions. All five strata retain their 35/15 target,
  every measured leakage intersection remains zero, and 39 network-free tests pass. Previously
  flagged IDs `5110`, `7632` and `9159` remain only because v4 selects later, substantive answers
  for them; their earlier diagnostic/status targets are gone.

## 2026-09-13 — E6 use-case alignment and adjudication scope

- The session-external review in `m18/REVIEW.md` correctly identified that the numeric/identifier
  audit slice does not reproduce bare terms such as `s3` and `k8s`. Its premise that issue bodies
  are evaluation queries is incorrect: issue/PR queries are titles, while bodies stay in the corpus
  and provide adjudication context. Measured median title/question length is nine words overall and
  12 in exact/numeric; only one exact/numeric development query has five or fewer words.
- Adopt the review's smallest direct acceptance artifact after vocabulary preparation: a fixed,
  report-only side-by-side top-10 for v1 and the selected candidate on the supported bare terms. It
  has no manufactured qrels and does not replace the registered metrics. Do not add a sixth gated
  stratum or wait on owner-authored questions; voluntarily supplied questions can still be added in
  a later milestone as originally ruled.
- Astra's v4 audit was still no-go: 27/40 clear, eight partial and five bad. More lexical exceptions
  would not test whether an answer actually addresses a question. The prospective pool is therefore
  capped at the newest 80 connected families per proposed stratum (344 pairs total after source-text
  duplicate union, bot-review removal and prose-question validation). Astra sees only query, source
  context and proposed target, before the final split exists; it sees no retrieval scores or model
  identity. Only `clear` pairs survive. This is a one-pass M18 data correction, not a reusable
  judgment framework.

## 2026-09-13 — E7 prospective adjudication executed

- One Astra judge reviewed all 344 content-bound prospective pairs without retrieval output, model
  identity or a pre-existing audit split. It rated 141 clear, 141 partial and 62 bad, and corrected
  94 plainly mismatched strata. The decision file covers the exact candidate ID set and copies each
  source/query/target identity hash; its SHA-256 is
  `d98144096f17a718bb70073655fbcd4f28319c75f41d670f0a376118aa505904`.
- Only the 141 clear pairs were split: 101 development and 40 sealed confirmation. Alias, concept
  and exact/numeric have fewer than 35 available clear families, so their smaller realized counts
  are disclosed rather than filled with partial labels. Confirmation content remains unread.
- The surviving clear pairs all enter held-out surfaces, leaving no adjudicated labeled structural
  training rows. To preserve project-language support without reintroducing bad positives, 297
  non-held-out issue/PR titles are retained through the existing teacher-only/query-only cache path;
  review comments are excluded from this fallback. Together with 482 evidence-backed alias views,
  the training protocol has 779 views. No rejected target is used as a relevance label.
- The v5 manifest again reports zero artifact, connected-family, answer-digest and
  development/confirmation overlap. Query-source documents remain excluded before evaluation
  metrics. Forty-three network-free tests pass. A fresh Astra reviewer must still spot-check the
  final development surface before full Stella encoding.

## 2026-09-13 — E8 independent adjudication audit and final corrections

- A fresh Astra reviewer independently sampled 40 of the 101 v5 development qrels: 39 were clear,
  none bad, and one partial. `m18q:issue:3822` answered a later commenter's unsupported version
  downgrade rather than the original reporter's EC2-snapshot atomic-write failure. Its decision is
  corrected to partial, leaving 140 clear pairs: 100 development and 40 sealed confirmation.
- The reviewer also found four long issue openings whose title appeared in a second opening chunk.
  Protocol rows now carry every chunk ID for their exact source GitHub object. Evaluation excludes
  those chunks only—not answer comments or the rest of the thread—and retrieves up to 120 results
  so 100 remain after the largest 20-chunk exclusion. The 140 held-out queries bind 474 source
  chunks in total; the four reported multi-chunk cases resolve exactly to their recorded lists.
- The corrected decisions hash is
  `e2257e103d893bdda1cb9b71786f6a5f75729eb9baa028e1bd113dc305e59c41` and the v6 protocol identity
  is `0a919bdde39f7059fbc0cd27c2768929bff5754bdc7da1c2b644f8963b9a3c1a`.
  Confirmation content remains unread. No corpus rebuild or broader evaluation machinery was added.
- The same independent reviewer verified the two corrections at `a685370` and issued **GO** for
  full Stella encoding and development scoring. It judged the dataset proportionate without more
  machinery, provided the planned report-only bare-term top-10 comparison remains in the final
  evidence. This closes the pre-expensive-run data gate.

## 2026-09-13 — E9 project-memory index complete

- Encoded all 79,269 indexable documents once with the pinned Stella revision into normalized
  1,024-dimensional fp16 vectors. Twenty resumable shards and the 162,343,040-byte combined array
  passed shape, dtype, finiteness, norm and SHA-256 checks. Wall time was 329.396 seconds, peak CUDA
  allocation 1,729,131,008 bytes and process RSS high-water 3,849,068 KiB.
- Built the registered local BM25 index over the identical ordered document texts. The combined
  index binds corpus, ordered IDs, indexed text, Stella revision/vectors and BM25/DBSF recipe under
  identity `a710f91a7b19f27a2f7251e369199dc0bc9fcfd56bf080bf31c3c01124fd368e`.
  This is the reusable system deliverable and frozen candidate bank, not a new document model.

## 2026-09-13 — E10 bounded vocabulary and preparation complete

- The first support threshold of 12 training contexts admitted only `qdrant`, `shard` and `flaky`.
  This was too strict for a protocol that normally contributes one natural query per source
  document. Astra approved the smallest correction: retain the five-distinct-document floor and
  lower only the context floor to five. No term was hand-pinned and no source data was added.
- The final 15 exact rows are `qdrant`, `shard`, `grpc`, `snapshot`, `deps`, `json`, `flaky`,
  `docker`, `kubernetes`, `hnsw`, `config`, `s3`, `turboquant`, `arm64` and `gridstore`. Supported
  examples named in the instruction but not admitted—including `k8s`, `mmap`, `cuda`, `tls` and
  `rocksdb`—remain a disclosed data-support limitation and will be included in the report-only
  bare-term retrieval comparison.
- The T1 random-suffix expression had treated `Log-Structured-Merge-Database` as an ephemeral pod
  suffix. It now requires a digit in the generated-name segment. The collision audit now reports
  only actual many-to-one transformations across relevance groups, while retaining unchanged
  strings when checking whether a transformed value collides with a literal. All 780 training
  views contain zero T1 transformations; T1 is therefore unsupported/inert and is skipped rather
  than expanding its pattern set.
- Preparation produced 142 eligible queries each for T0 and T2 from the shared 780-query Stella
  cache. V0-T0 and V0-T2 each contain the 15 initialized rows and occupy 31,392,036 float-table
  bytes. The collision audit passes and 45 network-free tests pass.
- Alias evidence is useful but uneven: 241 alias-pair views exist, with 128 digit-bearing or
  vowel-free short-form occurrences across 46 distinct short forms. Among admitted abbreviation-
  like rows, evidence is sparse (`hnsw`: 2 pairs, `s3`: 2, `arm64`: 1, `grpc`: 0). This is a
  limitation to report, not grounds for synthetic qrels or a larger vocabulary arm.

## 2026-09-13 — E11 execution recipe locked

- The source, corpus, protocol, index and preparation manifests plus the registered model,
  vocabulary, variant, training, retrieval, evaluation and serving recipe are bound by immutable
  `m18/execution-lock.json`. Its SHA-256 is
  `8bc47902ec10424e68784b47ee61745daf1a00ff841bc3c4b42715ccc8f046eb`.
- Registry state is now `LOCKED_EXECUTABLE`. Training must refuse any changed recipe or artifact;
  confirmation remains sealed until a selected recipe/checkpoint is separately decision-locked.

## 2026-09-13 — E12 first development baseline read

- The first locked baseline read completed before student training. Equal-weight stratum-macro
  nDCG@10 was 0.109231 for BM25, 0.087336 for released Zero v1 dense, 0.103431 for v1+DBSF,
  0.087336 for V0-T0 dense, 0.103431 for V0-T0+DBSF, 0.136560 for Stella dense and 0.140992 for
  Stella+DBSF. V0-T0's observed ranking metrics were exactly equal to v1 even though 28 of 100
  queries contain an added token; the parity receipt separately verifies the other 72 unchanged-
  token queries. Stella exceeded v1 by 0.049224 dense and 0.037561 fused, demonstrating headroom
  for the bounded student experiment.
- Query-bearing source chunks appeared in the unfiltered top ten for 99–100% of queries, confirming
  that the registered source-object exclusion is material. The report measures metrics only after
  excluding all such chunks.
- This result is retained as `results/m18_development_baselines_pre_k8s.json`. It predates the
  owner-approved, CTO-requested `k8s` training amendment and is immutable evidence rather than the
  baseline for the revised V0 identity.

## 2026-09-13 — E13 pre-training CTO `k8s` amendment

- After the first development baseline read but before any optimizer step, Dylan approved a lock
  revision to cover the CTO-requested `k8s` term. This is explicitly a revised recipe, not a claim
  about the original execution lock. The original lock is retained as
  `m18/execution-lock.pre-k8s.json`; the original manifests and baseline are retained with
  `_pre_k8s` suffixes and in commit `e6e20f6`.
- Qdrant source `gh_opening:7967:chunk:5` directly states `Kubernetes (K8s)` and is outside every
  held-out family. Its normalized source hash is pinned in the registry. Five already-admitted,
  distinct-source training contexts containing `kubernetes` receive one deterministic `k8s`
  substitution each. These are teacher-only views with no positive qrel; inherited AKS/EKS/GKE/LKE
  alias-pair metadata is stripped. Together with the one natural `k8s` issue title, this gives six
  distinct source contexts while clearly separating one natural from five augmented occurrences.
- Astra issued GO for this narrow source-evidenced augmentation and rejected the need for Qwen or
  a new evaluation stratum. The staged protocol contains 785 training views. Development and
  confirmation query/qrel hashes are byte-identical to the original protocol, all leakage counts
  remain zero, and confirmation content remains unread. Vocabulary, caches, prepared arms, V0
  bundles and their baseline will be rebuilt under fresh identities before training.

## 2026-09-13 — E14 amended preparation complete

- Fresh preparation under the amended protocol encoded all 785 raw training views and rebuilt the
  candidate cache, prepared arms and V0 bundles without reusing pre-amendment artifacts. The shared
  eligible set grew from 142 to 148 queries.
- `k8s` is selected as the sixteenth exact row with six distinct source documents and six contexts:
  one natural and five source-evidenced augmented views. All original 15 terms remain selected;
  no unrelated vocabulary row changed membership. T0 and T2 float tables are 31,393,064 bytes.
- T1 still transforms zero of 785 training queries and remains skipped. The amended collision
  audit passes, and all 47 network-free tests pass.

## 2026-09-13 — E15 amended execution recipe locked

- The amended source/corpus/protocol/index/preparation identities and recipe are sealed by the new
  `m18/execution-lock.json`, SHA-256
  `008e99173752dfe856395efb01b29cec424dcae7255dd8ff647b56ceb3fc8b60`. The registry is again
  `LOCKED_EXECUTABLE`; the original pre-`k8s` lock remains archived separately.

## 2026-09-13 — E16 amended V0 development baseline

- Re-evaluated the changed 16-row V0-T0 bundle on the identical 100-query development set. Its
  dense and fused metrics remain byte-for-byte equal to both released v1 and the pre-amendment V0
  result: macro nDCG@10 0.087336 dense and 0.103431 with DBSF. This is expected because the fixed
  development set contains no `k8s` query; the separate bare-term report will test that shape.
- The report retains the unchanged BM25 and Stella ceiling results under their verified identities.
  Registry `development_read` is now true; confirmation remains unread.

## 2026-09-13 — E17 small-pool trainer compatibility correction

- Before the first optimizer step, the trainer's redundant `eligible < batch` refusal exposed that
  148 eligible queries cannot satisfy the registered batch of 256 without repetition. The existing
  sampler already defines a deterministic cross-epoch stream, and the cosine/listwise losses have
  no in-batch-negative semantics. Astra therefore approved retaining batch 256, warmup 200 and the
  complete fixed schedule while removing only the contradictory refusal.
- Astra found that repeated query IDs could corrupt the alias loss: the former implementation
  paired any two slots with the same pair ID, including two copies of one view, and dropped pair IDs
  appearing three or more times. Alias slots now require the two distinct global query identities
  and count each complete pair once. The trainer records per-step unique queries, repeated slots,
  maximum multiplicity and actual alias-pair count; its version is bound into the execution lock and
  resume identity.
- The realized eligible pool contains 25 alias-bearing rows but zero complete eligible pairs, so
  alias consistency is truthfully inactive and its effective loss is zero. Orphan views cannot
  self-pair. The `k8s` row still receives cosine and listwise supervision from six contexts. Astra
  recommended reporting this limitation rather than concentrating the full alias weight on one
  newly manufactured pair.
- A 4,000-step sampler-only audit over the exact prepared pool observed 140–148 unique queries per
  256 slots (mean 147.497) and maximum multiplicity three. Fifty network-free tests pass.

## 2026-09-13 — E18 corrected trainer recipe locked

- The duplicate-safe trainer version and unchanged batch/schedule are sealed by the third and final
  pre-training execution lock, SHA-256
  `3251a72c573d171cbb6544eb83ea10052beb29a744e883498e48122de52754da`. The superseded amended lock
  remains at `m18/execution-lock.pre-batch-fix.json`; no optimizer step preceded this correction.

## 2026-09-13 — E19 T0 step-500 diagnostic passes

- The first real optimizer run reached the registered step-500 pause in 179.236 seconds with peak
  CUDA allocation 725,551,616 bytes. Loss fell from 0.56220 at step 1 to 0.49276 at 250 and
  0.42261 at 500; every inherited-row hash remained unchanged. Across 500 steps, 256 slots held
  140–148 unique queries (mean 147.496), maximum multiplicity three, and zero alias pairs as
  expected from the prepared-pool audit.
- All step-0/250/500 exports pass tokenizer/table integrity, NumPy loader parity (zero loader error),
  training-forward parity (maximum absolute error `1.49e-8`) and int8 error (maximum `8.27e-4`).
- Macro nDCG@10 progressed from 0.087336 at step 0 to 0.088115 at 250 and 0.088801 at 500 for
  dense retrieval. DBSF progressed from 0.103431 to 0.106384 at both trained checkpoints. The
  respective step-500 deltas versus v1 are +0.001465 dense and +0.002953 fused. They do not yet
  meet final eligibility margins, but both trained checkpoints are above—not below—step 0, so the
  registered early-halt condition is false and the fixed screen continues.
- T2 changes tokenization for six of 100 development queries (6%, so it is not inert), but its
  step-0 metrics are exactly equal to T0. T1 changes zero training queries and remains skipped.

## 2026-09-13 — E20 T0 first-seed screen complete

- T0 resumed from step 500 to the unchanged 4,000-step schedule. Macro dense nDCG@10 deltas versus
  v1 were +0.000779 at 250, +0.001465 at 500, +0.000923 at 1,000, +0.007488 at 2,000 and +0.007591
  at 4,000. The dense margin is met at 2,000/4,000 without a clearly regressing stratum.
- Fusion did not follow the dense gain. Its best macro nDCG@10 delta was +0.002953 at steps 250 and
  500; steps 1,000/2,000/4,000 yielded +0.000245, +0.000245 and +0.000717. No T0 checkpoint meets
  the required +0.010 fused margin, so none is eligible. T2 is the only remaining non-inert arm.

## 2026-09-13 — E21 T2 first-seed screen complete

- T2 completed the same 4,000-step schedule. Its losses differ slightly from T0, but every dense
  and fused per-query metric is exactly equal to T0 at all six checkpoints. It therefore has zero
  fused replacement delta, below the required +0.005, and the simpler raw-text T0 dominates it.
- No V0, T0 or T2 form meets both encoder eligibility margins. The provisional result is
  `ENCODER_NO_IMPROVEMENT`, with released v1 retained in the hybrid project-memory system. A second
  seed is not launched because there is no eligible trained form to confirm. Astra is independently
  auditing this decision before the confirmation recipe is locked.

## 2026-09-13 — E22 encoder decision independently audited

- Astra verified every eligibility comparison, checkpoint gate and execution identity and issued
  GO for released v1 with `ENCODER_NO_IMPROVEMENT`. V0 has zero deltas; every trained form misses
  the fused margin; no stratum interval is wholly below zero; and no second seed is applicable
  because no first-seed trained form survives the practical screens.
- T1 changes 0/100 development queries. T2 changes six at step 0 and produces different vectors
  from T0 for 29 queries at step 4,000, but every registered per-query ranking metric is equal;
  this is a genuine no-effect result rather than a shared-bundle error.
- `results/m18_encoder_decision.json` selects the exact released-v1 model, tokenizer and config
  hashes and binds the execution lock, index, protocol, evaluation recipe and development evidence.
  The confirmation CLI now verifies that binding before it can claim the sole read. T0 step 250 is
  predeclared only for the report-only bare-term comparison and cannot affect deployment.

## 2026-09-13 — E23 one-shot confirmation completed

- The pre-read decision lock has SHA-256
  `3f8ce905260f8b821f1acf1598060b3bb9864c3287eeadd0dd168cee8bacb453`. It binds released Zero v1,
  the encoder decision, execution lock, index/protocol/evaluation identities and every development
  evidence hash before confirmation access.
- The confirmation transaction completed exactly once. Its receipt has SHA-256
  `3dceb77f9003a9d9dab9a59324d3c7517e9b2f29a6bf6d62037750ce1cdb3194`, records `reads: 1`, and
  binds result SHA-256 `855fa1cee8c7b3d9ebfb9567732150e73c6589129e336fd99d63a5fab0e9c9ef`.
- On 40 queries, stratum-macro nDCG@10 is 0.109163 BM25, 0.153176 released-v1 dense and 0.151445
  v1+DBSF; Recall@10 is 0.200, 0.220 and 0.300 respectively. Error/troubleshooting scores zero for
  all three routes on its ten-query slice. Because no trained candidate was eligible, candidate
  confirmation gates are not applicable and the read does not reopen encoder selection.

## 2026-09-13 — E24 system finalized

- The report-only 21-term probe compares released v1 with the predeclared, unselected T0-step250
  checkpoint and has no qrels. All 16 admitted terms change dense and fused rank order with mean
  top-ten overlap 8.06. `k8s` and `s3` surface plausible project-memory targets; the five unsupported
  requested terms remain unchanged. This qualitative probe cannot affect eligibility or deployment.
- Three live DBSF queries passed against the final v1-backed system. `k8s readiness probes` ranks
  the matching readiness issue first, and S3 snapshot/HNSW configuration prompts retrieve relevant
  discussions. End-to-end latency was 173–264 ms on this host; this exact-search smoke is not a
  production benchmark.
- Final outcomes are `SYSTEM_READY` and `ENCODER_NO_IMPROVEMENT`. Registry confirmation state is
  closed, further training is refused by status, and `results/m18_system_manifest.json` binds the
  exact system inputs, evidence and invocation.
