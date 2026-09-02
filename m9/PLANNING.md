# M9 planning evidence (2026-08-30 planning session)

Facts behind `instructions-m9.md`. One section per decision. Every load-bearing figure was
recomputed from the source JSON by the planning session itself (a Haiku extraction agent misread
two numbers — treat subagent-extracted numbers as unverified until recomputed).

## 1. Headroom: the aim is possible; the band is a prior, not a forecast

Measured on this harness (recomputed from `results/m7_final_run.json` per-query values and
`results/perquery.json`):

| system | avg-6 | NDO-4 | scifact | nfcorpus | fiqa | arguana | scidocs | trec-covid |
|---|---|---|---|---|---|---|---|---|
| stella-400M symmetric (ceiling) | 0.5744 | 0.5640 | 0.7796 | 0.4134 | 0.5536 | 0.6369 | 0.2395 | 0.8234 |
| arctic-embed-m-v1.5 | 0.5264 | 0.5348 | 0.7159 | 0.3624 | 0.4241 | 0.5953 | 0.2148 | 0.8461 |
| leaf-ir-asym (aim) | 0.5155 | 0.5233 | 0.6990 | 0.3608 | 0.4165 | 0.5833 | 0.2034 | 0.8301 |
| mdbr-leaf-ir | 0.5123 | 0.5221 | 0.7056 | 0.3653 | 0.3917 | 0.5939 | 0.1977 | 0.8197 |
| bge-small (release bar) | 0.5042 | 0.5046 | 0.7127 | 0.3430 | 0.4035 | 0.6034 | 0.2052 | 0.7575 |

Required nano retention of the ceiling: **89.7% avg-6 / 92.8% NDO-4** (aim), 87.8% (release).
Published retention for 22–33M distilled query towers — a PRIOR only (each differs in teacher,
dose, data, dim): LEAF asym 97.7% BEIR-14 (97.9% on our six: 0.5155/0.5264); EmbedDistill
6-layer ~96% (supervised structure); 2306.11550 4-layer 96.2% / 2-layer 92.5% (8M MS MARCO
queries); ScalingNote 29M 99.9% R@50 (labeled industrial pairs). LEAF's dose was ~100 A100-hours;
our affordable dose is a fraction — achievable retention is estimated from M9's own pilot curves.
Weak row: the ceiling loses TREC-COVID to arctic-m (0.8234 vs 0.8461).

## 2. Teacher

- Ceiling measured, not projected (calibration `m7_calibration.json` predicted 0.556±0.035;
  measured 0.5744).
- MTEB raw per-task (mteb/results repo, main_score=ndcg_at_10): stella-400M avg-6 0.5609 ·
  stella-1.5B **0.5837** · Qwen3-Embedding-0.6B 0.5649 · gte-Qwen2-1.5B 0.5876 (1.78B, training
  data UNDISCLOSED → dead, harrier precedent) · arctic-m-v1.5 0.5269 · bge-base 0.5264 ·
  arctic-l 0.5289.
- Contamination (`results/m7_teacher_contamination.json`): stella discloses ArguAna, FiQA2018
  (2/6) and **FEVER (1 of the reserved four)**; not DBpedia/Quora/CQADupStack. Registry caveat:
  disclosed data bounds KNOWN contamination only.
- MTEB registry `embed_dim` bug: for instruct-wrapped models it reports backbone FFN
  intermediate_size (stella "4096"), not output dim. Use model cards.
- stella s2p query prompt, verbatim: `"Instruct: Given a web search query, retrieve relevant
  passages that answer the query.\nQuery: {query}"`. Docs promptless. L2-norm convention
  throughout the harness.

## 3. LEAF recipe (arXiv 2509.12539, ACL 2026; extraction verified against the HTML)

- **Loss: plain L2** on embeddings; MiniLM/TinyBERT-style auxiliary losses tried and REJECTED by
  the authors. Black-box teacher, no labels/negatives.
- Text: ~6.7M (3M FineWeb, 900K CC-News, 979K Amazon QA, 502K MS MARCO queries, 273K
  PubMedQA/TriviaQA, 27K LoTTE); both roles regressed; asymmetric serving is post-hoc. Teacher
  prompts baked into targets.
- Student: all-MiniLM-L6-v2 init, FULL backbone trained, mean pooling (beat CLS/EOS/max), Linear
  384→d head, Norm iff teacher unit-norm.
- Budget: bs=32 (beat 256 on final loss — "dense supervision favors more steps"), 3×10-epoch
  cycles, LR 1e-4→1e-5 cyclic, AdamW, ~100 A100-hours.
- One-epoch ablation: queries-only 46.7 NanoMSMARCO vs queries+docs 60.7 — but their student
  serves both roles; a query-only tower may differ (hence our registered mix screen).
- Adjacent: ScalingNote MSE+cosine (needs labels; NB: with normalized outputs squared-L2 and
  cosine are affine-equivalent — MSE+cosine dropped from our phase-2 pool); Jasper/stella distill
  (2412.19048) cosine+Gram-MSE+margin (phase-2 candidates); EmbedDistill (2301.12005);
  2306.11550 (layer-pruned students). DistilVDR (2608.10636) = the M12 image pointer.
- Sanity: LEAF paper per-dataset on our six reproduces our harness (52.63→0.5264; 51.6→0.5155).

## 4. Student shortlist (Aug 2026; no new sub-35M retrieval-tuned encoder exists)

| candidate | params | licence | vendor | fastembed | verdict |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22.7M | Apache-2.0 | community/clean | yes | **finalist** (LEAF-proven; orig fine-tune @seq128 → long-query bins registered) |
| bge-small-en-v1.5 | 33.4M | MIT | BAAI clean | yes (default) | **finalist** (retrieval-tuned @512; also the release-bar comparator) |
| ettin-encoder-17m/32m | 17/32M | MIT | JHU clean | no | out: ModernBERT ONNX friction (eager-attn-only export, no optimum optimization, RoPE decomposition at opset 17) — M10 risk |
| arctic-embed-xs/-s | 22/33M | Apache-2.0 | Snowflake (strongest justification) | yes | out: vendor optics + LEAF's teacher vendor |
| granite-30m / e5-small-v2 / gte-small | 30/33/33M | Apache/MIT | justify tier | no | out: no edge over bge-small |
| mdbr-leaf-ir | 22.6M | Apache-2.0 | MongoDB | no | OUT (vendor rule) — reference row only |

fastembed: `TextEmbedding.add_custom_model()` needs ONNX at a known repo path + tokenizer files +
explicit pooling/normalization/dim; official listing via a fastembed-team issue (we are Qdrant).

## 5. Protocol facts

- Reserved four: FEVER 5.4M docs, DBpedia-entity 4.6M, cqadup-android 23K, cqadup-english 40K.
  One access. FEVER: stella-disclosed AND was in M7's TRAIN stack (`m7_field_table.md`:
  fever-train 97,708 q after decontam, positives from the BeIR/fever corpus, doc store
  `fever-pos`) → dropped from M9 training; labelled sensitivity row only.
- M7 TRAIN stack (per `m7_field_table.md`): hotpotqa 81,780 · fever 97,708 (dropped for M9) ·
  squad 84,956 · esci-us 73,096 · mrtydi-en 3,310 · nqopen 85,871 (B-only) · triviaqa 134,761
  (B-only). ≈460K queries for nano after the FEVER drop. R3 overlap vs cqadup-untouched: 1 doc of
  854,921.
- Dev suite (M7, hash-pinned): nq-250k 3,452 q · hotpotqa 7,405 · cqadup-programmers 876 ·
  cqadup-physics 1,039 · heldout-train 7,325 · heldout-longq 55 (subset of heldout-train).
  Dev-reuse count at M8 close: 494.
- LoTTE-clean: 7 forum topic-splits, 20,122 q, CC BY-SA — never used for selection or training;
  M9's fresh surface (≤2 reads, macro over slices). Forum-heavy → same family as reserved CQA;
  disclosed.
- Confirmatory machinery precedents: paired stratified bootstrap, signflip_dep, Holm α=0.025
  (`m8src/decide.py`); M7 conformance tolerance |Δ| ≤ 3e-4.
- Six-set paired-bootstrap resolution ±0.007. M8 dev bars (0.0040/0.00519) are
  table-lever-specific; nano probe-style claims need their own floor.

## 6. Considered and rejected (reopening condition per row)

| avenue | why rejected | reopens if |
|---|---|---|
| Re-derive zero against stella-1.5B; rebuild the pair there | tower quality does not predict table quality (T1, Spearman 0.000; stella-400M was #1 of 9 ON THE TABLE); discards a frozen, confirmatory-verified artifact for a coin flip | teacher screen fires AND the 400M ceiling is the diagnosed cause |
| Teacher swap for nano alone | breaks the pair (two indexes); ceiling is not the binding constraint | screen swap rule: lower-CI>0 AND ≥0.010 |
| >35M student (~110M ≈ 0.568 at 99% retention) | third frontier point competing with arctic-m symmetric, not LEAF; muddies the pair | M10+ scoping, Dylan's call |
| Student init by pruning stella's layers (2306.11550-style) | ~50M+ at 1024 hidden (over cap); unproven vs LEAF's in-band init | a <35M pruning beating a finalist in the screen |
| Warm-start student token embeddings from zero's rows | convergence trick the headroom doesn't need; couples artifacts | phase-1 convergence diagnosed as binding |
| zero+nano dense fusion as a product mode | correlated dense channels; BM25 already carries the decorrelated one | the optional descriptive row measuring a non-trivial gain |
| Higher-dim stella index (2048/4096) | card: ~0.001 below 8192d at 1024d; index doubles for nothing | nothing |
| MRL / smaller-dim index | stella's alternative dims are separate learned heads → separate system + full re-encode, not a free truncation | M10+ as its own system |
| Reserved-four composition changes | protocol-frozen; FEVER handled by NDO-3 macro + labelled row | nothing |
| MS MARCO (even text-only) | terms non-commercial; priced by M7 (unresolved, sign-flipping) and declined | licence change |

## 7. Adversarial review disposition (gpt-5.6-sol, 2026-08-30)

Verbatim review: `research/m9-codex-plan-2026-08-30.md` (9 BLOCKER / 12 MAJOR / 1 MINOR + summary;
read-exclusion honored — log audited, zero reserved-path reads). **All 23 findings actioned.**
21 adopted as stated. Two adopted MODIFIED, reasons:
- **F12 (confirmatory gates):** gates stay on the SIX (M7 precedent, frozen comparators, maximal
  power) with Codex's tightened two-contrast rule verbatim; the reserved NDO-3 family-weighted
  macro becomes a pre-registered DIRECTION check for the aim headline instead of the gate surface
  (reserved sets are low-power and comparator-vector-free; the six carry the irreplaceable paired
  instrument).
- **F5 (teacher screen):** restored, but sharing arms with the student screen (4+2 arms) rather
  than a separate campaign; swap rule = Codex's margin rule verbatim (lower-CI>0 AND ≥0.010).
Highest-value catches: LoTTE double-use (F6 — would have destroyed the only fresh surface),
artifact-freeze-before-confirmation (F14), dose-in-units (F4), FEVER out of training (F9/F10),
system-level headline (F22), comparator bridge check (F23).

**Pass 2** (same reviewer, on the committed text: `research/m9-codex-mandate-2026-08-30.md`,
11 BLOCKER / 7 MAJOR / 1 MINOR — all actioned in the 2026-08-30 mandate rewrite; log audited,
no reserved reads). Adopted as stated except: seed replicas at full dose carry a Dylan-waiver
clause (multi-day-dose case), and the M9.1 arm list names bge-small as the anchor student
(pass 2 left it open). Owner rulings taken in-session and recorded in the mandate: FineWeb
APPROVED (conditions in §Data), git = branch `m9-work` for execution sessions, size = quality
first, 70 MB fp16 target, exceedable with logged measured justification. Also renamed:
the M7 model is **zero** ("zeo" was Dylan's typo, propagated since the M8 close — fixed
repo-wide this session). Structural changes pass 2 forced: M9.0 screen-lock stage; exhaustive
M7 carry-forward list (was "everything else binds", which silently re-imported M7's table
mission); bridge = phase 1 of the single six-set transaction; reserved manifest in the same
freeze commit; NDO-4/NDO-3 demoted to descriptive (the "unrestricted headline" tier was an
undeclared third gate); fully specified bootstrap/sign-flip algorithms; LoTTE atomic-read
protocol; six sequential screen arms with a conditional teacher branch; kill/extension/phase-2
all numeric-at-lock with "diagnosed defect" = implementation divergence only.
