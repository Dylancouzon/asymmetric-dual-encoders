# M10 planning evidence (2026-09-01 planning session, Mac, no box access)

Facts behind `instructions-m10.md`. One section per decision. Research sweeps ran in two Sonnet
subagents (web only); every number below that a rule reads was re-derived from the repo's own JSON
or checked against the cited primary source.

## 1. The M9 diagnosis in numbers

| SCREEN-3 component | ceiling (`m9_dev_symmetric_stella-400M-v5.json`) | M9 candidate (`m9_final_checkpoint_dev.json`) | retention |
|---|---|---|---|
| nq-250k | 0.8839 | 0.8289 | 93.8% |
| cqadup-physics | 0.4931 | 0.3501 | 71.0% |
| cqadup-programmers | 0.4681 | 0.2345 | 50.1% |
| macro | 0.6822 | 0.5606 | 82.2% |

Build curve (`m9/RUN_STATUS.md`, `m9/BUILD_LOG.md`): 0.346 (closed-form head) → 0.500 at 0.25B
tokens → 0.538 at 0.74B → 0.545 at 1.23B → flat 0.552–0.558 from 2.6B to 3.7B → 0.5606 annealed.
Query pool (`m9/M92_LOCK.md`): 242,786 `queries_pair` (hotpotqa/squad/esci/mrtydi) + 85,863 nqopen
+ 134,665 triviaqa = 463K real queries, all Wikipedia QA or product search; longest 108 words.
Documents 6.15M (581M tokens/epoch). Token mix 5/5/90; realized step 113 examples.

Reading: the student matched the teacher on the distribution it saw and not elsewhere. The six's
query forms (claims, biomedical questions, finance questions, 170-word arguments, paper titles,
COVID queries) are all "elsewhere". Capacity is not excluded by this table alone; §9 and the
M10.0 capacity probe test it directly.

## 2. Bars and headroom — unchanged

Ceiling 0.5744 avg-6; aim 0.5155 needs 89.7%, release 0.5042 needs 87.8% (`m9/PLANNING.md` §1).
The six are all out of M9's training distribution, so M9's avg-6 is expected in the 50–75%
retention band (0.29–0.43) — below BM25 on some datasets. M9's close-out six-set run replaces this
guess with a number and gives the first dev→six calibration point for a tower (M7's is for a
table: dev out-of-domain 0.764 vs six 0.755).

## 3. Levers with published numbers

| lever | evidence | source |
|---|---|---|
| Small batch | LEAF 1-epoch val loss: bs16 0.4194, bs32 0.4214, bs256 0.4593 | arXiv 2509.12539 |
| Cyclic anneals | LEAF: 3 × 10 epochs, 1e-4→1e-5 linear per cycle; M9's single anneal was worth +0.004 | 2509.12539; `m9/BUILD_LOG.md` |
| Query+document text | LEAF 1-epoch NanoMSMARCO: queries-only 46.7 vs both 60.7 | 2509.12539 |
| Query-form breadth | LEAF texts: 502K MS MARCO queries, 979K Amazon QA, 273K PubMedQA/TriviaQA, 27K LoTTE, 3M FineWeb, 900K CC-News — ~1.8M query-like across five styles; ours 463K across two | 2509.12539; `m9/M92_LOCK.md` |
| Ranking-aware KD | mxbai-edge-colbert: listwise KL over teacher scores, 17M/32M students at ~90%/~95% of a 130M+ teacher; EmbedDistill ablation NQ R@5: score distillation 44.3 → +doc embeddings 56.3 → +L2 query matching 61.2 → +synthetic queries 64.3 | arXiv 2510.14880; arXiv 2301.12005 |
| Init | 2306.11550: init strategy moves retention up to 6 points; 1/2/4-layer students retain 86.1/92.5/96.2% of BERT-base | arXiv 2306.11550 |
| Synthetic queries for distillation | EmbedDistill +3.1 R@5 from generated queries; Qwen3 generator precedent (Qwen3-Embedding, arXiv 2506.05176) | 2301.12005; 2506.05176 |
| Capacity gap | TAKD: a too-large teacher hurts a fixed student; a 2026 study finds student capacity, not the gap, gates KD gains. No published asymmetric-retrieval distillation uses a teacher above ~220M/768d — our 400M/1024d pair is outside the literature | arXiv 1902.03393; arXiv 2605.31191 |
| Alternative ≤35M students | ettin-32m (MIT, ModernBERT; ONNX friction, excluded in M9), granite-embedding-30m (Apache, BEIR 50.4), bge-small (BEIR ~51.7) | arXiv 2507.11412; arXiv 2502.20204 |
| Smaller teacher target | stella card: 1024d is 0.001 below 8192d; heads at 256/768/1024/2048+ exist (`2_Dense_*`); 512 does not | HF model card, revision `ffeb2b7e` |

## 4. Qdrant releases and the FineWeb-10B drop — what they change

- **Qdrant/FineWeb-10B** (ODC-By; Common Crawl terms): 10.07B FineWeb chunks embedded with
  `Alibaba-NLP/gte-multilingual-base` (768d) plus sparse vectors, with brute-force ground truth for
  100K queries. **The vectors are in the wrong space for nano** (gte-ml-base, not stella) and the
  bundled ground-truth queries are **MS MARCO-derived** — never used here. What transfers: the
  FineWeb text (ODC-By, already approved as document text), and Supernova's `nova-embed` as a
  GPU-batch encoder pattern for large teacher encodes. Companion drops (PubMed-MV on BGE-M3;
  Coyo-VE on Qwen3-VL-Embedding-2B) are whitepaper context; PubMed is a contaminating corpus here.
- **Qdrant 1.18 (TurboQuant) and 1.19 (TurboQuant 4-bit as primary storage, cold/cached/pinned
  memory tiers, per-tenant IDF)**: index-side levers for the M11 all-in quantization comparison
  and the edge footprint story; they do not change nano's training. **Cloud Inference does not host
  stella**; the document tower stays self-hosted. **Qdrant Edge** versions separately (private
  beta). **fastembed 0.8.0**: `add_custom_model()` has no post-pooling Dense slot — M9's per-token
  linear-head export remains the serving path, and a nonlinear head would need its own graph.
- **No Qdrant-published static or distilled-tiny query encoder exists** (HF org: `bm25`,
  `minicoil-v1`, `Qwen3-Embedding-0.6B-onnx`, CLIP pairs for Edge). zero and nano would be first.

## 5. Data arithmetic

- Query corpus ≈ 463K real + 1.0M PAQ + 3.0M synthetic = **4.5M** texts; ≈ 35 tokens mean (long
  forms 120–220 words at ~10% share) → ≈ 160M tokens per query epoch.
- Generation: 3.0M queries × (~300 prompt + ~60 output) tokens ≈ **1.1B tokens**, of which ≈ 180M
  are generated. Local: Qwen3-8B 4-bit via vLLM on the RTX 3080 — throughput **unmeasured**; at an
  assumed 1,000 generated tok/s ≈ 50 h, at 500 tok/s ≈ 100 h. Hosted open-weights inference at
  $0.1–0.3 per M tokens ≈ **$110–330**. Scale-up is gated on the per-form 200-query smoke, which
  measures the real rate (watch-long-runs rule).
- Teacher targets: stella query encode 2,076/s (`m9_throughput_probe.json`) → 4.5M queries ≈ 36
  min (A2's 4.037M PAQ control texts ≈ 35 min more). Hard-candidate mining: 4.5M × 1M × 1024
  ≈ 9.2e15 FLOP; at a sustained 30 TFLOPS fp16 on the RTX 3080 ≈ 5 min of matmul plus top-k
  over 1M columns per 1,024-query chunk — **unmeasured**; the mandate gates it on a 10K-query smoke
  and registers Qdrant HNSW with an audited recall@64 ≥ 0.98 as the fallback. Bank: 1M pool
  documents, seed 0.
- Build dose 50M examples. At the default 75/25 mix: 37.5M query examples × ~35 tokens + 12.5M
  document examples × ~230 tokens ≈ **4.2B tokens**; at 50/50 ≈ 6.6B. M9's *measured* mixed rate is
  **18,984 tok/s** (`m9/M92_LOCK.md`; the 25,970 tok/s seen live mid-build is not used for planning)
  → **2.6–4 days on the RTX 3080**, before the batch-32 throughput penalty that screen family E
  measures. Query epochs ≈ 8 over 4.5M texts; document epochs ≈ 2 over the 6.15M pool.
- Screens: 2.5M examples per arm ≈ 209M tokens ≈ 3.1 h at 75/25 (B 100/0 ≈ 66M tokens; B 50/50
  ≈ 497M). Nine arms (A1–A3, B 100/0 and 50/50, C, D, E, F; the anchor doubles as B 75/25) ≈ 2.2B
  tokens ≈ **1.3 days**. Confirmations, worst case: B's 50/50 wins (2 seeds × 497M + default
  2 × 209M = 1.41B) plus two ordinary decisions (each winner + default × 2 seeds = 0.84B) = **3.1B
  tokens ≈ 1.9 days**, plus the synthesized selected-recipe arm for LoTTE read #1 (0.2–0.5B).
  Family G adds two arms (+0.42B) and up to one more confirmed decision (+0.84B): screens ≈ 1.6 days,
  confirmations worst case ≈ 3.9B tokens ≈ 2.4 days. Screens total ≤ 4.5 days on the box.
- Surfaces to encode once with stella: COV corpora (MedicalQA 2K, BRIGHT a few K per slice,
  CorporateLobbying 319; LEDGER depends on its chunking, capped at 100K chunks) ≈ minutes to 1 h;
  LoTTE-clean ≈ 2.8M passages ≈ **4 h** at 210 docs/s. Neither existed on the box before.

## 6. Compute plan

| path | wall-clock to M10.4 | cost | notes |
|---|---|---|---|
| RTX 3080 box, when Dylan is back | ≈ 4–5 weeks: generation 2–4 d (unmeasured rate), data + COV admission 1.5 d, screens 1.6 d + confirmations ≤ 2.4 d, build 2.6–4 d plus the batch-32 penalty, M9 close-out + final 1 d, review gates between | $0 | data and caches already on the box; one conservative budget, revised only from measured rates |
| 1× A100 80 GB cloud | ≈ 10–12 days incl. 1 day to re-derive the pool, dev suite and fingerprints from HF; GPU work ≈ 80–110 h | ≈ $120–280 GPU + optional $110–330 hosted generation (Sept-2026 prices unverified) | reproducible from the repo except gitignored `work/` artifacts, which rebuild from the same sanctioned code |
| Mac M5 Pro | probes and code only; stella document encode 20–100 docs/s on MPS depending on document length | $0 | runs stella only in `.venv-mac` (transformers 4.57); transformers 5.x breaks stella's remote code |

## 7. Considered and rejected (reopening condition per row)

| avenue | why rejected | reopens if |
|---|---|---|
| Teacher change (Qwen3-Embedding-0.6B, gte-large-v1.5, arctic-embed-l-v2) | breaks the one-index pair; stella-1.5B measured −0.0023; gte-large-v1.5 is stella's own backbone; Qwen3-0.6B +0.004 nominal, never screened | the pair story is dropped by Dylan |
| >35M student in any role | **hard cap, Dylan 2026-09-01**: "109M is not an option. This isn't low compute anymore. 33M was already in the upper bound" | never |
| Regress to stella's 768d or 256d head | a smaller index is a separate system and a full re-encode of every reserved corpus; §9 says whether the 384-rank bottleneck even binds | §9 shows <95% at k=384 AND the MLP-head arm fails |
| Document-side co-adaptation (E14-LORA) | breaks the pair; M11+ as its own system | never inside M10 |
| FineWeb-10B vectors as targets | wrong embedding space (gte-multilingual-base 768d) | never |
| FineWeb text in any role (Dylan delegated 2026-09-01; ruled out) | documents: no reserved-set fingerprints exist (Codex pass 1 B2); seeds: a rights review and a URL blocklist for topics Wikipedia and the pool already seed | family A wins on forms yet COV shows a topic gap Wikipedia cannot seed |
| PubMedQA, Amazon QA, CC-News as query text (LEAF used them) | PubMed is a contaminating corpus for NFCorpus/TREC-COVID; Amazon QA and CC-News have no affirmative commercial grant | a licence review clears Amazon QA / CC-News |
| Symptom-gated phase 2 (M9 design) | the gate was never specified; a flat curve then had no registered response | never in that form — phase 2 is a screen arm |
| SCREEN-3 as the selection surface | NQ at weight 0.50 with NQ-adjacent training data inflated M9's read by ~11 points against its out-of-domain components | never |
| Spending LoTTE read #1 on the M9 candidate | would burn the only fresh surface on a candidate that misses | never |
| Hosted proprietary API as the query generator | output-use terms would be the weakest link in the licence story | never |

## 8. Adversarial review disposition

*Each table records the state at that pass; later passes superseded some numbers (multiplicity
/6 → /10 → /13, confirmations 3 → 4, the >35M conditional → hard cap, FineWeb → out, 3–4 weeks →
4–5). The mandate is authoritative where a row here disagrees.*

**Pass 1 — gpt-5.6-terra, high effort, read-only, 2026-09-01** (`research/m10-codex-plan-2026-09-01.md`;
read-exclusion audited: the reviewer opened only the twelve named files). Verdict "not
decision-grade"; 3 BLOCKER / 10 MAJOR / 2 MINOR. **All 15 actioned in the mandate rewrite:**

| # | finding | disposition |
|---|---|---|
| B1 | M9's six-set close-out before M10's lock is development on the six | **adopted** — close-out moves to after the M10.2 lock push; no M9 six-set output exists while any M10 decision is open |
| B2 | reserved-set document fingerprints are an unspent held-out access; "hash-only" is not blindness | **adopted** — decision 5 inverted: FineWeb documents excluded; no reserved bytes opened |
| B3 | CQA-2 cannot validate coverage of the six's forms; FORMS-12 is teacher agreement, not retrieval | **adopted** — COV (licensed, decontaminated, qrel-bearing, multi-domain) is the selector, admitted at M10.0-d with a four-component floor; FORMS-12 descriptive; the remaining form gap (claims, titles, arguments) is stated |
| M4 | arm 5 confounds data with recipe; a null stops M10 on weak evidence | **adopted** — family A: A1/A2/A3 differ only in data; A3 vs A2 isolates forms from volume; stop rule and its consequence written |
| M5 | 50/50 mix repeats M9's unresolved mix result | **adopted** — family B: 100/0, 75/25, 50/50 at matched query presentations; default 75/25 |
| M6 | batch pilot confounded (M9's own objection) | **adopted** — family E at equal examples and identical schedule; throughput recorded, decides nothing; default 32 |
| M7 | phase-2 KL not decision-ready | **adopted** — τ by a registered entropy rule before any arm, seed-document exclusion, fixed negative policy, phase 2 as a cycle-3 continuation compared at matched examples |
| M8 | rank probe cannot govern a head decision; MLP has no serving path | **adopted** — head fixed linear; probe descriptive. (The probe measures retrieval retention of projected queries against the frozen index, not reconstruction — the reviewer had not seen the script; the action is unchanged) |
| M9 | FineWeb seeding contradicts the licensing record; outputs called "ours" | **adopted** — seeds default to Wikipedia + approved corpora; FineWeb seeding is decision 3 with a rights review; outputs described under the generator's terms, not redistributed |
| M10 | contamination controls unspecified | **adopted** — thresholds (exact / 8-gram ≥ 8/32 / 4-gram containment), three screen targets incl. the seed passage, per-form quotas, PAQ revision + seed, FORMS-12 hold-out — all in §Data |
| M11 | screens multiplicity- and seed-blind | **adopted** — Bonferroni 0.025/6 on the lower bound; two confirmation seeds at screen dose; revert-to-default rule |
| M12 | cost arithmetic inconsistent; 25,970 tok/s is not the measured rate | **adopted** — one budget from 18,984 tok/s; generation rate marked unmeasured and gated on smoke; §5–6 rewritten |
| M13 | "beat LEAF" is not isolated | **adopted as already present, made explicit** — C2 is a whole-system comparison; the mandate says so and forbids any tower-isolating statement |
| m14 | 35M cap is a product choice presented as science | **adopted** — decision 6: a conditional >35M tier if the capacity probe clears 85% |
| m15 | reserved batch conditional on C1 only | **adopted** — `if C1 or C2` in the M10 re-registration |

**Pass 2 — same reviewer, on the rewritten text** (`research/m10-codex-plan2-2026-09-01.md`;
read-exclusion audited clean). Disposition audit: 10 of 15 pass-1 fixes landed, 5 did not
(M4, M5, M11, M12 arithmetic, M7 timing). New: 3 BLOCKER / 8 MAJOR / 1 MINOR. **All actioned in
the second rewrite:**

| # | finding | disposition |
|---|---|---|
| B1 | COV not protected from training data | **adopted** — admitted COV queries and documents join the protected index before any PAQ/synthetic construction; the M9 pools are re-screened; removals published before COV is scored |
| B2 | M9's close-out could spend the reserved access if its C1 passed | **adopted** — M9's close-out is amended to six-only (reserved conditional struck), disclosed and ratified with decision 1 |
| B3 | family B's "equal examples" contradicts "matched query presentations" | **adopted** — B is matched on 1.875M query presentations with documents added on top; totals and document cost stated; equal examples holds for the other families |
| M | family A conflates volume with forms; inequality undefined | **adopted** — A2 is a factoid-only volume control at 4.5M unique texts; A3−A2 isolates forms; the rule is point ≥ MDE and corrected lower bound > 0; the stop rule reads A3−A2 and its conclusion is scoped to COV families |
| M | COV can collapse to a narrow surface | **adopted** — equal weight per family, slices averaged first; four-family floor |
| M | COV eligibility unestablished | **adopted** — admission record (licence URL and terms, repo + revision, sizes, qrels format, metric) committed before a component is named COV |
| M | COV cannot carry the full coverage conclusion | **adopted** — the conclusion and stop rule are scoped to COV families; claims / titles / arguments are tested only by the six-set transaction, stated in §Surfaces |
| M | contamination exclusions omit NutritionFacts and Reddit finance | **adopted** — the full source-family list is in §Data and applies to seeds, regression text and COV |
| M | τ cannot be selected at M10.0-e | **adopted** — rule locked at M10.0-e; executed after the immutable manifest, before any arm; sample spec, tie-break and fallback fixed |
| M | multiplicity and confirmation ill-defined | **adopted** — ten contrasts enumerated; Bonferroni 0.025/10; confirmation re-trains winner and default with two seeds each, at most three decisions |
| M | dose and cost arithmetic (3.8B vs 4.2B; mining; screen days) | **adopted** — 4.2B in both files; mining FLOP count stated, measured on a smoke, HNSW fallback registered; screens re-budgeted at 2.5M examples: 1.3 days + ≤ 1.5 days confirmations; box path 3–4 weeks |
| m | rank-probe docstring says 512d; STATUS asks for a nonlinear-head dry run | **adopted** — docstring corrected (1024/768/256); the dry-run line removed |

**Pass 3 — same reviewer, verification of the second rewrite** (`research/m10-codex-plan3-2026-09-01.md`;
read-exclusion audited clean). 12 of 15 pass-2 dispositions land; two "not fully"; 4 new MAJOR.
**All actioned:**

| # | finding | disposition |
|---|---|---|
| M | family A's stop rule is weaker than "beat by the MDE" | **adopted** — three registered outcomes on A3−A2: corrected lower bound > MDE = resolved; point ≥ MDE with bound in (0, MDE] = positive-not-resolved, build proceeds labelled; else stop |
| M | A2 cannot reach 4.5M with a 4.0M PAQ cap | **adopted** — A2 = 463K + 4.037M PAQ = 4.5M; decision 4 says so |
| M | LoTTE read #1 has no candidate and no veto statistic | **adopted** — one synthesized selected-recipe arm at screen dose, hash committed before LoTTE opens; M9 §7's 0.004 non-inferiority rule verbatim; veto → the anchor recipe builds |
| M | STATUS says four surviving components, mandate says four families | **adopted** — STATUS corrected to four families |
| M | confirmation budget understated after the family-B redesign | **adopted** — worst case recomputed at 3.1B tokens ≈ 1.9 days plus the synthesized arm; box path ≈ 4 weeks; cloud GPU-hours 70–100 |

**Pass 4 — same reviewer, on the third rewrite, family G and the report page**
(`research/m10-codex-plan4-2026-09-01.md`; read-exclusion audited clean). Pass-3 items all land;
family G's export algebra and parameter count confirmed. 2 BLOCKER / 2 MAJOR / 2 MINOR on the plan,
5 number disagreements on the page. **All actioned:**

| # | finding | disposition |
|---|---|---|
| B | §9 promoted the PCA row to an upper bound on every 384-d subspace | **adopted** — reworded everywhere: the row is the reconstruction-optimal subspace's retention, the target L2 regression pushes toward; evidence, not a bound; the screen decides G |
| B | family A's rule contradicted the generic rule | **adopted** — A3−A2 exempted explicitly; resolved requires the corrected lower bound > MDE; A3−A1 and A2−A1 descriptive |
| M | stale counts (ten contrasts, nine arms) | **adopted** — eleven arms, thirteen contrasts everywhere |
| m | probe JSON provenance text said 512-d head | **adopted** — text field corrected to 768/256 (numbers untouched) |
| M/m | report page: LEAF 97.9 vs 97.7; build-curve first point; Mac docs/s; 33M; premature "four passes" | **adopted** — LEAF labelled 97.9% on our six (97.7% is BEIR-14) in `m9/FINDINGS.md` and the page; the curve's first point labelled as the build's 0.12B eval with the screen anchor drawn separately; 20–100 docs/s; 33.4M; the review paragraph rewritten after this pass |

**Pass 5 — same reviewer, cross-file consistency and omissions on the frozen set**
(`research/m10-codex-plan5-2026-09-01.md`; read-exclusion audited clean). 3 BLOCKER / 5 MAJOR /
2 MINOR. **All actioned:**

| # | finding | disposition |
|---|---|---|
| B | the CQADupStack components were scored by the Mac diagnostics yet proposed as COV | **adopted** — COV admits only surfaces no M10 decision has read; the CQA pair is DEV-6, reported beside every COV read; 86 raw reads logged in `m10/RESULTS.md`; floor three untouched families |
| B | A2 and A3 were not volume-matched (4,500,314 vs 4,463,314) | **adopted** — identical post-screen unique counts, the larger downsampled with seed 0, both hashes locked before any arm |
| B | BRIGHT's third-party documents fail an "every document's rights" standard | **adopted modified** — the standard applied is the dataset-level licence at the primary source, the same one that admitted CQADupStack and the six (also third-party text); BRIGHT is one family, its caveat disclosed, evaluation-only. Demanding per-document rights would disqualify the six themselves |
| M | student contradiction (bge-small architecture vs MiniLM default) | **adopted** — bge-small is the screen anchor, family F decides the build student, MiniLM default; MiniLM's head passes the same parity check first |
| M | generation gate not executable (20 vs 200, unpinned fallback, no decoding/seed/retry/rubric/approver) | **adopted** — §Data generation contract: 200 per form everywhere, pinned artifacts, sampling parameters, deterministic seeds, one retry, dedup, the 90%/80% rubric, Dylan as approver, two prompt revisions max |
| M | the ledger does not exist | **adopted** — `m10/LEDGER.md` skeleton committed with the sections the lock must fill |
| M | extension, margin, seed range, negatives RNG, missing-seed behaviour undefined | **adopted** — all defined in §Recipe and §Screen |
| M | seed-level leakage; 5-word rule vs 8-gram screen; lenient parser | **adopted** — seeds pre-filtered against the protected index; word-5-gram containment against the seed; `forms.parse` strict |
| m | stale numbers in §8 read as policy; "five passes" premature | **adopted** — this note at the top of §8; pass 5 recorded |
| m | dev-reuse accounting not fillable | **adopted** — exact counts (86) in `m10/RESULTS.md` |

## 9. Rank-bottleneck probe (`m10src/rank_probe.py` + `rank_probe_mix.py`, Mac, 2026-09-01)

`results/m10_rank_probe_mac.json`. Stella-400M query vectors (s2p prompt) projected onto their
top-k principal components, renormalized, retrieved by exact search against the unmodified stella
document vectors of the two CQADupStack dev components (manifest hashes verified). **The Mac
reproduces the box ceiling**: programmers 0.4681 vs 0.46807, physics 0.4932 vs 0.49314.
A student with hidden width h and a linear head emits pre-normalization vectors in ONE
h-dimensional affine subspace (L2 normalization does not change a ranking, so the subspace is what
matters for retrieval). PCA gives the **reconstruction-optimal** k-d subspace of the fit set — the
subspace an L2-regression objective pushes such a student toward — so the k = h row is the
retention of that subspace, **not an upper bound over every k-d subspace** (a ranking-optimal one
may do better). Retention of the component's own full-rank score:

| basis fit on | k | programmers | physics |
|---|---|---|---|
| NQ-open questions (20,000) | 384 | 79.8% | 89.3% |
| NQ + the other forum component | 384 | 82.7% | 92.0% |
| NQ + other + **the target's own queries** (oracle mixture) | 384 | 90.4% | 93.1% |
| the target's own queries only (single-distribution oracle) | 384 | 99.5% | 99.6% |
| NQ + the other forum component | 512 | 92.5% | 97.4% |
| NQ + the other forum component | 640 | 97.9% | 100.2% |
| NQ-open questions | 768 | 99.4% | 100.6% |

Explained variance of NQ queries at 384 components: 84.3% — stella's query space is not
low-rank. Stella's 768-d and 256-d MRL heads do not help (768-space k=384: 82.1% / 89.2%).

**Reading.** The reconstruction-optimal 384-d subspace serves ONE query distribution almost
perfectly (99.5% / 99.6%) and several distributions poorly: fit on NQ it retains 80–89% on forum
queries; fit on a mixture that includes the target's own queries, 90–93%. The aim needs 89.7% across
six distributions. Under L2 distillation — whose objective is reconstruction — this is **strong
evidence that a 384-wide linear head binds before training starts**, and consistent with M9's
50–71% on the same components; it is not a theorem about every 384-d subspace, and a ranking-aware
loss (phase 2) could pull the student elsewhere. At width 640 the same subspace retains 98–100%.
Within 35M parameters the width comes from the *feature*, not the backbone: mean-pool three layers
(bge-small layers 12, 8, 4) and concatenate → 1152-d feature → Linear(1152→1024), +786,432
parameters over the 384-d head (head 1.18M; ≈ 34.5M total), and still exportable per token so fastembed's own mean pooling
reproduces it exactly (mean pooling is linear; M9's trick; identical masking required). This is
screen family G in the mandate, default 1152; **the screen decides, not the probe.** Caveats: two
forum components stand in for the six; the capacity probe (now optional, report-only) has a
768-hidden student, so any gain it showed would be partly width. Diagnostic; read by no rule; dev reads
counted (2 components, 3 scoring passes each per basis).

## 9b. Head-width probe in closed form (`m10src/head_width_probe.py`, Mac, 2026-09-01)

`results/m10_head_width_probe_mac.json`. M9's head-probe design (frozen bge-small backbone, ridge
head to stella targets fit on 20,000 NQ-open questions, λ on a training-only holdout) with three
pooled features, scored on the two CQA components against the cached stella documents:

| feature | dim | programmers (retention of stella) | physics |
|---|---|---|---|
| mean of layer 12 (M9's head) | 384 | 0.1271 (27.1%) | 0.1750 (35.5%) |
| layers 12 + 8 | 768 | 0.1545 (33.0%) | 0.2022 (41.0%) |
| layers 12 + 8 + 4 (M10 default) | 1152 | 0.1722 (36.8%) | 0.2167 (43.9%) |

Each added layer helps a frozen backbone, monotonically, by 6 and 4 points on programmers. This is
a floor (no training, NQ-only fit set), not a forecast — M9's trained 384-d student reached 50% /
71% on the same components — and it is consistent with §9: with more output directions the
frozen features already reach further into stella's space. Family G's default is 1152; the screen
decides. Diagnostic, read by no rule; dev reads counted (2 components × 3 features).

## 9c. Serving parity of the three-layer per-token head (`m10src/head_width_parity.py`, Mac CPU)

`results/m10_head_width_parity_mac.json`. bge-small with the Linear(1152→1024) head applied per
token over the concatenated states of layers 12, 8, 4, exported at opset 17: **1073 nodes, zero
custom-domain ops, 34.54M parameters** (head 1.18M). fastembed 0.8.0 serves it as a custom
MEAN-pooled normalized model and reproduces the pool-then-head reference to **min-cos
0.99999984, max-abs 2.0e-07** on 64 texts of 1–300 words. **M10.0-a2 passes** (weights are
random; parity does not depend on them). Two serving pitfalls for M11: fastembed needs
`config.json` and `special_tokens_map.json` beside the graph, and transformers 5.x fast tokenizers
no longer write the latter.

## 10. Reuse, do not rebuild

`m9src/longrun.py`, `watchdog.py`, `guardian.sh`, `sentinel.sh`, `sacrificial.sh` (build and
supervision) · `m9src/final9.py` + `final_stats.py` (access machine, statistics, 16 tests) ·
`m9src/nano.py`, `warmfit.py`, `data.py` (student, closed-form head, pools) · `m9src/eval9.py`,
`m7src/evalkit.py`, `devsuite.py` (scoring) · `m9src/port.py`, `export_doc_model.py`,
`edge_cost.py` (ONNX, fastembed, costs) · `m9src/capacity_probe.py` (unchanged) ·
`m8src/blockcg.py`, `decide.py`, `noise_floor.py`, `paths_guard.py`, `protected_filter`,
`dev_reuse_m8.py` · `m7src/decontam.py` (R1/R2/R3 fingerprints), `fusion.py`, `freeze.py`,
`boot.py`. `m8/CODEMAP.md` and `m9/CODEMAP.md` carry the pitfalls.
