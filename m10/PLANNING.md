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
  memory tiers, per-tenant IDF)**: index-side levers for the M13 all-in quantization comparison
  and the edge footprint story; they do not change nano's training. **Cloud Inference does not host
  stella**; the document tower stays self-hosted. **Qdrant Edge** versions separately (private
  beta). **fastembed 0.8.0**: `add_custom_model()` has no post-pooling Dense slot — M9's per-token
  linear-head export remains the serving path, and a nonlinear head would need its own graph.
- **No Qdrant-published static or distilled-tiny query encoder exists** (HF org: `bm25`,
  `minicoil-v1`, `Qwen3-Embedding-0.6B-onnx`, CLIP pairs for Edge). zero and nano would be first.

## 5. Data arithmetic — SUPERSEDED (2026-09-04 evidence only; the mandate's §Data and §Compute are authoritative)

- Query corpus ≈ 463K real + 1.0M PAQ + 3.0M synthetic = **4.5M** texts; ≈ 35 tokens mean (long
  forms 120–220 words at ~10% share) → ≈ 160M tokens per query epoch.
- Generation: 3.0M queries × ~300 prompt tokens ≈ 900M prefill; output ≈ 60 tokens for ten forms
  and ≈ 120–270 for the conversational and argument forms (`max_new_tokens` 400) ≈ **250M generated
  tokens** (350M at the caps), plus retries. Qwen3-8B bf16 via vLLM on the A100 — **unmeasured**; at
  1,500–3,000 decode tok/s with prefill overlapped ≈ **30–60 GPU-hours** ≈ $45–150. Fallback if the
  smoke's end-to-end projection of the full job (all forms, prefill, retries, JSON failures) exceeds
  60 GPU-hours: hosted open-weights inference at $0.1–0.3 per M tokens ≈ $110–330, the second §6
  scenario. The per-form 200-query smoke measures requests/s end to end (watch-long-runs rule).
- Teacher targets: stella query encode was 2,076/s on the box (`m9_throughput_probe.json`); at an
  assumed 3× on the A100, 4.5M queries + A2's 4.037M PAQ control texts + ≤ 3M seed passages (D-NCE
  positives) ≈ **2 GPU-hours** with mining. Hard-candidate mining: 4.5M × 1M × 1024 ≈ 9.2e15 FLOP,
  minutes of matmul plus top-k over 1M columns per 1,024-query chunk — **unmeasured**; the mandate
  gates it on a 10K-query smoke and registers Qdrant HNSW with an audited recall@64 ≥ 0.98 as the
  fallback. Bank: 1M pool documents, seed 0.
- **Planning rate.** LEAF trained 6.7M texts × 30 epochs ≈ 201M examples at batch 32 in ~100
  A100-hours (`m9/PLANNING.md` §3) → **≈ 560 examples/s**. Our examples are shorter on average (≈ 84
  tokens at 75/25 against LEAF's 256-token FineWeb documents), so the rate is conservative in
  direction; it is unmeasured on our stack. M9's box rate (18,984 tok/s ≈ 226 examples/s,
  `m9/M92_LOCK.md`) is retired with the box. The first screen arm measures examples/s and §6 is
  re-derived from it.
- **Build dose 200M examples** (LEAF's; the 2026-09-01 amendment — the old 50M and its 83.4M cap
  were set by the box's days, not by evidence; M9's plateau was on a narrow pool). At 75/25: 150M
  query examples × ~35 tokens + 50M document examples × ~230 ≈ **16.8B tokens**; at 50/50 ≈ 26.5B.
  ≈ **100 GPU-hours** at 560 examples/s. Query epochs ≈ 33 over 4.5M texts; document epochs ≈ 8
  over the 6.15M pool (LEAF: 30 over everything). Each extension cycle is 66.7M ≈ 33 GPU-hours.
- Screens: 5M examples per arm (2.5% of the build; 2.5M was 5% of the old 50M build) ≈ 2.5
  GPU-hours at 560 examples/s, plus ≈ 0.3 h of DEV-6 and COV evaluation. Fourteen arms: A1–A3 (15M),
  B 100/0 (3.75M) and B 50/50 (7.5M; the anchor doubles as B 75/25), C, D-KL3, D-KL1, D-NCE, E, F,
  G-384, G-768, G-1536 (9 × 5M = 45M) = **71.25M examples ≈ 40 GPU-hours**. Confirmations, worst
  case: B's 50/50 wins (2 seeds × 7.5M + default 2 × 5M = 25M) plus three ordinary decisions (each
  winner + default × 2 seeds = 20M) = **85M ≈ 47 GPU-hours** with its 16 evaluations, plus the
  synthesized selected-recipe arm for LoTTE read #1 (5M ≈ 2.5 h). Capping confirmations at two
  decisions saves ≈ 20 GPU-hours.
- Encodes with stella, all re-derived on the instance (the box's `work/` caches do not travel): the
  6.15M pool (targets and the heldout dev components) ≈ 2.8 h, hotpotqa 5.2M + nq-250k ≈ 2.5 h, the
  six ≈ 0.1 h, COV (MedicalQA 2K, BRIGHT a few K per slice, CorporateLobbying 319, LEDGER ≤ 100K
  chunks) ≈ 0.3 h, LoTTE-clean 2.8M ≈ 1.3 h, the reserved four (FEVER 5.4M, DBpedia 4.6M, two CQA)
  ≈ 4.6 h **only if the reserved conditional fires** — ≈ **12 GPU-hours** at an assumed 600 docs/s
  (3× the box's 210, unmeasured), plus about a day of CPU and network to pull the pool, dev suite
  and fingerprints from HF.

## 6. Compute plan — SUPERSEDED (the all-cloud plan; the mandate's §Compute table is authoritative and `max_extension_cycles` reads it, not this)

"M10 won't be done on a 3080. M10 will be done on a GPU budget, if allowed, or not at all." The box
path is withdrawn (§7). One A100 80 GB (H100 if cheaper per example on the smoke), ≥ 500 GB
persistent disk, stopped between stages; $1.5–2.5/h assumed, unverified Sept 2026. Every line is at
the §5 planning rates and is re-derived from the day-one rate benchmark (mandate §Compute) before
generation starts, and again before the remaining screen arms.

| mandatory line | GPU-hours | $ at 1.5–2.5/h |
|---|---|---|
| re-derive encodes (pool, dev suite, six, COV, LoTTE; reserved four only if the conditional fires) | 12 | 18–30 |
| day-one rate benchmark (encode, three training mixes, generation smoke) | 2 | 3–5 |
| query and seed teacher targets, mining | 2 | 3–5 |
| screens, 14 arms at 5M with evaluations | 40 | 60–100 |
| confirmations, worst case 85M with 16 evaluations | 47 | 71–118 |
| synthesized selected-recipe arm, LoTTE reads, M9 close-out scoring, M10.0-c | 4 | 6–10 |
| build, 200M examples | 100 | 150–250 |
| export, parity, final run | 2 | 3–5 |
| persistent disk ≈ 3 weeks, egress | — | ≈ 40 |
| **subtotal without generation** | **209** | **$354–563** |
| **scenario A — generation on the GPU** (Qwen3-8B bf16, vLLM) | 30–60 | 45–150 |
| **scenario A total** | **239–269** | **≈ $400–715** |
| **scenario B — hosted generation** (if the smoke projects > 60 GPU-hours on the GPU) | 0 | 110–330 |
| **scenario B total** | **209** | **≈ $465–895** |
| optional: second build seed (decision 8) | 100 | 150–250 |
| optional: extension cycle, 66.7M examples, each | 33 | 50–83 |
| **ceiling requested** | | **$1,000, hard** |

**Allocation order at the lock:** every mandatory line first at the measured rates and the billed
price; then decision 8 if ≥ 100 GPU-hours remain; then `max_extension_cycles` whole cycles from the
remainder (mandate §Recipe). At $2.5/h nothing optional fits in either scenario and the plan still
completes; at $1.5/h with scenario A, seed 1 and three extensions fit.

Wall-clock ≈ 2.5–3 weeks sequential on one GPU (≈ 240 GPU-hours, a day of CPU and network, the
smoke approval and the review gates between stages); screen arms are independent, so 2–4 GPUs
compress the screen and confirmation stages at the same total cost. The same dose on the RTX 3080
would be ≈ 10 days for the build and ≈ 4 for the screens, before any confirmation seed. **From the
box, once, sha-verified:** `work/m9long/ckpt/last.pt` (M9's frozen candidate, `9d631b2c…`) for
family C, the M10.0-c baseline read and M9's close-out, plus any `work/` file `m9src/guard9.py`
hashes; nothing else transfers. The Mac stays what it was: probes, code, the generation smoke's
prompt development (stella only in `.venv-mac`).

## 7. Considered and rejected (reopening condition per row)

| avenue | why rejected | reopens if |
|---|---|---|
| Teacher change (Qwen3-Embedding-0.6B, gte-large-v1.5, arctic-embed-l-v2) | breaks the one-index pair; stella-1.5B measured −0.0023; gte-large-v1.5 is stella's own backbone; Qwen3-0.6B +0.004 nominal, never screened | the pair story is dropped by Dylan |
| >35M student in any role | **hard cap, Dylan 2026-09-01**: "109M is not an option. This isn't low compute anymore. 33M was already in the upper bound" | never |
| Regress to stella's 768d or 256d head | a smaller index is a separate system and a full re-encode of every reserved corpus; §9 says whether the 384-rank bottleneck even binds | §9 shows <95% at k=384 AND the MLP-head arm fails |
| Document-side co-adaptation (E14-LORA) | inside M10 it breaks the pair. Co-training the tower against **both** query paths at once keeps it, and it is the lever every ≥ 96% near-zero-query system used (LightRetriever's lookup 96% vs zero's 75.5%; ScalingNote 99%; CARE stage 2); costs a new index and a rebuild of zero | never inside M10; **recommended 2026-09-01 as the next-milestone candidate** (that slot was M12 then; it is **M16** after the 2026-09-04 renumbering), Dylan's call |
| The RTX 3080 as M10's execution target | Dylan 2026-09-01: cloud GPU budget or not at all. A LEAF-scale build is ≈ 10 days on it and the screens 4 more; the 50M dose and 83.4M cap it forced were box artifacts | never for M10 |
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

**Pass 6 — verification of the pass-5 fixes and the trim** (`research/m10-codex-plan6-2026-09-01.md`;
read-exclusion audited clean). 8 of 10 land; the trim removed nothing a box session needs. Two
items remain as **recorded dissents**, plus one contradiction fixed:

| # | finding | disposition |
|---|---|---|
| B | BRIGHT admitted although its third-party documents' rights are not conveyed | **not adopted, dissent recorded** — the reviewer's per-document standard would also disqualify the six and CQADupStack; the plan applies M7's dataset-level eval-use standard consistently, discloses the caveat, and uses COV for selection only, never for a claim. Owner may overrule (`m10/COV_CANDIDATES.md`) |
| B | COV floor three, reviewer wants four | **not adopted, dissent recorded** — with the CQA pair demoted, four untouched families exist only if LEDGER admits; the floor stays three, the report names the family count, and Dylan may raise it |
| M | mandate said M9 "is merged after the close-out cleanup" while STATUS says merged with close-out pending | **adopted** — one wording: merged 2026-09-01 after the repo cleanup; close-out pending from `m9-work` |

**Owner amendment 2026-09-01, after pass 6 — the compute ruling and the review taken with it.**
Dylan: "M10 won't be done on a 3080. M10 will be done on a GPU budget, if allowed, or not at all."
The same-day review of the plan against the goal (a best-in-class asymmetric pair) found four
places where the box, not the evidence, had set a number; they changed together:

| # | change | why | where |
|---|---|---|---|
| 1 | box path withdrawn; one rented A100; itemized budget, ceiling $1,000 | the ruling | §6, mandate §Compute, decision 2 |
| 2 | build dose 50M → **200M** examples; extension while a cycle end gains ≥ 0.003, capped by budget | LEAF's dose is 201M on an easier target (768-d, 109M teacher); the old cap was the box's days | §5, mandate §Recipe |
| 3 | screen dose 2.5M → **5M** | 2.5M was 5% of the old build; a 5% screen of 200M would cost ≈ 70 GPU-hours | §5, mandate §Screen |
| 4 | family D gains **D-KL1** and **D-NCE**; family G gains **1536**; 16 contrasts, 0.025/16 | a ranking signal only in cycle 3 leaves §9's own caveat untested; the seed passage is a free positive (CARE stage 1, EmbedDistill); §9b is still rising at three layers and a fourth costs 0.39M parameters | mandate §Recipe, §Screen |
| 5 | **COV resolution number** before the lock (descriptive since pass 7) | MedicalQA 2,048 documents, CorporateLobbying 340 queries, BRIGHT ~100 per slice: an MDE-sized contrast may be unresolvable, and the report must say whether an unresolved verdict was invisible | mandate §Surfaces, M10.0-d |
| 6 | seed-rank provenance field | a round-trip filter without a second generation pass | mandate §Data |
| 7 | generator in bf16; 4-bit and Qwen3-4B fallback withdrawn; hosted fallback if the end-to-end projection exceeds 60 GPU-hours | an 80 GB card | mandate §Data, `m10/COV_CANDIDATES.md` |
| 8 | decision 8: second build seed inside the ceiling | the headline's CI is a query-sampling interval only (`m7/FINDINGS.md` 9) | mandate decisions |
| 9 | co-training the tower recorded as the next-milestone candidate (now M16) | every ≥ 96% near-zero-query system co-trained the document side | §7 |

**Pass 7 — gpt-5.6-terra, high effort, read-only, on the amendment**
(`research/m10-codex-plan7-2026-09-01.md`; read-exclusion audited clean: the reviewer opened the
named plan files, `m9src/guard9.py`, `m9/PLANNING.md` §3, `m7/FINDINGS.md`, and ran one web search
on A100 vLLM throughput). Verdict "not decision-grade"; 2 BLOCKER / 6 MAJOR / 1 MINOR / 5
contradictions. **All actioned:**

| # | finding | disposition |
|---|---|---|
| B | the COV resolution check scored bge-small and MiniLM — family F's own backbones — and made a COV-read decision (screen scope) before selection | **adopted** — demoted to a descriptive **resolution number**: e5-small-v2 and gte-small (candidates in no family), distance only, no direction, first disclosed COV read, cuts nothing; the screen always runs fourteen arms |
| B | the hosted-generation branch was outside the budget table; at the high end seed 1 and extensions could not fit under $1,000 | **adopted** — §6 carries scenario A (GPU generation, $400–715) and scenario B (hosted, $465–895), an explicit allocation order, and the statement that at the high end nothing optional fits |
| M | D-NCE not executable: logits, denominator, positive's template, reduction, τ reuse, missing seeds | **adopted** — the literal 129-way softmax cross-entropy is in §Recipe, τ reuse disclosed, removed-seed fallback to L2 counted |
| M | extension rule not mechanical: baseline undefined, partial cycles, dollar→hour conversion, no ledger fields | **adopted** — m_k − max(m₁…m_{k−1}) ≥ 0.003, whole cycles, `max_extension_cycles` fixed at the lock from dollars minus mandatory lines at measured rates, hard stop on projected spend, ledger fields |
| M | decision 8 contradicted "full-dose replicas stay waived"; "if the lock says so" a loophole; STATUS weaker | **adopted** — waiver sentence replaced; seed-1 boolean, seeds and row labels locked before any seed-0 six-set output; seed 0 alone controls every action; STATUS repeats the ≥ 100 GPU-hour criterion |
| M | generation budget assumed 60 output tokens for every form; two forms allow 400 | **adopted** — ≈ 250M generated tokens (350M at caps), 30–60 GPU-hours, fallback gate on the end-to-end projection at 60 h |
| M | parity sample and `results/perquery.json` had no cloud source or hash check | **adopted** — perquery.json is tracked and sha-verified on the instance; the parity sample regenerates from `m9/registry.json` (tracked, sha pinned) and `port.py` refuses a mismatch, in which case the 512 texts transfer from the box |
| M | rates provisional; measuring only the first arm is too late; 46 h understated | **adopted** — a day-one rate benchmark (encode, three training mixes, generation smoke, billed price) re-derives §6 before generation and again before the remaining arms; 47 h |
| m | STATUS still scheduled a Mac 4-bit generator pass | **adopted** — renamed prompt prototyping that produces nothing entering the smoke record, the data, or the manifest |

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
random; parity does not depend on them). Two serving pitfalls (both hit and resolved in M11, see `m11/CODEMAP.md`): fastembed needs
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

## 11. Measured trainer rates (box, 2026-09-04) — the plan review's central number

`results/m10_rate_bench_box.json`, script `m10src/rate_bench.py`. Random token ids only; no corpus,
no protected path, no teacher, no data loading, no evaluation — it bounds the **hardware**, not the
pipeline. M10 recipe shape: bge-small + Linear(1152→1024) per token over pooled layers 12/8/4,
34.54M parameters, bf16 autocast, fp32 loss. Pass 2 (cite this one): single homogeneous chunk per
step — what length bucketing produces — 300 timed steps, fused AdamW, peak memory and allocator
retries logged.

| shape | steps/s | examples/s | padded tok/s | peak GB | retries |
|---|---|---|---|---|---|
| bs32 × 64 (query bucket) | 22.4 | 718 | 45.9K | 0.88 | 0 |
| bs32 × 128 (mixed padding) | 22.0 | 702 | 89.9K | 1.29 | 0 |
| bs32 × 256 (document bucket) | 18.6 | 596 | 152K | 2.11 | 0 |
| bs64 × 64 | 21.1 | 1,350 | 86.4K | 1.29 | 0 |
| bs128 × 64 | 18.1 | 2,311 | 148K | 2.11 | 0 |
| bs128 × 256 | 5.8 | 748 | **191K** | 7.03 | 0 |
| bs512 × 64 | 5.7 | 2,893 | 185K | 7.04 | 0 |

Pass 1 (60 steps, kept for two rows only): M9's **two-chunk** collate at bs32 gives **400 ex/s**
against pass 2's 702 for the same examples in one padded chunk — the 1.9× that makes length-bucketed
single-chunk batching part of the build. Plain vs fused AdamW: 400 vs 397, so the cost is not the
optimizer.

**Reading.** Launch-bound at batch 32 is now *established*, not asserted: 4× the tokens per step
(32×64 → 32×256) costs 20% more wall-clock, and 4× the batch at fixed length (32×64 → 128×64) costs
24%. The card's roof is 185–191K padded tok/s, reached only at 128×256 and 512×64; batch 32 runs at
24–80% of it. §5's planning rate — 560 ex/s, LEAF's realized A100 rate — is therefore *met or
beaten by the 3080*, and M9's realized 226 ex/s was a pipeline artifact rather than a hardware
limit. Build cost under length bucketing, 200M examples at the 75/25 example mix (150M query-bucket
+ 50M document-bucket, each at its own rate): **81.3 GPU-hours at bs32, 36.6 at bs128** — bs128 is
2.2× cheaper, which family E must weigh against any bs32 quality win.

**Corrected after the Fable review (2026-09-04).** Pass 1 read its slow bs512 two-chunk row as
"past the launch-bound regime". Wrong: padded tok/s *fell* to 66K there, and a compute-bound card
plateaus rather than dropping 2.9× below its own roof. Pass 2 shows 512×64 and 128×256 both reach
185–191K at 7.0 GB with zero allocator retries, so pass 1's row was memory pressure on a 10 GB card.
Two consequences: an 80 GB A100 is **not** launch-bound at bs128+, so family E's cost ratio and the
cloud build line are device-dependent and must be re-measured there; and the document bucket — the
shape 25% of build steps will run — was untimed in pass 1 and is the expensive one.

**The caveat that binds, and it is the reason nothing is committed on these numbers.** These are
hardware bounds. M9's realized pipeline ran at **18,984 tok/s** (`m9/M92_LOCK.md`) against a
comparable-shape roof of ~191K here — roughly **10% efficiency** — and the M10 trainer is being
ported from that pipeline (`m9src/longrun.py`: per-step numpy collate, `length_chunks`, target
fetch). So the build is priced as a **range from hardware bound to M9 efficiency**, and box-window
item 1 (real-data re-measure with `torch.cuda.memory_stats()['num_alloc_retries']` logged) is the
gate before any dollar or box-versus-cloud decision. At M9's efficiency the box build is 300+ hours,
not 81, and a screen arm is 6–9 hours, not 2 — which is why closing that gap is worth more than the
GPU rental (§13, idea 1).

## 12. Why generate at all, and why ≈1.0M rather than 3.0M (the 2026-09-04 question)

Dylan: *"I'm not sure why we need to generate synthetic data?"* and *"isn't synthetic data lower
quality?"* Research sweep: one Sonnet subagent, web only, 2026-09-04.

**The mechanism.** Phase 1 is `‖student(x) − stella(x)‖` — there is no label, and the teacher's
embedding of any text is a correct target by construction. So the classic synthetic-data failure
(a wrong relevance judgment) cannot occur here; the exposure is only (i) distribution shift, and
(ii) diversity collapse. Both are measurable before a training step, which is what amendment A8's
two gates do. Every comparable system is *more* exposed than we are — LEAF, Qwen3-Embedding and the
InPars/Promptagator lineage all pair synthetic queries with implied positives — and they worked.

**Why real queries cannot substitute.** LEAF's breadth came from MS MARCO 502K (non-commercial),
Amazon QA 979K and CC-News 900K (no commercial grant) and PubMedQA 272K (contaminates NFCorpus and
TREC-COVID). Our permissively-licensed real-query stock is factoid Wikipedia QA plus ESCI product
search — exactly M9's two forms. Generation is a licence workaround, not a scientific preference,
and the report says so.

**But most of the breadth is harvestable as real text**, which is amendment A2: titles, headings and
declarative lead sentences already sit in the licensed pool, and **three of the four clean-4
headline datasets fall in that half** (scidocs↔titles, scifact↔claim sentences,
trec-covid/nfcorpus↔headings and consumer-health). Generation is then confined to the interactive
forms no corpus contains. Indirect support: arXiv 2502.19712's query-type ablation on TREC DL19/20
has titles-only at 71.6–72.9 against real-MS-MARCO-style-only 72.8 and the six-type mix at
72.4–73.2, while claims-only (a single generated form) sits at 65.3–68.0 — form diversity, not
LLM origin, is what pays.

**Why 1.0M and not 3.0M.** Every scale curve found saturates below it: DistilVDR (arXiv 2608.10636,
Aug 2026 — asymmetric, pure cosine-regression, 70M query tower from an 8B teacher) saturates above
75% of a 1.49M pool, with ~5 points from quarter-scale to full; SPEED (2410.18634) is log-linear
only to ~920K and its sub-generators plateau at 10K–50K; doc2query-style scaling (2509.16442) gets
90–95% of maximum gain at 50–75% corpus coverage; mxbai-edge-colbert's distillation stage used
1.45M queries against a stella-1.5B teacher. **Nobody has published 100K vs 1M vs 3M**, so this is
a prior, not a measurement — which is why A4−A3 is a registered screen contrast that can drop the
generated half entirely.

**Corrections to §3 the sweep forced.**
- LEAF's loss is the L2 **norm** `‖e‖₂`, not squared L2, and its **Appendix B added distillation
  terms to plain regression and found no improvement**. §3's "Ranking-aware KD" row is evidence
  about *contrastive* systems, not about adding a term to embedding regression — the basis for
  cutting family D to one arm (amendment A1).
- LEAF's student is **MiniLM-L6-v2 plus a linear projection**, initialized off-the-shelf, document
  side frozen — the same architecture as ours, which is why its 97.7% is the right aim.
- LEAF pools the **last layer only**. Multi-layer pooling (family G) is ours, evidenced only by
  §9–9b's two Mac probes, so G's losing arms are load-bearing for the paper.
- LEAF's 800K "vocabulary" texts are Claude-generated definitions, on the *document* side — so even
  LEAF's corpus is partly synthetic.
- Nothing on-topic published 2026-08-01 to 2026-09-04 beyond DistilVDR; no new ≤35M permissively
  licensed embedding backbone released in that window.

## 13. Carried from the Fable review (2026-09-04) — adopted, and the one gap still open

Full findings and dispositions: `research/m10-fable-plan-2026-09-04.md`, tabulated in
`instructions-m10.md` §Amendment 2026-09-04. Read-exclusion audit clean.

**Adopted into the plan:** the arXiv scientific harvest source (§Data), form-balanced query sampling
(§Data), the registered plateau response and D-COV (§Recipe), family F at 20M with a third arm
(§Screen), fixed-sequence gatekeeping and the four planning proxies (§Goal), the resolution-number remedy
(§Surfaces — struck the same day by the Codex pass), hosted generation as the default, and the benchmark re-run (§11).

**Not yet done, ranked, with cost — these are the next actions, not ideas:**

| # | action | cost | why it is worth it |
|---|---|---|---|
| 1 | `torch.compile(mode="reduce-overhead")` / CUDA graphs on the fixed length buckets, then a real-data re-measure | 0 GPU-h, ~2–4 h engineering | §11's caveat is the whole cost story: M9's pipeline ran at ~10% of the hardware roof. Closing that gap is worth more than the GPU rental, and 1.5–3× at bs32 also dissolves most of family E's cost argument. **Box-window item 1** |
| 2 | arXiv licence evidence and harvest yields | CPU + ~500K stella encodes ≈ minutes | amendment A2's scientific forms do not exist without it, and the fallback (revert to generation) must be chosen on a measured yield, not assumed. **Box-window item 5** |
| 3 | `nq-250k` retention as a named per-arm diagnostic | ≈0 | see the open gap below |

### The gap the plan still does not answer: the in-distribution ceiling

M9 retained **93.8% on NQ while training on NQ-like data**. Coverage explains the 50–71% on forum
questions; it does **not** explain the 93.8%. LEAF reaches 97.7% overall on an easier target
(109M/768-d teacher). The four planning proxies sit at 89.3% / 91.6% / 91.3% / **94.9%** of the
ceiling (§Goal, 0.025 quantile), so **C2b demands better in-distribution retention than M9 achieved even where it was
fully covered.** Everything M10 adds — coverage, width, the optimizer regime — attacks the
out-of-distribution half. Only family G (width) and family E (regime) touch the covered half, and
D-COV is now the one arm aimed squarely at it.

This is recorded as an open weakness rather than closed. What would move it, in rough order of
promise: D-COV or another loss that weights document-discriminative directions; more output width
than 1536 (the parameter budget allows it only by shrinking the backbone); a longer dose on covered
forms; and — outside M10's premise — document-side co-adaptation, which is M16 and which every
≥96% near-zero-query system used. If M10 lands the release conjuncts but misses C2b, this section is
the reason, and it was known in advance.
