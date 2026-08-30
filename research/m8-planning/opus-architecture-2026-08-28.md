# M8 planning — Opus brainstormer: capacity and architecture (2026-08-28)

*Verbatim final report of an Opus 5 subagent briefed on m7/RECIPE.md, FINDINGS.md, EXPLORED.md,
instructions-m8.md, CLAUDE.md and the cited result artifacts. It ran its own cheap arithmetic
against the frozen artifacts (tokenizer fragmentation counts, size tables).*

---

## 0. Five framing corrections before any proposal

**(a) "Larger vocab = longer effective tokens" is mostly false for English.** Measured on the six's
frozen queries with the shipped tokenizer (30,522 WordPiece): **1.243 pieces per word** — inflated
by punctuation. Only **4.9% of word occurrences** split into multiple pieces. What splits is the
tail: **41.2% of distinct word types** and **65.7% of hapax types** (mean 2.62 pieces when split).
A 151K byte-level BPE sits at ~1.3 tokens/word on English too — it is not coarser, it is
*tail-coarser*. The vocab lever is **untying rare/technical words**, and should be priced on the
tail, not the average.

**(b) The tail-fragmentation split is exactly the clean-4 vs overlap-2 split.**

| dataset | teacher overlap | tok split % | table nDCG | BM25 | table − BM25 | retention vs teacher |
|---|---|---|---|---|---|---|
| scifact | clean | 22.3 | 0.6101 | 0.6791 | **−0.069** | 0.783 |
| nfcorpus | clean | 22.5 | 0.3124 | 0.3180 | −0.006 | 0.756 |
| trec-covid | clean | 13.7 | 0.5490 | 0.6099 | −0.061 | 0.667 |
| scidocs | clean | 12.9 | 0.1677 | 0.1565 | +0.011 | 0.700 |
| fiqa | stella-disclosed | 5.8 | 0.3728 | 0.2532 | **+0.120** | 0.673 |
| arguana | stella-disclosed | 4.2 | 0.5916 | 0.4878 | +0.104 | 0.929 |

Spearman(split%, table−BM25) = **−0.77**; Spearman(split%, retention vs teacher) = **−0.09**. Read
honestly: fragmentation tracks *where BM25 beats us*, and **not** where the table loses ground to
its own teacher. n=6, and split% is confounded with "technical domain". So this supports the vocab
lever as a *lexical-arm* story and does **not** support it as the explanation of the 0.755
retention gap. Anyone pitching vocab expansion as "the retention fix" is over-selling it.

**(c) Higher dimension is a provable no-op — kill it now.** stella's `config.json` has
`hidden_size: 1024` and every MRL head is a `Dense` with `activation_function: Identity`, i.e.
`d = normalize(W h + b)` with `h ∈ R^1024`. Any head with `out_features > 1024` produces document
vectors inside a **≤1024-dimensional subspace**. A 2048-d or 4096-d table adds **zero**
representable ranking capacity while doubling/quadrupling both the artifact and the document index.
The only live dimension question is *downward* (512/768 heads) — a byte lever (see P9).

**(d) The byte budget is 7.4x unspent against LightRetriever and ~1.0x unspent against the system
that actually beats us.** LR int8 233 MB / ours 31.3 MB = 7.4x. But bge-small-en-v1.5 — 0.5042 on
the six, **+0.070 above our table** — is 66 MB fp16 / ~33 MB int8, 5 ms/query, 1.3 s load. At
132 MB int8 (a 128K-row table) we would be 4x larger on disk than the small transformer we lose to,
keeping only latency (0.38 vs 5 ms) and cold start (0.22 vs 1.3 s). Consequence: **prefer capacity
that is (i) built at index time and never shipped, or (ii) paid in bits rather than megabytes.**

**(e) Expressivity is not the binding constraint; interference and domain are.** The capacity probe
fits dev queries to macro 0.99997. Adding rows only helps insofar as it reduces the *tying* that
forces one row to serve all contexts, or improves generalization. Every proposal must name which of
those it moves.

Cost basis throughout: one B16k→A2500 chain ≈ 3–4 GPU-h *(orchestrator note: the recipe agent
measured ~30 min for the same chain from logs; treat chain cost as 0.5–4 h pending a direct
check)*; one eval-only full-dev-suite pass ≈ 0.3–1 GPU-h; per-arm resolution on the out-of-domain
dev subset ≈ 0.005; recipe-perturbation band ≈ 0.005. Read every probe on the out-of-domain subset,
never the dev macro.

---

## 1. Proposals

### P1 — Index-time corpus adaptation of the table (top bet)

**What.** Stop shipping a *fixed* table. Ship a base table plus a **fitting recipe that runs at
index time**, on the deployment corpus, using **documents only** — no queries, no labels, no new
licensing surface. The server already runs the teacher over every document; additionally it draws
short spans from its own documents (existing `m7src/pseudoq.py` machinery), embeds them with the
teacher (~1% of corpus-encoding cost), and re-fits the rows by ridge **toward the shipped table as
the prior**: `(XᵀX + λI)W = XᵀY + λW₀`. Query time unchanged.

**Why it should work — three in-repo anchors.**
1. **We are the only system in the comparison that is not corpus-adapted.** BM25 is corpus
   statistics by construction. OpenSearch doc-v3 carries corpus postings + idf. The release bar we
   missed, **LR-dense-pertask 0.4583, is the *adapted* LightRetriever**; its unadapted sibling
   LR-dense-websearch is **0.4320**. That delta, **+0.0263**, is larger than our entire −0.0243
   miss. We lost to a bar defined by adaptation, using an unadapted artifact.
2. **Retention is a step function of familiarity.** Out-of-domain dev 0.764; six-set 0.755; on
   `heldout-train` (seen documents, unseen queries) the table **beats its teacher at 1.079**.
   Index-time adaptation moves a deployment corpus from the first regime toward the third. (Honest
   discount: `heldout-train` is also supervised and query-distribution-matched; realistic landing
   zone well below 1.079.)
3. The clean-4 weakness is a *domain* weakness, and this is the only lever that puts in-domain
   signal into the rows without any new data licence.

**Expected effect.** Wide: **+0.01 to +0.05** on the reserved four. P(>+0.01) ≈ 0.55, P(≥+0.025)
≈ 0.30. Largest variance and largest upside of anything here.

**Artifact size.** Released artifact **unchanged at 31.3 MB int8**. Adds a fitting script and a
per-deployment build step.

**Cost.** The fit is closed-form: Gram at V=30,522 is 7.5 GB float64 (inside budget; solve ≈ 107 s
per `m7_stage0_ridge_stella.json`). Span encode for 100–200K spans ≈ minutes. The real compute is
the reserved corpora themselves (FEVER + DBpedia ≈ 10.1M docs, 5–9 GPU-h), which M8 pays regardless.

**Risks.**
- **Protocol optics.** "Fitting on the eval corpus" will be attacked. Defence pre-registered before
  any number: documents only, a frozen script, query/qrel files unreadable to the fitting path
  (enforced in code), comparison labeled system-vs-system exactly as fusion is. The comparators all
  could do this too — BM25 already does.
- **Incremental corpora drift** — refit cadence; operational shape identical to BM25's idf, which
  nobody calls exotic.
- **Small corpora overfit** (scifact ≈ 5K docs): ridge-toward-W₀ with λ set by a pre-registered
  corpus-size rule; untouched rows stay at base.
- Doubles the "what is the artifact" explanation in the report.

**Probe (<2h).** On one out-of-domain dev component (cqadup-physics or programmers), corpus already
encoded: 100K spans → teacher-encode → ridge-toward-W₀ at 3 λ values → evaluate against the frozen
candidate. Bar: beat the base table by more than 0.005 on that component. A null closes it in an
afternoon.

**Touches.** Nothing closed. `pseudoq`'s own docstring labels itself "a VOCABULARY mitigation, not
a domain mitigation" — this is the missing domain mitigation. New axis, unpublished per
`research/m7-novelty.md` as far as it reaches.

### P2 — Spend the row budget: n-gram/word rows trained *through* the forward, implemented as a tokenizer swap

**What.** The one capacity direction M7's algebra left open. Two implementations; the second is
cleaner:
- (i) Additive n-gram rows on top of the 30,522 unigrams — pooling must handle overlaps; bigrams
  double-count the unigrams underneath.
- (ii) **Tokenizer replacement (recommended)** — train **our own** unigram/BPE tokenizer on the
  approved clean corpora at V = 64K–200K, *without whitespace pre-tokenization* so multi-word
  merges are legal. Deterministic non-overlapping segmentation: no double counting, principled
  selection, and a free clean init (forward each token's *string* through the teacher —
  `init_table.teacher_rows` already does this for arbitrary strings). **The query tokenizer does
  not have to be the teacher's** — there is no shared model on the query side, only rows in the
  teacher's output space. Zero licensing surface; query-time cost unchanged.

**Expected effect — best-anchored number here.** The closed-form joint probes measure bigram rows
in the same solve as the unigram block: 5,000 rows **+0.0101** [0.0073, 0.0128]; 10,000 rows
**+0.0143** [0.0111, 0.0174] — **+0.0042 per doubling** across the comparable points. Log-linear
extrapolation to ~100K rows gives +0.028 dev-proxy; saturating fit less; honest closed-form band
**+0.018 to +0.030**. Discounted for the trained table already recovering some of it, for
n-gram-coverage transfer, and for dev→six shrinkage: **estimate on the six +0.010 to +0.020
dense**, P(>+0.01) ≈ 0.45. Expect roughly half to survive into the *fused* system (BM25 already
supplies phrase-ish lexical evidence) — **measure the lever fused, not just dense**.

**Artifact size** (int8 = V·d + 4V bytes): 64K → 65.8 MB; 128K → 131.6 MB; 151,936 (LR row parity)
→ 156.2 MB; 226K → 233 MB (LR int8 cap). At 4-bit or d=512, halve these.

**Training cost.** Init ~128K teacher forwards, minutes. Rows at 128K×1024 fp32 + Adam ≈ 2.1 GB
VRAM — fits. One chain unchanged. **The real cost is coverage**: a 128K vocab needs roughly 3–5M
spans (targets 6–10 GB fp16, memmapped); use **targeted coverage sampling** (draw spans containing
rare rows). Implementation ~1–2 days: `Preproc` fingerprint carries tokenizer identity,
`table.CLS_ID` and the conformance suite move with it, `init_table` generalizes.

**Risks.** Coverage tail (Zipf: types below rank ~15K get <100 occurrences at today's pool size);
greedy segmentation brittleness; **out-of-domain phrase coverage is the exact weakness we are
trying to fix** — a training-corpus-selected phrase list may not transfer to TREC-COVID at all.
That last risk is what makes P1 and P2 complements: index-time vocabulary extension puts the
corpus's own jargon into rows without shipping the bytes.

**Probe (<2h).** Closed form caps at V ≈ 45K (Gram is V² float64). **Use an iterative solver**:
X is sparse (~20 nnz/row), block-CG on `(XᵀX + λI)W = XᵀY` with all 1,024 right-hand sides — ~1 s
per iteration on the 3080, W at 128K×1024 fp32 = 524 MB, 50 iterations ≈ 1 minute. Run it on the
**current** 30,522 vocab too as the paired control, so the solver change is not confounded (the
"changing the solver breaks comparability" note binds the teacher sweep, not a new self-contained
frame). Deliverable: table-quality-vs-vocab-size curve in half a day, before any chain is bought.

**Touches.** The bigram close is explicitly scoped to *residual integration onto a trained table*;
the joint retrain is named still open in both `EXPLORED.md` and `instructions-m8.md`. The +0.0143
joint-solve probe is the evidence the joint frame is right; the failure was the frame, not the
lever.

### P3 — Bits, not megabytes: 4-bit / PQ rows (the enabler for P2)

G4 measured int8 as quality-free to an upper bound of 0.00013 against a 0.005 bar — large
unexploited redundancy. Sweep 6/4/3-bit per-row quantization and a PQ variant (gather + dequantize
is a handful of lookups). Payoff: **the row budget doubles or quadruples at fixed bytes.** If 4-bit
lands within 0.002, a 128K-row table ships at **66 MB** — below bge-small fp16. P ≈ 0.6. Cost:
eval-only, 1–2 GPU-h. **Run first** — its answer sets the size envelope every row-spending proposal
is designed against. Risks: quantization × ANN interaction (check with `ann_sweep.py`); int4 gather
kernel for the Edge story.

### P4 — A learned doc-side linear map (2 MB, index-time GEMM) — a lever dismissed on a wrong reason

**What.** Learn `M` (1024×1024, ~2 MB fp16, init identity, regularized toward it) jointly with the
rows; apply server-side at index time to the **cached** teacher vectors, renormalize. Query side
untouched.

**Why it is real capacity.** `m7_absorb_check.json` is explicit: `q·(Md) = (Mᵀq)·d` holds only
without renormalization; with it, rank agreement with the absorbed form is **0.000**. The
per-document factor `1/|Md|` is a learned, content-dependent document prior — structurally what
BM25 gets from length normalization and single-vector dense retrieval lacks.

**And the stated reason for dropping it is wrong.** `m7/LEDGER.md` says "changing the document map
means re-encoding the corpus with a different teacher." It does not: a linear map applied to
already-cached teacher output vectors is one GEMM over the index, no transformer re-run. A genuine
oversight, not a closed avenue.

**Expected effect.** +0.003 to +0.010, P(>+0.005) ≈ 0.35. Modest, very cheap, and one of only two
levers here that can reshape the *document* space toward bag-linear approximability.

**Risks.** 1.05M parameters vs 340,850 pairs — regularize toward `I`; re-check ANN behavior; the
released system becomes a pair of artifacts ("31.3 MB table + 2 MB doc projection").

**Probe (<2h).** Freeze the shipped table, fit `M` alone on cached pairs (minutes), evaluate on the
out-of-domain dev components. If frozen-table `M` cannot clear 0.005 there, the joint version is
unlikely to be worth a chain.

### P5 — Routed mixture of tables (sense rows without a per-sense oracle)

Base table + K low-rank deltas; at query time gather → mean → K dot products → argmax → re-gather
with the selected delta. Global route (topic switch) or per-token route (polysemy attack).
Non-absorbable (piecewise-linear in the bag). K=8 experts at rank 32 ≈ **7.8 MB** int8. Expected
+0.005 to +0.015, P ≈ 0.25. Real risks: straight-through/soft-to-hard training mismatch, routing
collapse, double gather, most complex to get right. No honest <2h probe — smallest test is one
chain with K=4 + a pre-registered collapse check. Rank below P1/P2/P4 for that reason alone.

### P6 — Low-rank bilinear consecutive-pair term (n-grams without n-gram rows)

`q = normalize(mean_i(r_i) + λ · mean(U r_i ⊙ V r_{i+1}))`, r ≈ 64; ~25K flops per 20-token query.
Non-absorbable; gives **every** bigram capacity from a shared tensor — no coverage/tail problem
(exactly P2's weakness); asymmetric U≠V on consecutive pairs adds order-awareness. 0.26 MB.
Expected +0.002 to +0.008, P ≈ 0.20 (a rank-64 factorization of a phenomenon the explicit-row probe
reached +0.0143 on — likely weaker). Worth one chain only if P2's probe says n-grams are alive but
coverage is the problem.

### P7 — Pooling structure beyond `sqrt` (weakest of the non-absorbable levers)

Multiplicity-dependent pooling is legal capacity but measured returns are poor (`sqrt` +0.0040 on
one artifact, failed to replicate on the next; lever #6 arm (a) failed). Untried members: (i)
elementwise **max/mean blend** `normalize(a·mean + b·max)` — an order statistic, genuinely outside
the linear family, 0 MB; (ii) w(token, length-bucket) non-separable weights ~0.24 MB; (iii)
position buckets. Expected +0.000 to +0.004, P ≈ 0.15. Do not spend a chain unless a free rider; if
the arm is redesigned per `instructions-m8.md`, the max blend is the interesting half, not counts.

### P8 — The lexical arm (attacking the +0.057 fusion bright spot)

**Vendor/licence check first, because it kills the obvious options.** SPLADE v2/v3 weights are
CC BY-NC-SA → fails commercial release. `opensearch-...doc-v3-gte` is excluded twice (vendor +
circularity). Every strong public learned-sparse model is MS-MARCO-trained. So "swap in a learned
sparse arm" is **not available off the shelf**; training our own doc-side term-weighting model on
the clean stack is an M9-sized sub-project.

What is actually cheap:
- **Fusion re-selection** at the new candidate (mandatory anyway; free).
- **A zero-compute query-adaptive fusion weight**: `w = f(query length, max/mean idf, fragmentation
  rate)` — computable from token ids, no neural compute, not absorbable into either arm. Motivated
  by the per-dataset spread of the fusion gain (trec-covid +0.153 … arguana +0.006). Risk: prime
  overfitting territory — pre-register the feature set, fit on dev components only.
- **doc2query at full dose** stays blocked on Dylan's generator ruling; the M7 close was explicitly
  "at the cheap-test price, not disproved" (+0.0054, p=0.085 at 1/8 dose). If the ruling lands, a
  real +0.005–0.015 doc-side lever.

Expected from the cheap half: +0.003 to +0.010 **on the fused system**, P ≈ 0.30. The cheapest path
from "ties OpenSearch" to "CI-resolved above OpenSearch".

### P9 — Dimension *downward* (a byte lever)

stella publishes 512/768 MRL heads. The Gram in a closed-form probe is d-independent; targets at
all dims come from one encode pass. A 512-d table halves the artifact (31.4 → 15.8 MB) **and the
document index (2.05 → 1.03 GB per 1M docs)**; at fixed bytes it buys 2x rows for P2. Expected
quality cost 1–2 points (MRL typical); P(within 0.005) ≈ 0.4. Probe <2h once P2's solver exists.
Combined with P3: a 128K × 512 4-bit table is **33 MB — 4x LightRetriever's vocabulary at today's
artifact size.** That is the size story worth telling.

---

## 2. Ideas that are absorbable or no-ops — flagged, per the algebra

Do not let these into a plan; each is refuted by `m7_absorb_check.json` or one line of algebra:

| idea | verdict |
|---|---|
| Softmax "attention-lite" pooling over learned per-token logits | Absorbable: softmax denominator is a per-query scalar killed by L2; what remains is exp-scaled per-token weights. |
| Any per-token scalar (IDF, SIF, stopword dropping, per-row renorm before averaging) | Absorbable, proven to 9e-14; trained weights are already IDF-like. |
| Learned constant vector added to the bag before normalizing | Absorbable: mean(W_t)+b = mean(W_t+b). |
| Query-side centering / whitening / top-PC removal | Absorbable, machine precision. |
| Length scaling 1/sqrt\|T\| | Vacuous under L2. |
| Higher table dimension (2048/4096/8192 heads) | No capacity: teacher hidden is 1024-d, heads are Identity-activation linear. Pure cost. |
| Concatenating two independently trained tables | (q₁+q₂)·d → absorbable into their sum; per-arm-normalized variant is a marginal reweighting. |
| Per-token ReLU scoring (SPARTA-style) | Genuine capacity but breaks the single-vector ANN path — needs an architecture renegotiation. |

---

## 3. Ranked top 5 by expected value per week

| # | lever | EV on reserved four | weeks | probe | why |
|---|---|---|---|---|---|
| 1 | P1 index-time corpus adaptation | +0.01…+0.05 (P≈0.55 of >+0.01) | 1–1.5 | 2h, one OOD dev component | Only lever whose in-repo analogue (pertask − websearch = +0.0263) exceeds the whole miss; ships zero extra bytes; attacks the diagnosed cause. |
| 2 | P2 vocab/n-gram rows via trained-through tokenizer | +0.010…+0.020 dense (P≈0.45) | 2–3 | half-day CG vocab curve | Best-quantified lever (+0.0042/doubling, measured twice). Byte cost is why P3/P9 sit under it. |
| 3 | P4 doc-side learned linear map | +0.003…+0.010 (P≈0.35) | 0.5 | 2h frozen-table fit | Cheapest genuine capacity; corrects a dismissal made on a wrong reason. |
| 4 | P3 bit budget (4-bit/PQ) | 0 quality, 2–4x rows | 0.3 | 1–2 GPU-h eval-only, run FIRST | Frame lever: decides whether P2 ships at 132 or 66 MB — whether P2 is defensible against bge-small at all. |
| 5 | P8 fusion mechanics + zero-compute adaptive weight | +0.003…+0.010 fused (P≈0.30) | 0.5 | free | Cheapest route from tie to CI-resolved win over OpenSearch. |

Below: P9 (inside P2's probe), P5 (highest implementation risk), P6 (conditional), P7 (no chain).

## 4. The single bet

**If only one lever runs: P1, index-time corpus adaptation.** We missed a bar that is itself
defined by an adapted artifact, using the only unadapted artifact in the comparison
(LR-dense-pertask 0.4583 vs LR-dense-websearch 0.4320: +0.0263 > our −0.0243 miss). The repo's own
retention ladder (0.764 → 0.755 → 1.079 on seen docs) says corpus familiarity is worth more than
any capacity knob measured. It ships no extra bytes, keeps query time exactly, costs no licensing
surface, is closed-form, and turns the deliverable into something Qdrant-shaped — a build-time step
in the index pipeline. Unpublished in this form as far as `research/m7-novelty.md` reaches.

Mind-changers: the 2-hour probe failing to clear 0.005 on an out-of-domain dev component; or Dylan
judging the protocol optics unacceptable — in which case the bet moves to **P2 + P3 together**
(n-gram rows at 4-bit, ≤66 MB), same upside for three times the weeks and a real out-of-domain
coverage risk.
