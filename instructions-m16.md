# M16 — a better `constella-zero` (idea register)

**Parked, long-term, may never run** (Dylan, 2026-09-03). This is an idea register, not a mandate:
every idea for beating M7's table with the evidence it rests on, written down while the evidence is
fresh so a future session does not re-derive it. Nothing here is scheduled and no compute is
committed. Read `CLAUDE.md`, then `m8/FINDINGS.md` and `m8/EXPLORED.md` before proposing anything:
M8 measured twelve levers and those two files say which ideas are already dead.

## The gap to close

| target | avg-6 | `zero` is behind by |
|---|---|---|
| `zero` released (int8 table) | 0.4339 | — |
| `LR-dense-pertask` (release bar) | 0.4583 | −0.0243, CI-resolved |
| `opensearch-doc-v3-gte` (the aim) | 0.4868 | −0.0529 |
| stella-400M symmetric (retention ceiling) | 0.5744 | −0.1405, i.e. 75.5% retained |

**No table-side lever M8 measured moved the dev endpoint more than ~0.005.** The gap is five to ten
times that, which is the case for Group A. The pyNIFE retention result below is the one piece of
evidence that a table-side recipe change might still be worth ~0.035, and it is why Group B is not
closed.

## The premise M16 reopens

Every M7 and M8 lever ran against a **frozen off-the-shelf document tower**. That constraint is
where the deficit lives: M8's own reading is that what was never tested at capacity is
**co-adaptation of the document side, which is exactly what the system we lost to does**
(`m8/FINDINGS.md` §1). LightRetriever's table works because its document space was co-trained to be
additively predictable from query tokens; stella's space never was (`instructions-m7.md`, Mission).

So Dylan's read is right: **the largest untested ideas are on the tower, not the table.** Reopening
the frozen-tower premise is his call and needs an explicit answer (`CLAUDE.md`, "Past decisions are
revisitable"). The trade it makes:

- **Lost:** dropping into stella's existing index. That property is the current product pitch and
  the whole reason `zero` is a drop-in for anyone already running stella.
- **Gained:** a tower whose training data we control, which removes the stella contamination
  caveat (ArguAna and FiQA of the six, FEVER of the reserved four) from every headline claim.
- **Cost:** the document index is re-encoded once, and the released artifact becomes a *pair* of
  weights instead of a table.

## Group A — document-side co-adaptation (largest untested)

| idea | what it rests on | what would kill it |
|---|---|---|
| **A1 · `E14-LORA` at a real budget.** LoRA on stella, trained jointly with the table. | Authorised by Dylan in M8 and **deliberately not run**: only a dev-scale proxy against a shrunken stale negative bank was affordable, which is the same weakness that made `E14-HEAD` uninformative (`m8/FINDINGS.md` §2). | A budgeted arm that fails to clear the chain floor 0.00519. Nothing cheaper closes it — the proxy already didn't. |
| **A2 · Train the tower FOR distillability.** Select or adapt the document space on the table it yields, not on its own retrieval score. | Spearman(teacher ceiling, distilled table) = **0.000** over eight candidates (`m7_learnability_report.json`). Distillability is therefore a free-floating property nobody has optimised directly. | A pilot showing the tower's distillability cannot be moved without costing more retrieval quality than the table gains. |
| **A3 · Full co-training, LightRetriever-style.** Both sides trained together from the start. | The strongest form of A1, and the mechanism behind the number we lost to. | Compute. Price it before proposing it. |

A1 is the entry point: it is the cheapest arm that tests the premise, and it is already authorised.
The pyNIFE retention comparison below does not weaken Group A — it adds a second, cheaper front.

## Group B — table-side, still untested

| idea | what it rests on | prior |
|---|---|---|
| **B1 · `R-LIST` — hard-candidate listwise distillation.** | `B2` measured the shipped objective **inert**: uniform-bank KL median 4.73e-07 nats, and the table already ranks the positive first for **99.75%** of training queries. The `teacher_top200` variant measures **0.777 nats**, so the KL class is open and only the degenerate instance is closed. | The strongest untested table-side lead M8 leaves. Class prior is poor (M7's whole lever programme transferred at 0.000 ± 0.005). |
| **B2 · `B10` — multiplicity-dependent pooling.** | Pooling multiplicity and new rows are the **only** two things that add capacity to a table; everything else on the query side is absorbable (`m7_absorb_check.json`). New rows are closed, so this is the surviving half. | +0.0011, CI straddling zero. Untested. |
| **B3 · Distil in cosine space instead of L2.** | pyNIFE reports cosine beat MSE and KL outright; M9's plateau was diagnosed as a linear head that L2 regression cannot push past ~90–93% (`m10/PLANNING.md` §9). | Cheap arm, no new data, no new asset. Run it before anything expensive. |
| **B4 · Drop the instruction prefix from distillation.** | A static model can only ever treat an instruction as a constant offset, and `m7_absorb_check.json` proves a constant offset is absorbable — so the prefix spends rows on something the table cannot use. `zero` currently carries stella's query prompt. | Cheap, and algebraically supported rather than guessed. |
| **B5 · Larger vocabulary, tokenizer-first.** Retrain the tokenizer before distillation (pyNIFE uses ~100k rows), train every row jointly. | **Only reopens `D2` in this exact form.** `D2-PRE` closed additive n-gram rows and multi-word segmentation *at equal budget on frozen incumbent rows*, and found two-thirds of 35,014 added rows inert. A tokenizer trained before distillation with all rows learned is a different design, not a re-run. | Costs table bytes: 100k × 1024 int8 = 103 MB, still under LightRetriever's 466 MB. |

## Group C — already measured, just ship it

**Fusion is worth ten times any table-side lever.** M7 measured +0.057 over dense alone, and the
fused system (0.4911) ties OpenSearch (0.4868). A zero-compute product ships fused. Carried here
only so no M16 plan forgets it while chasing the dense scalar.

## Do not re-propose

One line each; the reopening condition lives in `m8/EXPLORED.md`, which is the canonical register.

| closed | by what |
|---|---|
| Query-side centering, whitening, top-PC removal, per-token scalar weights | Absorbable into the rows, proved to machine precision (`m7_absorb_check.json`) |
| Post-hoc linear projection into a frozen doc space | M2/M7: potion-32M→arctic-m reaches 0.3280 against its own 0.3427, with test-set-tuned regularisation |
| A doc-side head on a finished document vector | `E14-HEAD`: LIN −0.0244, MLP −0.0293 |
| Doc-centroid training targets | `B8`: −0.167; a 50/50 blend −0.003 |
| Train-free dense PRF | `VECTOR-PRF`: −0.051 dense, −0.021 fused, negative on all six |
| More Phase-A pairs | `B3`: +0.00135 at 4× dose; the bar needs ~17.6× the pool |
| Additive n-gram or segmentation rows on frozen rows at equal budget | `D2-PRE`: every arm negative out-of-fold, best −0.0028 |
| A better off-the-shelf teacher | `T1`: granite-r2 −0.052, gte-modernbert −0.109 |
| MS MARCO in the release stack | Non-commercial terms; the exclusion is priced and unresolved in both directions |

**One caveat on the teacher row.** `T1` scored gte-modernbert at −0.109 in a shared student frame
that assumes stella's `bert-wordpiece-30522` vocabulary; gte-modernbert does not ship it, so that
number is teacher **plus** tokenizer, which `m8/EXPLORED.md` already labels as such. pyNIFE distils
the same teacher with its own retrained tokenizer and retains 73.4%. That is direct evidence the
−0.109 was substantially the tokenizer, and it is the second argument for **B5**.

## pyNIFE, measured head to head (2026-09-03)

pyNIFE is `zero`'s construction, published before M7 started, and it is **prior art** rather than a
comparator: see `research/m7-novelty.md` §pyNIFE for the withdrawal. What it did not have is a
number on a full BEIR test split. Measured on BEIR fiqa, 57,638 documents, 648 judged queries, one
index per teacher, same engine and `hnsw_ef=128` on both:

| system | fiqa nDCG@10 | its teacher | retention | query asset | encode p50 |
|---|---|---|---|---|---|
| `zero` | 0.3713 | stella-400M, 0.5531 | **67.1%** | 94.5 MB | 0.034 ms |
| pyNIFE | 0.3353 | gte-modernbert-base, 0.4570 | **73.4%** | 156.0 MB | 0.090 ms |

**Two readings, and they point at different groups.**

- **Absolute quality: `zero` wins by +0.036**, on a 1.7× smaller query asset. Teacher choice is why
   — stella is 0.096 nDCG ahead of gte-modernbert on this corpus.
- **Retention: pyNIFE wins, 73.4% against our 67.1%, on the same corpus.** Their recipe extracts
  more of its teacher than ours does. The three things it does differently are exactly **B3**
  (cosine-space distillation), **B5** (a tokenizer retrained before distillation, ~100k rows) and
  **B4** (no instructions in the distillation target).

So Group B is **not** the arithmetically-dead half it looked like from M8's lever programme alone.
Closing the retention gap on fiqa is worth roughly **+0.035** at stella's ceiling — seven times any
lever M8 measured. Treat that as a signal to run B3 first, not as a projection: it is one dataset,
not paired, on one of the two of the six that stella's disclosed training data touches, and a
stronger teacher may be intrinsically harder to distil because more of its quality comes from
contextual interaction the table cannot represent.

**Also worth recording:** pyNIFE reports 89.2% retention on NanoBEIR (59.2 against 66.34). On a
full BEIR test split it retains 73.4%. NanoBEIR's 50 queries per set flatter this architecture, so
do not compare our numbers to anyone's NanoBEIR row.

## Inherited protocol — unchanged, and not negotiable retroactively

- **Reserved four** (FEVER, DBpedia-entity, cqadup-android, cqadup-english): **one** confirmatory
  access, unspent through M9. `results/perquery.json` is irreplaceable.
- Bars, comparators and statistics come from `registry.json` and frozen comparator vectors; pairing
  never re-runs a comparator.
- Every claim in Group A or B is measured against the noise model M8 calibrated: A-leg σ ≈ 0.00106,
  chain σ = 0.00153, so 0.0040 binds A-leg arms and 0.00519 chain-varying arms.
- Student cap 35M is a hard rule (Dylan, 2026-09-01) and binds anything M16 adds on the query side.

## What would have to be true for M16 to run

Recorded so the question does not get re-opened from scratch. None of this is pending; it is the
shape of the decision if the milestone is ever picked up.

1. **The frozen document tower is reopened.** Group A is the whole reason M16 would beat M7, and it
   costs the drop-in-to-stella's-index property. Without that, M16 is Group B and Group C.
2. **A GPU budget exists**, on the same footing as M10 (`instructions-m10.md` §Compute). M10 is
   already paused waiting for one, and M16 does not jump that queue.
3. **M10's status is settled first.** M10 is the nano retry on the same frozen tower. If the tower
   unfreezes, M10's recipe lock rests on a premise M16 discards, so the two are resequenced rather
   than run in parallel.

**If only one thing here ever runs, run B3** (cosine-space distillation). It needs no new data, no
new asset, no budget approval, and the retention comparison above prices it at roughly +0.035.

## Inherited from M12 (2026-09-04): fusion-aware training of the table

`constella-zero-hybrid` — train the table against the FUSED ranking so it stops competing with BM25
and covers what BM25 misses. Cut from M12 before any code was written; both broken drafts and the
evidence are in `m12/EXPLORED.md`. The idea is **not refuted** — M8 explicitly left hard-candidate
listwise training open (`m8/FINDINGS.md:51`) — it is **unfunded and unimplementable as specified**.

**All of these must be true before it runs again:**

1. **A trainer that carries per-query candidate lists with aligned lexical scores.** Today training
   scores one shared bank of 32,768 negatives per batch (`m7src/train.py:679-688`); both miners
   return IDs only, and `mine_bm25_negatives` mines within each query's own store (`:217-220`), not
   the global pool. 1–3 engineering days, plus a cache schema for raw scores and normalisation stats.
2. **Both retrievers searching the same corpus**, or the union does not resemble deployed fusion.
3. **A registered operator-specific loss**, with the detach decisions named. RRF is untrainable
   (piecewise constant); DBSF is differentiable only with its statistics detached, which is a
   surrogate choice Qdrant does not specify.
4. **A measured noise floor for per-arm fusion re-selection.** The 0.004 fused floor is frozen-`w`
   and does not calibrate it; `m7/LEDGER.md:655-656` requires re-selection when the checkpoint changes.
5. **A bar derived from that floor**, not from the historical recipe band. At the 0.005 effect size
   every table-side lever has produced, a 3-vs-3 design clears 0.008 with probability ≈ 0.

Ranked below document-side co-adaptation, which remains this milestone's lead: it is the genuinely
untested capacity lever, at a plausible δ ~0.02 against this idea's ≤0.005 class.
