# M12 — `constella-zero-hybrid`: a table trained to be BM25's teammate

Created 2026-09-04 (Dylan). Everything binds from `instructions-m7.md` unchanged: decision
authority, licensing and decontamination rules, dev-only selection, the freeze/ledger protocol, the
headless git contract. Working files under `m12/`, branch `m12-work`.

**Runs on the local 3080.** No cloud budget, no A100. M10 stays paused and M12 does not touch it.

## The question

`zero` is trained to imitate stella's query vector — a dense-only objective. But the shipped
headline is hybrid (0.4911 fused vs 0.4868 OpenSearch). A bag-of-token-vectors table is *naturally*
good at the lexical matching BM25 already does, so under fusion the two partly duplicate each other.

**Does training the table against the FUSED ranking beat training it to imitate a dense teacher?**

Miss-is-publishable. A NO is a real finding: it says the table has no non-lexical capacity to
redistribute, which is the sharpest statement anyone has made about this architecture's ceiling.

## Gate 0 — headroom, before any training code

Fuse **stella itself** with BM25 on the four dev components and compare to fused `zero` (0.5727).
Inputs all exist: `dev-{comp}-queries-pfx` and doc vectors in `work/enc`, BM25 runs in
`work/fusionruns/bm25-{comp}-d1000.npz`, the release table `work/runs/p35w-2m-s2500.release.npz`.
Reuse `select_fusion.dense_run` with teacher vectors in place of `model.encode`, `fusion.select_on_dev`,
`evalkit.per_query_ndcg`. ~15 min. Dev-only, descriptive, spends no confirmatory access.

Free by-product from the same per-query arrays: bucket the teacher−zero dense loss by BM25
per-query nDCG. **If the loss sits on queries BM25 already answers, a hybrid objective has nothing
to redistribute** and Gate 0 fails on mechanism, not just magnitude.

**Dense gap is 0.6350 − 0.5370 = 0.098; that is the ceiling.** The number that matters is
fused(teacher) − fused(zero) — the part fusion has NOT already recovered.

**KILL: if that residual is below 0.005** — the band every M7/M8 lever landed in — M12 stops here
and the finding is written up. Do not proceed on hope.

## Gate 1 — pick the fusion operator the product can run

**Pre-register before Gate 2, and this is the decision most likely to waste the milestone.** M7's
`convex0 w=0.8` per-query min-max over `bm25s`-lucene (`m7src/fusion.py:26-29`) is **not what Qdrant
ships** — native hybrid is RRF/DBSF, and FastEmbed's `Qdrant/bm25` uses a fixed `avg_len`, not corpus
avgdl. Best dev RRF was **0.5504 vs convex0 0.5727** (`m7/LEDGER.md:922`): a 0.022 gap, larger than
every table-side lever ever measured here.

Train against the operator that will actually run in Qdrant. If that is RRF, the Gate 0 headroom is
re-measured under RRF before Gate 2 — a residual computed under convex0 does not license training
under RRF. Record the choice and the re-measured residual in `m12/LEDGER.md`.

## Gate 2 — train it

One recipe, not a sweep. Warm-start from the released `zero` table; keep the M7 vocabulary,
preprocessing, row indexing and int8 output unchanged, so the artifact stays drop-in to the same
stella index and the same 31 MB. The loss is a ranking loss on the **fused** score with the dense
half differentiable and BM25 a frozen per-query constant.

Registered before the first run: loss form, the fusion parameter's treatment (co-registered, not
re-fitted after seeing results), the dev step-selection rule from `m7/LEDGER.md`, and the arm budget.
**Arm budget: 6.** M8 spent twelve probes to move nothing; M9's own findings log records 58 arms and
322 evaluations as overfitting the dev suite. Six arms or stop.

## Gate 3 — report both numbers, always

Every result row carries **fused AND dense-only** for the same table. The dense-only score is
expected to fall; how far decides whether one artifact serves both use cases:

| | index location | who runs BM25 | cost of hybrid |
|---|---|---|---|
| **Offline device** | on device | the device — needs a local inverted index | real, unmeasured |
| **CLI / thin client** | Qdrant cluster | the cluster — already built in | **zero** |

`constella-zero-hybrid` **ships alongside `constella-zero`, never replacing it.** Two 31 MB files
against one index; the deployment picks. If the dense-only regression is small, say so and let
users pick on cost; if it is large, that is the finding.

**Also measure and publish the offline BM25 cost** — inverted-index size and query latency on the
existing Edge prototype corpus. It is a hole in the M7 headline today (`results/costs.json` is
query-side only; `results/edge_prototype.json` has no sparse collection) and M12 is where it closes.

## Comparators and confirmatory surface — the two ways this becomes worthless

1. **Hybrid must be compared to hybrid.** Today no dense+BM25 comparator rows exist except
   `lightretriever-…-hybrid` 0.4720. Build `bge-small+BM25`, `mdbr-leaf-ir+BM25`,
   `LR-dense-pertask+BM25` under the SAME Gate-1 operator. Giving only our own system BM25 is not a
   result. These are a six-set read: register the file before generating it, and **never overwrite
   `results/perquery.json`** (sha-pinned in `m9/FINAL_LOCK.md`).
2. **The six cannot judge this model.** Their fused-vs-dense per-dataset rows were observed
   2026-08-28 (`results/m7_final_run.json`), so a fusion-trained table is designed with knowledge of
   how they respond to fusion. **Confirmatory surface is the reserved four or LoTTE, chosen and
   registered at Gate 1** — one access, per `instructions-m7.md`. Six-set rows are descriptive only
   and must be labelled as such.

## Out of scope — do not expand this

- **nano.** Hybrid nano waits until hybrid zero pays. M10/M13 are untouched.
- **The document tower.** Frozen. Co-adaptation is M15.
- **A better teacher.** Closed 2026-09-04: stella first of eleven, and first on exposure-free
  SQuAD/ESCI (`research/teacher-reviews-2026-09-04.md`, `m7/RESULTS.md`).
- **A fusion-parameter sweep** beyond the single registered Gate-1 choice.
- **Anything requiring cloud compute.**

## Deliverables

1. Gate 0 residual + the mechanism read, in `m12/FINDINGS.md`. (May be the whole milestone.)
2. The registered operator choice and loss, in `m12/LEDGER.md`, before any training.
3. `constella-zero-hybrid`, frozen and verified releasable, fused and dense-only on the registered
   surface, alongside the unchanged `constella-zero`.
4. Hybrid comparator rows under one operator.
5. The offline BM25 cost row.
6. Decisions logged in `CLAUDE.md`; transferable lessons in `m12/FINDINGS.md`; dead ends in
   `m12/EXPLORED.md`.
