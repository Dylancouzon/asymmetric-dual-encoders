# M12 — closed avenues

## 1. Fusion-aware training of the table (`constella-zero-hybrid`) — CUT 2026-09-04, before any code

Proposed by Dylan; two drafts written and both broken by review the same day. Moved to
`instructions-m16.md`. Recorded here so it is not re-proposed without the answers.

**Draft 1, broken by Fable.** Framed the objective as "stop imitating stella's query vector". Wrong:
`zero`'s final phase is objective A, **InfoNCE** against frozen doc vectors
(`work/runs/p35w-2m-s2500.meta.json`). Under fusion BM25 is a **fixed additive logit bias** and the
gradient reaches `W` only through the dense score — so the design variable is the **candidate set**,
not the loss. M8 measured the objective **inert** on the current candidates: positive ranked first
for **99.75%** of training queries, uniform-bank KL **4.73e-07 nats**, against `teacher_top200`
**0.777** (`m8/FINDINGS.md` §3.1). Draft 1 also had no control arm (warm-start step count alone moves
dev 0.0027–0.0078), a kill gate that could not fire (teacher dense 0.6350 already beats fused zero
0.5727 by 0.062, against a 0.005 threshold), and named **RRF as a training target** — piecewise
constant, zero gradient a.e.

**Draft 2, broken by Codex.** Rebuilt around union-mined candidates and two "failable" probes. All
three of its new mechanisms rest on things that do not exist:

| assumed | actual |
|---|---|
| `mine_bm25_negatives` mines the 6.17M pool and returns scores | mines **within each query's own store** (`train.py:217-220`) and returns **IDs only** (`:258`) |
| `block_cg_ridge` takes per-query weights | no weights argument (`m8src/blockcg.py:69`) |
| dense runs are cached in `work/fusionruns` | only the four **BM25** arrays are; dense is re-retrieved on CUDA (`select_fusion.py:50,73`) |
| the 0.004 fused floor calibrates per-arm re-selection | that floor is **frozen-`w`**; re-selecting "would measure the floor of a fitting procedure, a different and much larger quantity" (`m8_noise_floor_fused.json`) |

And the power arithmetic kills it independently: fused seed SD **0.000332**, so a 3-vs-3 mean
difference has SE ≈ **0.000271**; at a true effect of 0.005 — the ceiling of every table-side lever
measured here — **P(observed ≥ 0.008) ≈ 0**. The design guaranteed its own null.

**What would have to be true to reopen it** (all of them, see `instructions-m16.md`): a trainer that
carries per-query candidate lists with aligned lexical scores; both retrievers searching the same
corpus; a registered operator-specific loss with the detach decisions named; a measured noise floor
for per-arm fusion re-selection; and a bar set from that floor rather than from the historical
recipe band.

**What survived:** the observation that Qdrant ships RRF/DBSF and not our hand-fitted convex0, which
is now the whole of M12.
