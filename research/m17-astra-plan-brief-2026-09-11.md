# Brief: adversarial whole-plan review of M17 (Zero v1.1), revision A2

You are reviewing a research plan, read-only. Do not edit, create or run anything. Do not
search recursively; open only the files named here.

## Files to read, in this order

1. `CLAUDE.md` (project rules)
2. `instructions-m17.md`
3. `m17/STATUS.md`, `m17/PLANNING.md`, `m17/registry.json`, `m17/LEDGER.md`, `m17/REVIEW.md`,
   `m17/CODEMAP.md`, `m17/FINDINGS.md`
4. `results/m17_planning_probe.json`, `results/m17_tokenizer_followup.json`, `results/m8_b2_entropy.json`
5. `results/m7_run_p35w-2m-s2500.json` (the base recipe's config: lr, batch, steps, bank, KL)
6. `m7src/train.py` lines 600 to 700 only (the existing distill loss), `m7src/program.py` lines 90 to 110 and 230 to 245
7. `research/m7-data-licensing.md`
8. `research/m17-plan-review-loss-2026-09-11.md`, `research/m17-plan-review-data-2026-09-11.md`

**Forbidden reads:** anything under `results/frozen_eval/untouched-*`, any reserved qrels cache,
anything under `work/m9reserve`, and any six-set or LoTTE evaluation payload. Reserved sets are
FEVER, DBpedia-entity, cqadup-android and cqadup-english. Do not open them. List every file you
read at the end of your report.

## What the plan is

Zero is a 30,522 x 1,024 token lookup table serving as a query encoder against a frozen Stella
document index. M17 plans, for a future 72 hours on one RTX 3080, to add up to 3,072 new
vocabulary rows (jointly trained, sum-initialized), compare a hard-candidate listwise KL loss
against the old cosine-only continuation, and test alias consistency, late-checkpoint averaging
and an int8 resident-row loader. Five short screen arms (C, V, L, VL, VL-A) then two full runs
with two seeds. Selection is routed on the reused pinned development suite; a small judged panel
(600 queries, six domains, half sealed) is descriptive and audit only. Nothing has been run.

Revision A2 (today) changed: final dose 16,000 to 6,000 steps at batch 256; query cap 300k to
600k; bank 131k to 262k documents; averaging snapshots 4,500/5,250/6,000; panel built pre-clock;
Kubernetes docs admitted in principle as a 10 percent technical slice; untrained export V0;
per-domain cap of 1,024 new rows; `k8s` pinned by CTO request; a held-out alias test set; a
pre-clock end-to-end rehearsal.

## What I believe, and want you to break

- The arithmetic in the registry is right (34,433,850 params under the 35M cap; 72 hours; mixes sum to 64).
- The listwise arm is a genuinely new test: M7's KL used 31 uniform distractors and B2 showed
  it was one-hot; teacher top-31 from a 262k bank will carry information at T=0.05.
- 6,000 steps x 256 over up to 600k queries is enough signal without memorization for a table.
- The anchor at 0.001 is inert given row element RMS 0.39, and that is acceptable for a control.
- Routing selection on the reused dev suite is the least bad option given a 600-query panel.
- The plan does not leak protected evaluation surfaces or M13's frozen comparator.
- Sum initialization plus sqrt-count pooling has a known drift mechanism (P0b) that is
  documented but not a blocker.

## Questions to answer, ranked by how much they matter

1. Is there any way the screen or final selection can be decided by noise, by a leak, or by an
   unregistered degree of freedom? Name the exact rule and the exact fix.
2. Is any registered constant internally inconsistent with another (steps vs snapshots, batch
   fractions vs alias pairs, bank cap vs positive policy, query cap vs data actually admitted)?
3. Does the loss definition have a bug: KL direction, temperature use, normalization, the
   alias term interacting with the cosine target, the anchor on folded vs unfolded rows?
4. Does anything in the plan violate `CLAUDE.md`: licensing, the 35M cap, frozen index,
   protected access, "never overwrite perquery", torch.compile, M13 non-interference?
5. Is the 72-hour allocation plausible for one RTX 3080 given teacher encoding of up to 600k
   queries plus alias views, mining over 262k documents, five screen arms and four full runs?
   State which phase you think is underpriced and by roughly how much.
6. Is the Kubernetes-docs admission handled correctly under the licensing table's own standard
   (CC BY 4.0 attribution; no-LLM-training clickwraps; contamination against any eval set)?
7. What is missing that would make a competent reviewer refuse to authorize execution later?

Be adversarial. A review that only confirms is useless. Give findings as P1/P2/P3 with file and
line references, one paragraph each, then a short list of what you would change in the registry.
