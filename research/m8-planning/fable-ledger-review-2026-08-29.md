# Fable scientific-judgment review of m8/LEDGER.md v1 (2026-08-29)

Brief: not a protocol audit (Codex ran that in parallel) but the question that matters most —
**will this plan, executed exactly as written, produce the best zero-compute query encoder
obtainable under the constraints, and if not what is it leaving on the table?**

Verdict: the protocol machinery is the best this project has produced and nothing in it will
produce a false win — but executed as written the most likely outcome is a well-defended null.
Full text of the findings is folded into `m8/LEDGER.md` v2 §16; the items, in the reviewer's own
ranking, were:

1. **E14 — doc-side co-adaptation** (LoRA/fine-tune the document tower jointly with the table).
   The structural fact the plan never states: LightRetriever's table works because its document
   encoder was co-trained to be reachable by a bag of token vectors, while M7 fit a bag to a
   frozen doc space never trained to be bag-compatible. That is the literal content of "the gap is
   architectural". Every ruling survives (query side stays pure lookup, E1 intact; it is
   training-time not index-time, E5 intact; one served doc file, E3 satisfiable; vendor CLEAN).
   Costs: breaks doc-vector sharing with frozen M7, doubling the reserved-4 pre-encode, and it
   revisits the "frozen off-the-shelf document tower" premise — which CLAUDE.md explicitly lists
   as revisitable with arithmetic and Dylan's sign-off. **Owner question, not a session decision.**
2. **Shadow GO is mis-specified** — LoTTE reads only the CQA half of the estimand while the
   flagship data lever (Wikipedia ICT) lands on the other half.  → two-legged GO, §2.3.
3. **B17's routing rule** can formally endorse concentrating budget on the class whose transfer
   M7 measured at 0.000 ± 0.005, on in-domain evidence. → OOD corroboration condition, §9.
4. **Promote B7 and B6's ONNX precondition to Wave 1** — the capacity gates sat behind five
   recipe probes. → §9.
5. **Register a retention-decomposition diagnostic** (within-dataset length strata + subword
   fragmentation), free, from existing per-query dumps. → run; `results/m8_retention_decomposition.json`,
   LEDGER §17. It overturned the plan's H3 framing.
6. **Register an ordered nested fallback** R1-full → R1-data-only → R0 before any validation
   number, so one bad adopted setting does not revert every surviving win. → §7.
7. **Price the E12 LR-dense pre-encode** (a 1.5B Qwen over 10.12M documents) in the schedule with
   a pre-agreed published-numbers fallback. → §14 G6 + wake-up note.
8. **D2 compositional init floor**: cold multi-word rows initialize to the mean of their
   constituent unigram rows, so coverage failure degrades to M7 behaviour, not to noise. → §8.
9. **Cite the clean-stack-tax number against H1** so B3's prior is honest. → §3.
   Plus: B11's removal reason was wrong (C1 is fused-vs-fused, so fusion is live for the PRIMARY
   leg; the correct reason is the frozen fusion operator's no-routing rule). → §7.
   Plus: doc-side small-k multi-vector recorded as considered-not-adopted. → §8.

Sound and needing no change per the reviewer: H2 and the listwise arm; the statistics family and
its calibration caveat; the one-shot mechanics port; the int8-always rule and the kill-list; the
teacher swap bar with the CI-widening penalty; the six-set no-regression guard as the
anti-memorization blocker.
