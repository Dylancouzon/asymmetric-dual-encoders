# M7 experiment results (append-only)

Every run lands here, including stopped, failed and OOM ones — that is the experiment ledger the
mandate requires. Written by `m7src/sweep.py`.

**Read the dev metric carefully.** The column is the *fast proxy macro* over three components
(nq-250k, cqadup-programmers, cqadup-physics) that in-training evaluation uses, so a sweep of ~25
configs stays affordable. Every **selection and gate** decision instead uses the full pinned dev
suite of six components (adding hotpotqa, heldout-train, heldout-longq) via `gate.py`. The two
are not comparable numbers: the teacher scores 0.5722 on the proxy and 0.6672 on the full suite.

Reference rows, full pinned suite (`work/devres/refs.json`): teacher (bge-base symmetric,
prefixed) **0.6672** · BM25 **0.4525** and potion-retrieval-32M **0.3801** on the four
text-backed components, which is where the G1 and G3 comparisons run — BM25 and potion have no
row on the held-out slices, whose corpora are pool row indices carrying no document text.

| run id | config | dev metric (proxy macro-3) | verdict |
|---|---|---|---|
| p1-objB | `work/runs/p1-objB.json` | 0.4548 | ok |
| p1-objB | `work/runs/p1-objB.json` | 0.4548 | ok — objective B (distillation, 8k steps). Matches the closed-form flat bound (0.4542) to +0.0006, so learned per-token weights + the KL ranking term buy nothing over flat MSE distillation. Confirms no optimisation pathology. Coverage 27,314/30,522 rows (89.5%), median 262 updates/row. |
| p1-objA | `work/runs/p1-objA.json` | ~0.353 (plateaued by step 4000) | ok but weak — pure contrastive from the teacher init plateaus at roughly potion's level (0.3525 proxy), far below objective B's 0.4548. Suspect `reg_init=1e-3`: it pulls rows toward the teacher init, which alone scores only ~0.20 (measured at lambda=10 in the ridge sweep), and the pull is strongest early when update counts are low. B overpowers it with a dense cosine target; A's contrastive gradient is sparse per row and may not. Tested by `p4-reg/reg0`, and by whether objective C's A-phase degrades B's checkpoint. |
