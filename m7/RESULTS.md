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
