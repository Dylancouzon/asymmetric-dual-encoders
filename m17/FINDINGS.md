# M17 findings — planning evidence only

## The vocabulary problem is partly representation

The local Nano tokenizer's vocabulary map equals Zero v1's. The pinned Stella specification
uses the same BERT WordPiece vocabulary. A transformer can contextualize pieces that Zero
simply pools. Splitting an acronym does not mean the input is unknown, and adding a whole-term
row does not by itself teach its meaning. M17 must measure retrieval behavior, not just token count.

P0 confirms that S3, k8s, Kubernetes and kubectl split into several pieces and that ordinary
added-token matching can give them dedicated rows. P0b finds similar fragmentation in science,
medicine, finance, legal and everyday examples. These are hand-authored illustrations, not a
measured distribution of coverage gaps. Full tokens and mapping evidence:
`results/m17_planning_probe.json`, `results/m17_tokenizer_followup.json`.

## Compositional initialization has a specific limit

P0's single-term/repetition examples agreed after sum initialization, but did not exercise the
shared-subword mechanism. A standalone digit and its continuation-token counterpart have different
IDs, so the original apparent sharing fixture was not a counterexample. P0b corrects the fixture:
Kubernetes and kubectl share a base piece; k8s and EKS share another continuation piece.

Under sqrt pooling, the original combined count contributes `sqrt(a+b)` times the shared row;
separately replaced terms can contribute `sqrt(a)+sqrt(b)` times that row. Those differ. P0b
measures the resulting vector drift on the released rows. Sum initialization remains a useful
warm start, but arbitrary tokenizer expansion does not inherit M8's compiled-bag parity proof.
This neither improves nor worsens retrieval by itself; no quality was scored.

## Resource feasibility is encouraging but incomplete

The local GPU is available. Disposable cached-target table updates at both tested batch sizes
complete with finite losses and no allocator retries, using a small fraction of VRAM. The
current release's CPU path is also fast on the small fixtures. Exact rates, timings, array
sizes and limitations live in `results/m17_planning_probe.json`.

Synthetic resident tensors omit I/O, teacher encoding, document mining, sqrt-count preparation,
regularization, checkpointing and evaluation. Their throughput must not be multiplied by the
training window to advertise a feasible data dose. The future real-path smoke must time these
stages separately at two sizes. No inference speedup for an expanded trained model was measured.

## Research narrows the experiment

M8's frozen-row vocabulary additions failed; improved fragmentation did not repair retrieval.
Jointly trained extensions remain a different unrun question. The local entropy result, rather
than shorthand prose in older findings, distinguishes teacher-distribution entropy from KL:
the old bank produces nearly one-hot teacher targets, while teacher-neighbor lists carry more
distributional information. This supports testing listwise supervision, not expecting a gain.
See `results/m8_b2_entropy.json` and the three `research/m17-*-2026-09-11.md` notes.

True ColBERT-style late interaction needs document token representations. Summing weighted token
dots against one existing document vector just moves the pooling operation after the dot product.
This is why M17 prioritizes richer training targets while preserving the shipped scorer. No new
empirical closure of late interaction, tokenizer families or model capacity is claimed.

## The init anchor is inert at the optimizer level (step-5 smoke, 2026-09-11)

The registry asked for this to be measured rather than asserted (`training.init_anchor_note`,
`training.optimizer.note`). In the pre-clock smoke on the real 10,000-query prepared directory,
the anchor term's gradient norm on the rows is exactly 0 at step 1 (the effective rows ARE their
initialization) and still rounds to 0 at 8 decimals after 400 VL-A steps and 100 C steps, against
a total per-step row-gradient norm of 2.3e-2 to 3.2e-2 (2.99e-2 to 2.66e-2 in the re-run below). At the registered weight 1e-3 the anchor
contributes no measurable share of the row gradient at these step counts; it is not drift
protection, exactly as the registry's note says. The anchor's share of the row UPDATE norm is not
observable after Adam's per-parameter scaling, so the run record carries the gradient share and
says so. The first reading of the other three terms used a wrong denominator (the sum of the
per-term norms, Sol step-5 P3-9); re-measured against the norm of the TOTAL row gradient on the
rebuilt 10,000-query directory (100 VL-A steps, batch 256), the listwise term carries 0.887 of it
at step 1 and 0.883 at step 100, the teacher cosine 0.251 to 0.268, and the alias consistency
term 0.0032 to 0.0024, against a total row-gradient norm of 2.99e-2 falling to 2.66e-2. The
shares sum slightly above one because the cosine and listwise gradients partly oppose each other,
which the old denominator hid. The alias term is a small nudge on top of the two fit terms, not a
competitor to them. These are rehearsal numbers on a subsampled pool, not a registered
observation.

## A long GPU stage must be durable in chunks (2026-09-11, step 6)

The full pool's teacher stage encodes ~550k texts in ~23 min. Attempt 1 died at 120k with a CUDA
illegal-memory-access and lost every vector, because the cache wrote nothing until the whole
missing set was encoded. The fix is small — encode misses in 25k chunks, flush each with a tmp +
replace and drop an unindexed vector suffix on reopen — and it changed the failure cost from "the
stage" to "one chunk". The same fault did not recur at the same text order on the retry, so it is
recorded as transient (WSL2 GPU), not diagnosed; the measurement that would change that is a
second fault at the same position under the same chunking. Lesson generalized: before any run
longer than a few minutes, ask where the first durable write is.

## Support, not slots, bounds the vocabulary extension (2026-09-11, step 6)

Over the full 600,000-query pool, discovery produced 210,447 candidate terms and selected 445:
199,763 fell below the registered support minima (20 distinct source documents, 50 training
contexts), 10,239 abbreviations were dropped, and neither the 3,072 total nor the 1,024
per-domain row caps was reached. A tenfold vocabulary would need ~275k rows (~280M parameters,
8× the 35M cap) AND a pool large enough to support them; the cap is the premise of the cost
comparison and the pool is bounded by commercially licensed sources. Breadth is `narrow`
(441 general, 3 cloud-software, 1 legal) — the Kubernetes slice contributed `k8s`,
`kube-apiserver`, `kubernetes`. Pre-screen numbers; the on-clock screen re-derives the list.

## A rehearsal that stubs the only on-clock import proves nothing about it (2026-09-12, step 6b)

A pre-clock rehearsal that stubs the only on-clock import does not exercise the on-clock import
order; the first screened build died at `parity` on it. Smoke the real import chain at least once
before binding a source hash — the fix was one line, but it cost a dated amendment to an invariant
build input already bound by the pre half.

## 2026-09-12 — screen `no_survivor`: the registered objective moved v1 away from itself

All five 4,000-step screen arms read 0.021–0.035 nDCG@10 macro below the untrained V0 export on the
same loader, manifest and quantization path, while train and held-out losses fell monotonically and
the schedule completed. Lessons: (1) a warm start from a released table needs a registered
eligibility reference with numbers, not the word "v1"; (2) an anchor at weight 1e-3 contributes
1e-6 to the loss and constrains nothing — if the anchor is meant to bind, its weight must be set from
the measured drift, not assumed; (3) monitoring losses on training sources cannot detect a retrieval
regression on the dev suite, so a warm-start recipe should budget an early read of an intermediate
checkpoint before spending a full screen; (4) the protected screen can shrink a fixed-size held-out
slice — pins should reference the lock-bound manifest, not a literal. Details: `LEDGER.md`
2026-09-12 entries, `screen_decision_2026-09-12.json`.
