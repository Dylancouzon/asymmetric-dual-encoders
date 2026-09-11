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
