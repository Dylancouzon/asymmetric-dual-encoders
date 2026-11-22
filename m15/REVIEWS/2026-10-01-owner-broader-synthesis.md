# Broader Constella framing: owner challenge and disposition

2026-10-01. The owner asks whether the paper is good, whether Constella's interchangeable query
paths and near-zero encoding belong in it, and whether the previous rewrite overfocused on one
aspect of the project. This is a reassessment of our editorial judgment, not a new experiment.

## Judgment

V11 is more scientifically defensible than v10, but narrows the paper too far. It never names
Constella and leads with a closed-form teacher screen that does not test the released Zero or
Nano recipes. The screen is an interesting independent finding; it should not substitute for
explaining the project's actual research object. The $95 optimization loop is useful accounting,
not the organizing discovery. Passing focused correctness reviews establishes bounded claims,
not reader interest or likely community impact.

The best supported paper is an applied empirical study of **making query computation selectable
while one document index stays fixed**. Zero, Nano, and the original Stella query path establish
the available choices. The main research result explains their consequences: an almost negligible
encoding tier still pays for graph search, and changing the query encoder changes the search effort
needed over an unchanged document graph. Complementary query outcomes make selective compute
interesting, while the measured simple routers fall well short of the oracle. Teacher-table
learnability is a second finding about designing the family before indexing. Build accounting
shows what construction requires without promising complete reproduction for $95.

This can be a good paper for search engineers without being a new architecture or a state-of-the-art
claim. Its strength is useful measured decisions, public compatible artifacts, and honestly bounded
findings. The current evidence does not establish a general mechanism of teacher learnability, a
production routing policy, or a breakthrough retrieval method. No editorial revision can guarantee
that the paper will be a hit.

## Correcting the emphasis

- Name Constella and state the one-index design question in the title, abstract, and introduction.
- Explain compatible per-request selection and the original-teacher fallback before discussing costs.
- Put the served-tier encoding/quality frontier and ANN study ahead of teacher screening.
- Use existing rows from one binary-quantized collection for a literal one-index deployment example.
  Independently fastest results over multiple quantizations remain a broader configuration sweep.
- Briefly expose routing headroom and the failed simple-policy result in the body; keep full routing,
  blend, prefix, and shuffle diagnostics in appendices.
- Keep teacher selection and component training costs as supporting design/build results.
- Point readers to the existing inference quickstart and public artifact identities. Using the
  family is distinct from training new aligned students; a standalone training kit remains absent.

## Claim boundaries

“Hot-swappable” means selecting an available, aligned query encoder for the same populated index.
E2 demonstrates three encoder batches against each unchanged persistent collection in one server
lifetime. It does not time model loading, concurrent request interleaving, or production failover.
No arbitrary same-width encoder is compatible. Zero is a no-transformer query path, with measured
44-microsecond warmed encoding on the stated CPU; tokenization and retrieval are not zero latency.
Fusion adds a lexical field and a measured search cost rather than constituting another dense tier.

Prior art already establishes query-side distillation, static queries, and static/contextual
switching. Attribution limits architectural priority; it does not make the practical capability
irrelevant to the paper's motivation or explanation.

## Independent checks

The same prior reviewers were explicitly asked to challenge their narrowing recommendation.
`2026-10-01-owner-broader-reader.md` reverses that emphasis; the correctness review in
`2026-10-01-owner-broader-correctness.md` verifies the stronger positive capability against E2's
source and receipts. The revisions use existing published aggregate evidence. No training,
protected evaluation, cloud rental, public release, or new quality experiment is needed.
