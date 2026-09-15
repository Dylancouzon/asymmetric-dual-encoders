# M13 findings — final

Authoritative values live in the result JSONs; `m13/STATUS.md` is the concise numerical summary.

## Scientific findings

- Nano exceeded BGE-small on the registered clean-four and all-six partitions, and exceeded LEAF
  asym on all six. It did not establish superiority to LEAF on clean four.
- The limitation is concentrated rather than hidden: Nano trails LEAF on TREC-COVID by 0.042982,
  while its all-six aggregate remains positive.
- ArguAna and FiQA are the two overlap-affected datasets disclosed by Stella. They remain in the
  all-six sensitivity view, while the clean-four result is the headline generalization evidence.
- M9 was substantially weaker than Nano on every dataset. The paired descriptive difference is
  +0.154009 on clean four and +0.160004 on all six, with both 95% intervals wholly positive. This
  comparison is descriptive because the systems differ in more than one controlled factor.
- The 279-example dose shortfall is 0.0001395% of the plan and is exactly accounted for. It does
  not look like catastrophic undertraining, but it must remain disclosed rather than rewritten as
  exact 200M compliance.
- The frozen ONNX route matches the reference model closely and has BGE-small-class CPU latency,
  while its 1024-dimensional vectors cost more index storage than BGE-small's 384 dimensions.

The evidence supports advancing to release preparation under the owner's “release unless gross
overfitting or catastrophic failure” policy. It does not support an unrestricted superiority
claim, and publication is still blocked on M14's reserved-four and BEIR-18 descriptive work.

## Execution findings

- Protected access needs a durable spend receipt before the first protected byte, and recovery
  must consume only authenticated atomic rows. Both six-set runs followed that rule.
- Corpus-only pre-encoding and query/qrel scoring are different contact classes. The M14 package
  keeps them in separate guarded processes and permits only exact reserved corpus-cache metadata
  needed by the dataset loader.
- Resume safety requires validating a completed file, not merely noticing it exists. The reserved
  package binds each system output to transaction, payload, encoder, and document-cache hashes.
- Long benchmark stages should be allowed to finish without speculative implementation work. The
  final CPU TREC-COVID row took almost five hours after SCIDOCS and completed normally.
- Operational evidence that lived only in ignored work paths is now archived under `audit/m13/`;
  the original large scientific artifacts remain preserved in their verified local backups.

## Scope boundary

R19 moved A100 execution—not its trigger, datasets, systems, statistics, or interpretation—to
M14. M13 did not resume a GPU, create `m8-reserved-spent`, read reserved queries/qrels, invent a
reserved score, publish to Hugging Face, or open a FastEmbed PR. The full 18-dataset BEIR suite is
also M14 pre-publication descriptive validation and must not be retrofitted into M13's registered
gates.
