# M17 Zero v1.1: data and bounded training notes (2026-09-11)

Planning only. No corpus was downloaded and no training or evaluation was run. The hardware
confirmed local RTX 3080 with 10 GB VRAM.

## Reusable recipe and caches

The release is a 30,522 x 1,024 WordPiece table (62.5 MB fp16; 31.3 MB int8), served by
tokenize/gather/pool/normalize with no query-time transformer. Its document side is the frozen
`NovaSearch/stella_en_400M_v5` revision `ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`
([`m7/RECIPE.md`](../m7/RECIPE.md), lines 14–24). Existing cached teacher query/document
vectors are therefore the valuable reusable input: cache identity and dtype must be checked,
and an fp16 cache must round-trip through the cold path as M10 records ([`m10/CODEMAP.md`](../m10/CODEMAP.md),
lines 42–50). Re-encoding the teacher is the likely wall-clock bottleneck; table updates are
small relative to loading/tokenizing the pools and cached vectors.

The historical shipping schedule is Phase B 16,000 steps followed by Phase A 2,500 steps,
batch 512, seed 0. B uses teacher-context row initialization (30,522 teacher forwards),
2,000,000 requested pseudo-queries (which deterministically saturate at 924,704 spans), IDF
weights, cosine + top-32 ranking KL, 32,768 bank negatives and `fn_margin=0.02`. A resumes
the B checkpoint with rows and weights and uses 2,500 steps with 200-step warmup
([`m7/RECIPE.md`](../m7/RECIPE.md), lines 46–91). It is a reproducibility baseline, not a
mandatory primary rerun: v1.1 can warm start from the existing checkpoint and run a bounded
fine-tune. Avoid changing teacher, index, cap, or release policy.

The train implementation confirms that query teacher vectors come from `encode_cached`, while
the table forward is tokenized locally; objective-B extra query text and pseudo-query targets
are teacher encoded once and then retained for training ([`m7src/train.py`](../m7src/train.py),
lines 433–500). The 2M pseudo pool is deliberately held as flat int32 ragged IDs because a
Python list representation is about 10.4 GB host RAM; this is the dominant local memory trap
alongside the vector pools ([`m7src/table.py`](../m7src/table.py), lines 74–125). Do not infer
that a cached checkpoint is directly shippable: `save_release` folds learned scalar weights
into rows, whereas training checkpoints remain unfolded and a folded table cannot resume
([`m7/RECIPE.md`](../m7/RECIPE.md), lines 89–91; [`m7src/table.py`](../m7src/table.py), lines
325–366).

## What v1.1 may test

The focused S3/Kubernetes/DevOps vocabulary can draw on existing admitted Wikipedia-derived and
already approved dataset text; those sources remain in scope. Only an actually new corpus or
publisher requires a fresh licensing ruling. The licensing ledger permits commercial derived weights for its
approved stack, requires attribution for CC BY-SA, and keeps MS MARCO validation-only;
non-commercial validation material cannot provide gradients, targets, negatives, or synthetic
seeds ([`research/m7-data-licensing.md`](m7-data-licensing.md)). No new source is approved by
this note.

Kubernetes’ official website repository is a plausible optional new corpus candidate: its direct
`LICENSE` artifact identifies the documentation as Creative Commons Attribution 4.0, while code
samples may retain Apache terms (<https://github.com/kubernetes/website/blob/main/LICENSE>).
The official docs expose rich,
directly relevant concepts—Pods, Deployments, Services, RBAC, storage, scheduling, observability
and troubleshooting—(<https://kubernetes.io/docs/contribute/>). This supports proposing a
small Kubernetes documentation slice for owner approval with URL/version manifest, attribution,
and code-sample/content separation. CC BY 4.0’s attribution and adaptation terms are at
<https://creativecommons.org/licenses/by/4.0/legalcode>.

AWS documentation remains out for this plan: a public page or scraper availability is not an
affirmative commercial training grant. Do not conflate AWS documentation content with licenses
on AWS SDK/repository code. Ansible’s documentation repository is marked GPL-3.0, so it is not a
clean candidate for this Apache-derived-weights corpus without a separate ruling
(<https://github.com/ansible/ansible-documentation>). Terraform/HashiCorp pages inspected here
did not provide a clear corpus-level documentation license; leave them as unapproved pending
primary terms evidence. The practical candidate list is therefore “Kubernetes CC BY 4.0,
pending owner approval” plus existing approved sources.

## Objective and quality measurement

Do not revive mined hard negatives: teacher/BM25/mixed mining was closed after gains tracked
seen-document memorization, while provided ESCI/Mr TyDi negatives and the existing false-negative
margin remain distinct and usable ([`m7/RECIPE.md`](../m7/RECIPE.md), lines 63–73, 152–164).
Hard-candidate listwise distillation (`R-LIST`) is the open alternative: the shipped uniform-bank
instance was inert, but the `teacher_top200` variant had measurable mean entropy of 0.776925 nats
(rounded 0.777); the uniform bank averaged only 0.018168 nats. These are entropy diagnostics,
not KL measurements ([`results/m8_b2_entropy.json`](../results/m8_b2_entropy.json)); the class has
not been measured and needs its own protocol ([`m8/FINDINGS.md`](../m8/FINDINGS.md), lines 53–61).
If v1.1 uses listwise candidates, define candidate source, K, exclusions, and a fixed quality bar
before observing results; do not silently substitute the failed bank objective.

Quality should be measured on a fixed dev slice only for smoke/diagnostics, with out-of-domain
macro reported separately and contamination/reuse disclosed. The prior full-dev release macro
0.6153 and out-of-domain 0.3672 are selection numbers, not evidence on the six
([`m7/RECIPE.md`](../m7/RECIPE.md), lines 8–10). A changed checkpoint invalidates any fusion
parameter and requires re-selection. M12’s deployed fusion is fixed DBSF@100 with no fitted
weight; v1.1 should remeasure fixed DBSF against the dense baseline and, only if registered,
re-select a convex weight. Preserve the frozen Stella 1,024d index and compare dense and any
fused result against the same int8 artifact. The released v1 artifact is MIT; historical prose in
the licensing sweep uses older Apache wording and should not be copied into v1.1 claims.

## Bounded execution recommendation and risks

1. Verify CUDA, free VRAM/RAM, cache manifests, teacher revision, tokenizer/preprocessing
   fingerprint, and decontamination masks. A CUDA-enabled torch install alone does not prove a
   usable GPU.
2. Run a short real-path smoke with resume and realistic shapes, recording throughput and peak
   RSS/VRAM. Keep `batch_tokens` at or below the observed ~32,768 failure boundary on this card;
   M10 recorded `CUDA driver error: device not ready` above it ([`m10/CODEMAP.md`](../m10/CODEMAP.md),
   lines 57–67).
3. If the smoke is healthy, run the pinned B then A schedule, periodically retaining checkpoints;
   export only after folding and serialized-artifact parity checks. If cache encoding is needed,
   budget it first because teacher encoding/re-encoding is the expected bottleneck.
4. Report per-form exposure, cache hits/misses, peak host memory, GPU memory, wall time, loss
   components, and dev quality. A resumed run must derive stream position from step; M10 found
   call-count loaders silently replayed data ([`m10/CODEMAP.md`](../m10/CODEMAP.md), lines 49–52).

Exact local files read: `CLAUDE.md`, `instructions-m17.md`, `m7/RECIPE.md`, `m7src/train.py`,
`m7src/table.py`, `m8/FINDINGS.md`, `m8/EXPLORED.md`, `m10/FINDINGS.md`, `m10/CODEMAP.md`, and
`research/m7-data-licensing.md`, and the permitted diagnostic `results/m8_b2_entropy.json`.
`m10/registry.json` was requested but is absent at that path;
it was not read. Web sources read are the Kubernetes contribution page, Kubernetes website
license page, the CC BY 4.0 legal code, and the Ansible documentation repository listed
above; no content corpus was fetched.
