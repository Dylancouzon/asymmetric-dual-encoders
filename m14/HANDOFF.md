# M14 release entry — constella-nano

> **M13 is closed and unpublished.** Fetch `origin/main` plus tags and require
> `refs/tags/m13-closed^{commit}` to be an ancestor of `origin/main`. Under owner ruling R19, M14
> owns both the triggered reserved A100 execution and publication. Do not publish until the
> reserved four and the separate broad descriptive BEIR-18 run are complete.

This is the single entry document for the different agent that will execute M14. It is intended
to stand alone: use the paths, hashes and gates below rather than relying on chat history or on a
machine-specific symlink.

## Scope and release policy

- Intended new repository: `DylanCouzon/constella-nano`. Verify the authenticated Hub owner is
  exactly `DylanCouzon` immediately before creating it.
- Existing public repositories `DylanCouzon/constella-zero` and
  `DylanCouzon/stella-en-400M-v5-doc-onnx` already shipped in M11. Do not recreate, overwrite or
  republish them.
- M14 owns the final release bundle/model card, private-first Hub upload, downloaded-byte
  verification, public transition, and one clean FastEmbed PR based on upstream main. The old
  `add-constella-models` branch also contains the unrelated #703 padding fix and must not be merged
  wholesale.
- Owner policy: release unless the evidence shows gross overfitting or catastrophic failure.
  Missing a superiority target alone is not a release veto. Claims remain limited to what the
  registered tests established.

## Frozen candidate and durable retrieval

The immutable candidate is a bge-small-en-v1.5 backbone with a three-layer feature tap and linear
1024-dimensional head. It has 34,540,672 parameters, max length 512, no query prefix, and emits
normalized vectors intended for the already-published Stella document index.

| artifact | authoritative path | bytes | SHA-256 |
|---|---|---:|---|
| frozen torch checkpoint | `work/m13-final/cycle3.pt` | 413,567,295 | `3e49e0bfaa633abe276da7847d0ac44cbfb3d7d5215b9343f255706eeb5789a1` |
| independently backed-up checkpoint | `work/m13cloud-build-preflight-retry-backup/work/m13build/BUILD-200M/cycle3.pt` | 413,567,295 | `3e49e0bfaa633abe276da7847d0ac44cbfb3d7d5215b9343f255706eeb5789a1` |
| fp32 ONNX graph | `work/m13cloud-build-preflight-retry-backup/work/m13build/BUILD-200M/onnx/model.onnx` | 137,802,369 | `9ba0acf57b71dc31bc5512c5445078a797fa51cf3e85587d6b8a506bfc55dbc2` |

The ONNX directory also contains `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, and `vocab.txt`. Its complete 21-file backup manifest is
`work/m13cloud-build-preflight-retry-backup/manifest.json`, SHA-256
`cb5341e8e9bcff8f07c650345b484ea5d19e6b9d9987cc1282b5fb03337c2602`; M13 independently rehashed
all 21 local files with zero mismatches on 2026-09-15. The same checkpoint is retained on stopped
Runpod `k3aee2m68765em` at `work/m13build/BUILD-200M/cycle3.pt`; the reserved controller verifies
and materializes the hash-identical `work/m13-final/cycle3.pt` copy before use.

Restore locally from the verified backup without overwriting an existing target:

```bash
cd /home/dylan/asymetric-dual-encoders/work/m13cloud
test ! -e work/m13-final/cycle3.pt
install -D -m 0644 \
  work/m13cloud-build-preflight-retry-backup/work/m13build/BUILD-200M/cycle3.pt \
  work/m13-final/cycle3.pt
printf '%s  %s\n' \
  3e49e0bfaa633abe276da7847d0ac44cbfb3d7d5215b9343f255706eeb5789a1 \
  work/m13-final/cycle3.pt | sha256sum -c -
```

Do not delete either local copy or terminate the retained A100 volume before M14 has published and
verified a durable remote copy. Credential locations are under `/home/dylan/.config/runpod/` and
are documented in `m13/RESUME.md`; never copy credential values into Git, logs, cards or a pod.

## Export and serving evidence

The frozen build record is `results/m13_build_record.json` (SHA-256
`e2dbbf5b41e84f3a650d520a365d8a65612bf90fd202794b2c54066195fcd88d`). It records ONNX checker
success, opset 17, no custom-domain operators, 34,540,672 parameters, served max length 512,
Torch-to-ORT minimum cosine 0.9999999404, and stock-FastEmbed minimum cosine 0.9999999811. The
additional eight-input serving validation, including Unicode, whitespace, empty and overlength
text, is `results/m13_serving_validation.json`: minimum cosine 0.9999999557 and maximum absolute
error 1.1121e-7.

The build export serializes right-side `BatchLongest` padding in `tokenizer.json`. Preserve it as
historical evidence. The M11 release checklist requires the final staged FastEmbed bundle to set
`tokenizer.json` padding to `null`, letting FastEmbed install dynamic padding, while retaining
right truncation at 512 and both `model_max_length=512` and `max_length=512`. Perform this only in
the M14 staging copy and rerun reference, direct ORT and stock-FastEmbed parity—including an input
longer than 512 tokens—before upload.

Same-machine CPU measurements are in `results/m13_serving_costs.json` (three fresh processes per
model, batch 1, four threads). Medians across the three trials:

| model | hydration | first query | warm 20-word p50 | peak RSS | model assets |
|---|---:|---:|---:|---:|---:|
| constella-zero | 0.2618 s | 0.3529 ms | 0.1119 ms | 275.4 MiB | 90.1 MiB |
| bge-small | 0.6726 s | 8.2401 ms | 6.8400 ms | 291.0 MiB | 127.6 MiB |
| constella-nano | 0.6907 s | 7.6685 ms | 7.2511 ms | 280.9 MiB | 132.3 MiB |

These are synthetic query latencies, not a query-distribution estimate. Nano and zero both emit
1024-dimensional fp32 vectors (4.096 GB per million document vectors if used as an index);
bge-small emits 384 dimensions (1.536 GB per million). Reuse `m11/CODEMAP.md`, especially its
ONNX-port checklist and native-FastEmbed registration requirements.

Validated package versions on the M13 release-evidence host were: Torch `2.8.0+cu126`,
Transformers `4.57.6`, Sentence Transformers `5.7.0`, ONNX `1.22.0`, ONNX Runtime `1.29.0`,
FastEmbed `0.8.0`, Tokenizers `0.22.2`, NumPy `2.3.5`, and Datasets `5.0.1`.

This smoke runs directly against the frozen local ONNX bytes before native FastEmbed registration:

```python
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

BUNDLE = Path(
    "/home/dylan/asymetric-dual-encoders/work/m13cloud/"
    "work/m13cloud-build-preflight-retry-backup/work/m13build/BUILD-200M/onnx"
)
LOCAL_NAME = "m14-local/constella-nano"
TextEmbedding.add_custom_model(
    model=LOCAL_NAME,
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf=LOCAL_NAME),
    dim=1024,
    model_file="model.onnx",
    description="local constella-nano verification",
    license="mit",
    size_in_gb=(BUNDLE / "model.onnx").stat().st_size / 1e9,
)
query_model = TextEmbedding(LOCAL_NAME, specific_model_path=str(BUNDLE), threads=4)
q = np.asarray(next(iter(query_model.embed(["how do mRNA vaccines work?"]))))
assert q.shape == (1024,) and np.isfinite(q).all()

# The document half is already public and must be used without a query prompt.
doc_model = TextEmbedding("DylanCouzon/stella-en-400M-v5-doc-onnx")
docs = [
    "mRNA vaccines deliver messenger RNA encoding a viral antigen.",
    "The Treaty of Westphalia ended the Thirty Years' War in 1648.",
]
D = np.stack(list(doc_model.embed(docs)))
ranking = np.argsort(-(D @ q))
print([(docs[i], float(D[i] @ q)) for i in ranking])
```

`add_custom_model` is a local verification bridge only and cannot support FastEmbed
`parallel>1`. The released path must use a native FastEmbed model entry, after which the card uses
`TextEmbedding("DylanCouzon/constella-nano")`. For reference parity, instantiate
`m13src/score13.py:Nano10Student` from `m10/FREEZE.json`, tokenize with padding/truncation at 512,
and compare normalized rows by true cosine. The existing four- and eight-text results are evidence,
not a substitute for M14's frozen real length-stratified fixtures and negative controls.

## Training evidence and mandatory disclosure

The nominal plan was 200,000,000 examples. The immutable candidate actually saw 199,999,721,
a 279-example (0.0001395%) shortfall caused by nine one-row document tail batches losing 31 rows
each. `results/m13_dose_reconciliation.json` reproduces this exactly from metadata. Preserve both
the original FAILED supervisor receipt (`results/m13_cloud_build_preflight_retry.json`) and the
reconciled evaluation authorization. Never claim exact 200M compliance or relabel the failed
receipt.

Final selection-informed COV macro was 0.528708958 (94.9669% of the Stella teacher). The distinct
DEV-6 macro was 0.613048 (91.18% retention). COV is development/selection evidence, not proof of
generalization. Training sources and exact hashes/counts are in `results/m13_build_record.json`,
`results/m10_data_manifest.json`, and `results/m10_corpus_manifest.json`; decontamination removals
are in the same build record. The Stella teacher card discloses training/evaluation contact with
ArguAna and FiQA, two of the six final datasets, so every all-six result must identify those two
and the clean-four partition is the headline.

Licence/card inputs: the pinned bge-small source revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` declares MIT, and the pinned Stella document tower
revision `ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20` declares MIT. Attribute both and disclose that
Nano was trained to match Stella-space document/query targets. Consult `m10/EXPLORED.md`,
`m10/LEDGER.md`, `research/m7-data-licensing.md`, and the source rows in the build record before
finalizing the training-data section; validation-only non-commercial datasets must not be
presented as training sources.

## Final benchmark evidence

The registered six-set run is durable at `results/m10_final_run.json` and per-query rows are under
`results/m10_final_scores/`. Nano nDCG@10:

| dataset | nano | nano − bge-small | nano − LEAF asym |
|---|---:|---:|---:|
| SciFact | 0.721097 | +0.008391 | +0.022084 |
| NFCorpus | 0.363080 | +0.020129 | +0.002301 |
| FiQA† | 0.477765 | +0.074253 | +0.061296 |
| ArguAna† | 0.623296 | +0.019848 | +0.040043 |
| SCIDOCS | 0.217710 | +0.012498 | +0.014345 |
| TREC-COVID | 0.787116 | +0.029573 | -0.042982 |

† Stella discloses training/evaluation overlap. Registered conclusions: Nano beat bge-small on
clean four (delta +0.017648, lower one-sided 2.5% bound +0.003674, sign-flip p=0.006410) and all
six (+0.027449, lower +0.017271, p=0.000010); Nano beat LEAF asym on all six (+0.016181, lower
+0.006504, p=0.000540), but did not establish superiority on clean four (-0.001063, lower
-0.014456, p=0.559474). The TREC-COVID loss to LEAF is the principal per-dataset limitation.

The six-set result ends `INCOMPLETE_RESERVED` because those rejections triggered the registered
descriptive reserved four. R19 moves execution—not interpretation or scope—into M14. M13 terminal
evidence is now durable:

- `results/m9_final_run.json`, SHA-256
  `770344d7a7c18d0933e0f922af385fb0b4e514c98822e955e959cf139a9c936b`;
- `results/m13_paired_m9_nano.json`, SHA-256
  `4eaf20ebef5548338314fd547abd0f76c55eae8b5334e3be2f11260d2f8071b4`;
- the tested reserved scorer/controller and exact execution/cost pin in
  `m13/RESERVED_EXECUTION.md`;
- the qualified release recommendation and final benchmark table in `m13/STATUS.md`.

Do not infer or invent a reserved number. M14 must produce `results/m13_reserved_preencode.json`,
`results/m13_reserved_manifest.json`, `results/m13_reserved_run.json`, the controller receipt, and
the updated `results/m10_final_run.json` ending `COMPLETE` before publication. The rows are
descriptive, zero alpha; FEVER is double-contaminated sensitivity, not confirmatory evidence.

The full 18-dataset BEIR suite is deliberately not part of M13’s preregistered confirmatory gate.
Run it in M14 before public release as a clearly labelled broad descriptive validation, with exact
dataset/version/config pins and overlap caveats. Do not mix it into or reinterpret the M13 gates.

## Storage and cloud state

At the latest M13 read (2026-09-15), all three retained 500 GB pods were `EXITED`:

- `k3aee2m68765em` — final A100 build and reserved-batch target;
- `exulxoxelug5um` — replacement attempt, failed before producing unique scientific artifacts;
- `wnzk8eeqrrkw4m` — gate-chain/upload source, whose exact 394-file inventory was verified and
  migrated to the final A100 (`results/m13_storage_verification.json` and
  `results/m13_migration_ready.json`).

The historical instruction is STOP-only: do not terminate a pod or destroy a volume without new
owner authorization. Stopped storage is still billed (three 500 GB volumes). After the reserved
run, record its controller receipt and confirm the target returned to `EXITED`. M14 may recommend
retirement after Hub download verification, but must not infer deletion authority from this
handoff.

## M14 execution checklist

1. Require M13’s terminal closure commit and verify every result/artifact hash in this document.
2. Execute the pushed reserved controller on the retained A100. Require the exact registered four
   datasets and three systems, durable `m8-reserved-spent` receipt, all atomic score outputs,
   `m13_reserved_run.json`, updated `m10_final_run.json` ending `COMPLETE`, cost receipt, and
   confirmed pod STOP. Treat every reserved comparison as zero-alpha descriptive evidence.
3. Reconcile the full 18-dataset descriptive BEIR run before public release; keep it separate from
   M13’s registered inference.
4. Build a fresh release staging directory from the verified ONNX/tokenizer/config bytes. Apply
   only documented packaging transforms (including dynamic-padding metadata), recording hashes.
5. Create real, length-stratified parity fixtures with the 511/512/513 boundary. Run ONNX checker,
   opset/domain/dtype census, Torch-reference parity, direct ORT, stock FastEmbed, native
   FastEmbed registration, multiprocess behavior, and negative controls. Do not rely solely on
   the four build or eight serving-validation texts.
6. Write an MIT model card with source/teacher attribution, exact 199,999,721 dose, training data
   and decontamination summary, two-of-six overlap disclosure, all final and broad descriptive
   results, TREC-COVID/LEAF limitation, 512-token behavior, normalized 1024-d output, serving
   costs, and a runnable asymmetric usage example with the published Stella document tower.
7. Execute the model-card code offline against the staged bytes. Refuse any literal wrong repo id,
   accidental Hub access, missing sibling model, or stale FastEmbed import path.
8. Authenticate the Hub user, require owner `DylanCouzon`, create `DylanCouzon/constella-nano`
   private with `exist_ok=False`, upload, and verify a post-gate hash snapshot. Verify large files
   by LFS oid and download the published revision into a fresh directory before making it public.
9. Branch from current upstream FastEmbed main and make one clean PR containing all three native
   entries plus reference-derived canonical vectors. Do not include the unrelated #703 fix. Run
   the upstream tests and wait for Nano before opening the PR, as already ruled.
10. Confirm the Hub repository is public and the downloaded serving path reproduces the frozen
   parity/results. Record commit/revision URLs and only then request storage-retirement authority.

No publication command has been run by M13.

Run the reserved controller from the preserved artifact worktree after confirming it is detached
at the exact pushed terminal `main` commit:

```bash
cd /home/dylan/asymetric-dual-encoders/work/m13cloud
git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
.venv/bin/python scripts/m13_reserved_cloud.py
```

The controller refuses a dirty or unpushed local tree, switches the clean retained cloud checkout
to `origin/main`, binds the transaction to that commit, and stops the paid pod in `finally`. It
must not be retargeted to a deleted M13 topic branch.
