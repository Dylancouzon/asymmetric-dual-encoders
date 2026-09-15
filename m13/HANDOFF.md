# M13 terminal handoff

M13 is closed and unpublished. Fetch `origin/main` and tags, then require
`refs/tags/m13-closed^{commit}` to be an ancestor of `origin/main`. The complete release runbook
for the different M14 agent is `m14/HANDOFF.md`; this file records the boundary and compact result.

## Completed in M13

- Built and froze constella-nano at 199,999,721 examples; reconciled the 279-example shortfall.
- Verified checkpoint/ONNX/tokenizer assets, reference-to-ORT and stock-FastEmbed parity, and two
  independent local artifact copies.
- Completed Nano's registered six-set exact-search evaluation and M9's independent six-only
  closeout, each with durable spent tags and authenticated per-query rows.
- Produced the registered descriptive Nano-minus-M9 comparison and same-machine serving costs.
- Prepared, tested, committed, and pushed the separate M14 reserved execution path, including
  corpus-only pre-encoding, protected scoring, atomic resume, cloud cost checks, and STOP-in-finally.
- Preserved historical launchers/evidence under `audit/m13/` and left all retained pods stopped.

## Results

| dataset | Nano | M9 | Nano − M9 |
|---|---:|---:|---:|
| SciFact | 0.721097 | 0.634021 | +0.087076 |
| NFCorpus | 0.363080 | 0.313345 | +0.049735 |
| FiQA† | 0.477765 | 0.214259 | +0.263507 |
| ArguAna† | 0.623296 | 0.542813 | +0.080484 |
| SCIDOCS | 0.217710 | 0.142694 | +0.075017 |
| TREC-COVID | 0.787116 | 0.382908 | +0.404208 |

† Stella discloses training/evaluation contact. Nano-minus-M9 is +0.154009 on clean four
(95% interval [0.132215, 0.175887]) and +0.160004 on all six ([0.144713, 0.175498]). Nano's
registered BGE and LEAF conclusions, including the TREC-COVID limitation, are in `m13/STATUS.md`.

Result identities:

- Nano: `results/m10_final_run.json`,
  `f5b5ad8a63060fbe1184aa3e5319259855d3ca4a9e3c3de1e91eeb76ac685b23`.
- M9: `results/m9_final_run.json`,
  `770344d7a7c18d0933e0f922af385fb0b4e514c98822e955e959cf139a9c936b`.
- Paired: `results/m13_paired_m9_nano.json`,
  `4eaf20ebef5548338314fd547abd0f76c55eae8b5334e3be2f11260d2f8071b4`.

## Mandatory M14 work

Owner ruling R19 moved the triggered A100 execution to M14. M13 intentionally preserves Nano's
`INCOMPLETE_RESERVED` marker and did not resume the A100, create `m8-reserved-spent`, or open any
reserved query/qrel payload. Before public release M14 must:

1. run the unchanged registered reserved four (FEVER, DBpedia Entity, CQADupStack Android and
   English; Nano, BGE-small and LEAF; descriptive, zero alpha) with the committed controller;
2. run the full BEIR-18 suite as separate broad descriptive validation with overlap caveats;
3. perform final packaging and real boundary-length parity tests, upload privately, verify a fresh
   download byte-for-byte and behaviorally, then make the repository public;
4. create one clean FastEmbed PR from current upstream main without the unrelated #703 padding fix.

The release recommendation is to proceed unless those remaining checks reveal gross overfitting
or catastrophic failure. Missing an ambitious superiority target alone is not a release veto.

## Artifacts and infrastructure

- Checkpoint: `work/m13-final/cycle3.pt`, SHA-256
  `3e49e0bfaa633abe276da7847d0ac44cbfb3d7d5215b9343f255706eeb5789a1`.
- Verified checkpoint backup:
  `work/m13cloud-build-preflight-retry-backup/work/m13build/BUILD-200M/cycle3.pt`.
- ONNX: `work/m13cloud-build-preflight-retry-backup/work/m13build/BUILD-200M/onnx/model.onnx`,
  SHA-256 `9ba0acf57b71dc31bc5512c5445078a797fa51cf3e85587d6b8a506bfc55dbc2`.
- Backup manifest: `work/m13cloud-build-preflight-retry-backup/manifest.json`, SHA-256
  `cb5341e8e9bcff8f07c650345b484ea5d19e6b9d9987cc1282b5fb03337c2602`.
- Target A100 pod: `k3aee2m68765em`; all three retained pods were `EXITED` at the last provider
  check. Authority remains STOP-only; do not terminate pods or volumes without owner approval.
- Credentials remain under `/home/dylan/.config/runpod/`; never copy values into Git or logs.
- The artifact worktree must be preserved. It may be detached at the terminal `origin/main` commit
  after integration; do not delete it because the large ignored artifacts are M14 inputs.

Validation at closure: 267 M13 tests, all 36 M8 guard checks, focused controller tests, Python
compilation, and the earlier real cached corpus-only metadata preflight all passed. No monitor or
benchmark process remains active. Machine-readable closure state is
`results/m13_live_status.json`; storage evidence is `results/m13_storage_closure.json`.
