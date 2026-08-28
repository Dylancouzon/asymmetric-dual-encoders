# M7 status

**Stage: review #3 + #3b findings implemented; the corrected dev audit is running (2026-08-28 early).**
Candidate: **whatever `results/m7_dev_audit_full.json` names as `surviving_candidate`** — the three
lever decisions (500k adoption, 2m cross-arm pick, s2500 extension) are being re-judged under
dependence-preserving statistics, and the pre-registered rule in `m7/LEDGER.md` lets any of them
revert. Until that file exists, no run id may be called "the candidate".

Two Codex reviews are actioned, both verbatim in `research/`: `m7-codex-review-2026-08-27.md`
(4 BLOCKER / 8 MAJOR / 4 MINOR) and `m7-codex-review-2026-08-27b.md` (3 BLOCKER / 5 MAJOR /
6 MINOR, on the repair itself). Accepted reframe: **every dev p-value is SELECTION evidence**;
only the three frozen-test comparisons are confirmatory.

## Done since review #3

- **Dependence** (`boot.signflip_dep` / `paired_dep`, `test_dep_stats.py`): one shared sign per
  underlying qid, stratified bootstrap, and a three-way report that separates conditioning from
  covariance. The dependence-blind interval is 1.43x too narrow under full duplication.
- **Pin** (`freeze_heldout.py`): all six dev components hash-pinned including the pool's bytes,
  spans and store hashes; nesting properties asserted, not merely recorded; `dev_components()`
  aborts on a missing or changed component; the audit refuses to run unpinned. Late pin, disclosed.
- **Ablation harness**: `Cfg.init_preproc` separates runtime prefixing from prefix-conditioned
  rows; `sweep.chain/chains` runs every arm as two runs (B, then a fresh A from that checkpoint);
  `input_emb_rows` sliced to `tok.vocab_size`; reg control is 1e-3-vs-0; 7 chains + the two
  pseudo-query attribution controls + one labelled exploratory chain (`run_ablations.sh`).
- **Equivalence**: matrix shortcut vs released `QueryTable` measured per query AND per ranking.
- **Provenance**: per-query dumps, unrounded CIs, table/encoder/code hashes; bigram fit cache is
  content-addressed; doc2query expansions hashed with their recipe.

## Run next, in this order

1. Read `results/m7_dev_audit_full.json` -> `surviving_candidate`, and record the verdict in
   `m7/LEDGER.md` and `m7/RESULTS.md`.
2. `./run_ablations.sh` (attribution controls first, then the 7 mandatory chains, then the
   exploratory one). ~20 min per chain.
3. Lever #4 (count saturation) verdict from `results/m7_lever4_pooling_full.json`; if adopted,
   `Preproc` grows `pool_mode`, conformance extends, artifacts re-saved before anything downstream.
4. Fusion re-selection on the surviving candidate (int8, DEPTH=1000, incl. convex0) -> ANN sweep
   + costs -> gate as a mechanical eligibility audit -> freeze -> single final run.

## Open for Dylan

1. Nothing blocking.
2. doc2query revival (licensing ruling on a clean generator) remains yours if ever wanted.
3. Windows Update reboots remain the top operational risk.
