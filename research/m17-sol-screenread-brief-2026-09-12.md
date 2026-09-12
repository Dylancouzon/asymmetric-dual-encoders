# M17 Sol review brief — the 6d screen-read entry point, narrow (2026-09-12)

READ-ONLY adversarial review. Edit nothing; run nothing that writes under `results/`, `work/`
or `m17/`. Findings only, under 30 lines. Narrow scope: ONE commit.

## Context

Repository `/home/dylan/asymetric-dual-encoders`, branch `m17-zero-v1.1-planning`, HEAD `7ea4180`.
M17 is on the clock, `LOCKED_EXECUTABLE`. The single V0 read is done and committed
(`results/m17_v0_read.json`, `reads: 1`, registry `v0_export.read: true`). Step 6d trains five
screen arms (C, V, L, VL, VL-A; 4,000 steps, seed 0) and each arm gets exactly ONE quality read on
the pinned dev suite (`training.quality_reads_per_screen_arm = 1`). Commit `7ea4180` adds
`screen_read()` in `m17src/evaluate.py` as a sibling of the reviewed V0 reader: same gate
(`--allow-dev-suite`, `LOCKED_EXECUTABLE`, executed lock half), same preflight, clean-tree refusal,
atomic receipt and no-overwrite; binds the arm's own trained bundle; refuses the locked V0 bundle;
writes only `work/m17/runs/<run_id>/screen.json`; never touches `v0_export.read` or
`results/m17_v0_read*`. The V0 reader itself was reviewed three times (Astra, Sol, Astra re-check;
`m17/REVIEW.md`). Do not re-review it; break only the sibling and its interaction.

## Files you may open (no others; no `grep -r`, `rg`, `find`)

- `git diff 2aa5507 7ea4180` (the whole change), `m17src/evaluate.py`, `m17src/test_evaluate.py`
- `m17src/export.py`, `m17src/train.py` (only the run_id / bundle output parts)
- `m17/registry.json` (`training`, `training.decision_protocol`, `screen_routing_surface`)
- `work/m17/logs/run_6d.sh`
- You may run `.venv/bin/python -m pytest m17src/test_evaluate.py -q` with
  `TMPDIR=/home/dylan/asymetric-dual-encoders/work/m17/scratch/pytest_tmp`.

Read-exclusions: never open `results/frozen_eval/untouched-*`, reserved qrels caches,
`work/m9reserve`, anything about FEVER, DBpedia-entity, cqadup-android, cqadup-english; never read
`work/m17/prepared/full`, `work/m17/bundles/V0` or `results/m17_v0_read*` as a quality surface;
never execute `screen_read`, `dev_suite_read`, `--screen`, `--surface dev-suite` or the launcher.

## Questions

1. Can `screen_read` write to, read, or be confused with the V0 result/receipt, or mutate
   anything under `results/`?
2. Does it bind the trained bundle it scores (digests recomputed with the exporter's conventions,
   recorded in the receipt), and can an arm's read be silently repeated (second read on the same
   arm overwriting or duplicating `screen.json`)?
3. Does the `verify_bundle` hook change the V0 path's behaviour in any way?
4. In `run_6d.sh`: can a resumed or re-run launcher spend a second read on an arm, skip an arm
   wrongly, or run an arm against the wrong prepared directory (C/L on `base`, V/VL/VL-A on `ext`)?
5. Does the surface match `screen_routing_surface` exactly (six components, int8 folded, depth,
   nDCG@10 and Recall@10 with equal-weight macro), and is `decision_protocol.technical_metric`
   (cqadup-programmers nDCG@10) present in `screen.json`?

Output: P1/P2/P3 with file:line, one-sentence failure scenario, smallest fix. Say explicitly
whether anything blocks launching the five arms.
