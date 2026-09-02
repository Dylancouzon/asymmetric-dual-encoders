# M10 status — PLANNED 2026-09-01; branch `m10-work`; waiting on the training box (Dylan travelling)

Mandate `instructions-m10.md`; evidence `m10/PLANNING.md`; M9's record `m9/FINDINGS.md`.
Nothing has trained. The Mac ran the rank-bottleneck probe (`results/m10_rank_probe_mac.json`, PLANNING §9): a 384-wide linear head is at the aim's ceiling, so the default student pools three layers (1152-d feature, family G).

## Next, in order

**Dylan (from anywhere):** rule on the six owner decisions in `instructions-m10.md`
§Owner decisions — ratification of M9's final lock is the one that blocks work.

**First session on the box** (or a cloud GPU, if funded):
1. `git pull`; `nvidia-smi`; disk ≥ 200 GB free; `./run_tests.sh` and `./run_m8_tests.sh` green.
2. M10.0-b/c: `m9src/capacity_probe.py` unchanged; per-component DEV-6 of the M9 candidate. M9's close-out later runs from `m9-work` (guard9 pins that branch).
3. M10.0-d: COV admission — licence evidence, contamination check, fingerprint screen per
   candidate component; record in `m10/LEDGER.md`; **four admitted families** minimum (not four components).
4. M10.0-e: `m10/LEDGER.md` §0 (screen lock: nine arms, τ rule, Bonferroni, seeds) pushed before
   any arm runs.
5. M10.1: generation (Qwen3-8B 4-bit via vLLM; 200-query smoke per form, read by a person, rate
   measured, then scale), PAQ sample, decontamination with the fixed thresholds, FORMS-12
   hold-out, targets, mining, manifest.
6. M10.2: screens, confirmations, the synthesized selected-recipe arm → lock commit pushed → Codex/Fable review → **only then**
   `final9.py`'s scoring path and M9's six-only close-out (its reserved conditional struck by a
   ratified amendment; needs Dylan's decision 1) → LoTTE read #1.

**Mac, meanwhile:** data pipeline code (`m10src/`), the 12 form prompts, the PAQ sampler, the
COV admission records (licence URLs, revisions, sizes), and the M10.0-a2 fastembed parity check of
the three-layer per-token head (CPU, reuse `m9src/port.py`).

## Guardrails that bite here

No six/reserved/LoTTE access outside the registered transactions. `results/perquery.json` is
never rewritten. Every review brief carries the reserved read-exclusion; grep the log after.
Long runs: smoke, arm the failure-signature monitor, check the rate, watch the machine.
