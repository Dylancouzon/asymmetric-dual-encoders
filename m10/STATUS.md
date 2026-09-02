# M10 status — PLANNED 2026-09-01 on branch `m10-work`; nothing has trained

Mandate `instructions-m10.md` · evidence `m10/PLANNING.md` · runs `m10/RESULTS.md` · closed
avenues `m10/EXPLORED.md` · selection-surface drafts `m10/COV_CANDIDATES.md` · code `m10/CODEMAP.md`
· M9's record `m9/FINDINGS.md`. Owner report: https://claude.ai/code/artifact/fce61c94-5444-4c78-bb2e-46112cb7547a

**Where things stand.** M9 is closed as a measurement and merged to main; its six-set close-out
waits for M10's recipe lock and Dylan's ratification. M10's plan went through five Codex passes.
Three Mac diagnostics fixed the architecture: a 384-wide linear head binds under L2 regression once
queries are diverse, so the student pools three layers (1152-d, 34.5M parameters, fastembed-exact).
The 35M cap is hard. FineWeb is out. `m10/LEDGER.md` does not exist yet — it is created on the box
at M10.0-d/e.

## Dylan — four open decisions (defaults apply meanwhile)

| # | decision | default |
|---|---|---|
| 1 | Ratify M9's final lock plus the six-only amendment (one sentence: "run M9's six-set scoring as registered, six only, no reserved batch") | blocks only the M9 close-out |
| 2 | Money: hosted open-weights generation (≈ $110–330) removes 2–4 GPU-days from the box; a cloud A100 (≈ $120–280) compresses the rest | box-only, ≈ 4–5 weeks |
| 4 | PAQ as query text (CC BY-SA data, official release) | include |
| 7 | Confirm LoTTE read #1 withdrawal and the M10/M11/M12 renumbering | as recorded |

Optional lever not yet taken: cap confirmation seeds at two decisions instead of four (saves ≈ 1 GPU-day; PLANNING §5).

## Mac — before the box is reachable (no GPU; run only when Dylan says so)

1. **Per-form generation smoke with Qwen3-8B 4-bit via mlx-lm** (~5 GB download, ~10 min of
   generation): `pip install mlx-lm`, load `mlx-community/Qwen3-8B-4bit`, 20 queries × 12 forms
   from `m10src/forms.py` over Wikipedia seed passages, save to `results/m10_forms_smoke_mac.json`
   for Dylan to read. The prompts are adjusted from that read, before any scaling.
2. Port the trainer to the M10 recipe with a CPU smoke on a tiny model: cyclic schedule,
   example-mix batcher, three-layer pooled head, phase-2 KL term, `test_resume.py` equivalence.
3. The PAQ sampler against the official Facebook release (1.0M build sample, 4.037M A2 sample, hashes).
4. The COV fingerprint-screen script (reuse `m7src/decontam.py`), run on the box at admission.
5. The M10.1 generation harness for vLLM (per-form quotas, JSON contract from `forms.py`,
   provenance records, rate log).

## Box — day one, in order

1. `git pull` on `m10-work`; `nvidia-smi`; ≥ 200 GB free; `./run_tests.sh` and `./run_m8_tests.sh` green.
2. M10.0-c: per-component DEV-6 read of the M9 candidate incl. `heldout-longq` (baseline row).
3. M10.0-d: COV admission — verify LEDGER's structure; fingerprint screen and corpus-level check
   per component; pushed records in `m10/LEDGER.md`; **four admitted families** minimum; then add
   every admitted corpus, query set and document set to the protected index; encode the admitted
   corpora with stella.
4. M10.0-e: `m10/LEDGER.md` §0 screen lock (eleven arms, thirteen contrasts, τ rule, Bonferroni,
   confirmation design, DEV-6-once rule) pushed before any arm runs.
5. M10.1: generation smoke (if not done on the Mac), then generation at scale; PAQ; decontamination
   with the fixed thresholds; FORMS-12 hold-out; teacher targets; mining smoke then mining;
   manifest; τ table.
6. M10.2: eleven arms → confirmations → the synthesized selected-recipe arm → lock commit pushed →
   Codex and Fable review → `final9.py` scoring path + `if C1 or C2` change → M9's six-only
   close-out from `m9-work` (decision 1) → encode LoTTE-clean (~4 h) → LoTTE read #1.
7. M10.3 build (2.6–4 days plus the batch-32 penalty) → export, parity on M9's locked sample,
   freeze, pre-freeze review, LoTTE read #2. M10.4 final.

## Guardrails that bite here

No six/reserved/LoTTE access outside the registered transactions. `results/perquery.json` is never
rewritten. Never edit a `guard9` protocol-scope file before M9's close-out runs. Every review brief
carries the reserved read-exclusion; grep the log after. Long runs: smoke, arm the
failure-signature monitor, check the rate, watch the machine. Stella on the Mac runs only in
`.venv-mac`.
