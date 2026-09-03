# M10 status — PLANNED 2026-09-01 on branch `m10-work`; nothing has trained; waits on the GPU budget

Mandate `instructions-m10.md` · evidence `m10/PLANNING.md` · runs `m10/RESULTS.md` · closed
avenues `m10/EXPLORED.md` · selection-surface drafts `m10/COV_CANDIDATES.md` · code `m10/CODEMAP.md`
· M9's record `m9/FINDINGS.md`. Owner report: https://claude.ai/code/artifact/fce61c94-5444-4c78-bb2e-46112cb7547a

**Where things stand.** M9 is closed as a measurement and merged to main; its six-set close-out
waits for M10's recipe lock and Dylan's ratification. M10's plan went through six Codex passes, then
Dylan's compute ruling (2026-09-01): **M10 runs on a rented GPU budget or not at all**; the RTX 3080
is out. With it the plan took LEAF's dose (200M examples), a 5M screen dose, three more screen arms,
a descriptive COV resolution number and a $1,000 ceiling (PLANNING §8, amendment block; Codex pass 7
actioned). Three Mac diagnostics fixed the architecture: a 384-wide linear head binds under L2
regression once queries are diverse, so the student pools three layers (1152-d, 34.5M parameters,
fastembed-exact); a four-layer 1536-d arm is screened too. The 35M cap is hard. FineWeb is out.
`m10/LEDGER.md` is a committed skeleton; the GPU session fills it at M10.0-d/e.

## Dylan — open decisions (defaults apply meanwhile)

| # | decision | default |
|---|---|---|
| 1 | Ratify M9's final lock plus the six-only amendment (one sentence: "run M9's six-set scoring as registered, six only, no reserved batch") | blocks only the M9 close-out |
| 2 | **GPU budget:** expected $400–715 with generation on the GPU, $465–895 if generation is hosted; ceiling $1,000 hard; itemized in PLANNING §6; one A100 80 GB, provider of your choice | **blocks every GPU stage**; refusal closes M10 unstarted |
| 4 | PAQ as query text (CC BY-SA data, official release) | include |
| 7 | Confirm LoTTE read #1 withdrawal and the M10/M11/M12 renumbering | as recorded |
| 8 | Second build seed (≈ 100 GPU-hours, $150–250) as the headline's replication band | runs only if ≥ 100 GPU-hours at the billed price remain under the ceiling after every mandatory line; seed 0 alone controls C1/C2, release, recipe and headline; seed 1's rows are descriptive |
| 9 | The 2026-09-01 review amendments (200M dose, 5M screens, D-KL1 / D-NCE / G-1536, COV resolution number, seed-rank field) | adopted; strike any item and it reverts to the pass-6 text |

Optional lever not yet taken: cap confirmation seeds at two decisions instead of four (saves ≈ 20 GPU-hours; PLANNING §5).

## Mac — before the budget is approved (no GPU; run only when Dylan says so)

1. **Prompt prototyping for the 12 forms** with `mlx-community/Qwen3-8B-4bit` via mlx-lm (~5 GB,
   ~80 min): 200 queries × 12 forms from `m10src/forms.py` over pre-filtered Wikipedia seed passages
   under the §Data contract, saved to `results/m10_forms_prototype_mac.json` for Dylan to read. This
   pass settles the prompts and nothing else: it produces no smoke result, no generated example and
   no provenance record. The approving 200-per-form smoke runs on the GPU with the pinned bf16
   artifact (90% contract / 80% on-form; at most two prompt revisions per form).
1b. MiniLM-L6's three-layer head and both students' four-layer heads through
   `m10src/head_width_parity.py` (CPU, minutes), so families F and G may run those arms.
2. Port the trainer to the M10 recipe with a CPU smoke on a tiny model: cyclic schedule,
   example-mix batcher, three- and four-layer pooled heads, the phase-2 KL term and its D-KL1 and
   D-NCE variants (the literal 129-way loss in the mandate §Recipe), `test_resume.py` equivalence,
   an examples/s counter in the log.
3. The PAQ sampler against the official Facebook release (1.0M build sample, 4.037M A2 sample, hashes).
4. The COV fingerprint-screen script (reuse `m7src/decontam.py`) and the COV resolution-number
   script (the contrast bootstrap on e5-small-v2 vs gte-small, distance only), both run on the GPU
   instance at admission.
5. The M10.1 generation harness for vLLM (per-form quotas, JSON contract from `forms.py`,
   provenance records incl. the seed-rank field, end-to-end requests/s log).
6. An instance bootstrap script: clone, env, HF pulls, sha checks of `results/perquery.json`
   (`6b18e3dd…`) and of the transferred M9 checkpoint, regeneration of M9's parity sample from
   `m9/registry.json`, and the day-one rate benchmark (mandate §Compute).

## GPU instance — day one, in order (after decision 2)

1. One A100 80 GB, ≥ 500 GB persistent disk, deploy key for `m10-work`; `git pull`; `nvidia-smi`;
   `./run_tests.sh` and `./run_m8_tests.sh` green; sha-verify `results/perquery.json`. Transfer
   `work/m9long/ckpt/last.pt` from the box (sha `9d631b2c…`, verified) and any `work/` file
   `m9src/guard9.py` hashes. **Day-one rate benchmark (≈ 2 GPU-hours):** stella docs/s, examples/s
   at batch 32 on the 75/25 mix, the 50/50 mix and MiniLM-L6, generation requests/s per form on the
   200-query smoke, the billed $/h → **re-derive PLANNING §6, push it, and pick scenario A or B**
   before anything scales. Then re-derive pool, dev suite, fingerprints and encodes (PLANNING §5).
2. M10.0-c: per-component DEV-6 read of the M9 candidate incl. `heldout-longq` (baseline row).
3. M10.0-d: COV admission — verify LEDGER's structure and chunk count; fingerprint screen and
   corpus-level check per component; pushed records in `m10/LEDGER.md` §2; **three untouched
   families** minimum (the CQADupStack pair is DEV, not COV); add every admitted corpus, query set
   and document set to the protected index; encode the admitted corpora with stella; measure and
   push the **COV resolution number** (descriptive; it decides nothing).
4. M10.0-e: `m10/LEDGER.md` §0 screen lock (fourteen arms, sixteen contrasts, A2/A3 matched counts
   and hashes, τ rule, the literal D-NCE loss, Bonferroni, confirmation design, DEV-6-once rule)
   pushed before any arm runs.
5. M10.1: the approving generation smoke on the GPU with the pinned bf16 artifact (Dylan reads it),
   then generation at scale under scenario A or B; PAQ; decontamination with the fixed thresholds;
   FORMS-12 hold-out; teacher targets incl. seed passages; mining smoke then mining; manifest with
   the seed-rank field; τ table.
6. M10.2: arms → **re-derive and push PLANNING §6 from the measured arm rates before the remaining
   arms** → confirmations → the synthesized selected-recipe arm → lock commit pushed with the
   allocation under the ceiling (`max_extension_cycles`, decision 8 boolean and seeds) → Codex and
   Fable review → `final9.py` scoring path + `if C1 or C2` change → M9's six-only close-out from
   `m9-work` (decision 1) → encode LoTTE-clean → LoTTE read #1.
7. M10.3 build (200M examples ≈ 100 GPU-hours at the planning rate; whole extension cycles per the
   mandate's rule, up to `max_extension_cycles`) → export, parity on M9's regenerated sample, freeze,
   pre-freeze review, LoTTE read #2 → seed-1 replica if decision 8's condition holds. M10.4 final.
   Stop the instance between stages; the disk stays.

## Guardrails that bite here

No six/reserved/LoTTE access outside the registered transactions. `results/perquery.json` is never
rewritten. Never edit a `guard9` protocol-scope file before M9's close-out runs. Every review brief
carries the reserved read-exclusion; grep the log after. Long runs: smoke, arm the
failure-signature monitor, check the rate, watch the machine. Stella on the Mac runs only in
`.venv-mac`. A stopped instance costs disk only; an idle running one costs the budget.
