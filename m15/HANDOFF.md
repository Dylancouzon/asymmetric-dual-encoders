# M15 handoff (2026-09-30)

Start here in a fresh session on branch `m15-whitepaper`. Read, in order: this file, `m15/PLAN.md`
(v5.3, agreed with the owner and reviewed adversarially by Astra), `m15/EVIDENCE_INDEX.md` "Corrections 2026-09-30", and the last entry of
`m15/LOG.md`. The owner's review page is https://claude.ai/artifact/U2Ph1TXSUmLDXG7bHXRKkN.

## State

- The paper is a research paper (CLAUDE.md and `instructions-m15.md`, both amended 2026-09-30). It
  leads with retention as query compute drops. The registered test goes in full in the evaluation
  section. `m15/PAPER.md` is the superseded v2 draft; the next draft is written fresh from `PLAN.md`.
- All M20 numbers are in: `results/m20_beir15_run.json`, `results/m13_reserved_run.json`.
- Nothing has been measured for M15 yet. No method file exists yet.

## Next steps, in order

1. **Write `m15/MEASUREMENTS.md`**, a dated method file for E1-E8: estimand, data, exact procedure,
   outputs (`results/m15_*.json`), and what each result can and cannot show, as `PLAN.md` states it.
   Freeze it before any run. Then one Codex Astra review (`codex exec -m gpt-6-astra -s read-only`)
   with the brief rules below, fix P0/P1, commit.
2. **Mac runs**, sequentially: E5 (committed per-query rows, minutes), E6, E1, E4, then E2 (overnight
   Stella encode of the 1M MS MARCO subset first), and E7 if wanted. Each writes a compact receipt.
3. **E8 on Runpod** once the cleaned fit list is recovered and the owner has supplied the items in
   "What Runpod needs". Encode, fit and
   score on the pod; bring back only the result JSON and a receipt.
4. **Figures F1-F4** from committed JSON (`m15/figures/`), then **draft v3** of `m15/PAPER.md` in the
   `PLAN.md` outline. Gates on the draft, in order: Astra correctness, Fable hostile read,
   `/andrey-review`, the Codex writing pass (`gpt-6-sol`), `/humanizer` last. Then LaTeX for arXiv.

## Rules that bind every step

- Reserved four (FEVER, DBpedia-entity, cqadupstack-android, cqadupstack-english) are spent and
  known-test: no role in any M15 measurement, including fitting, thresholds or diagnostics. Their
  committed aggregate results are quoted, never recomputed.
- Never read `results/frozen_eval/untouched-*`, reserved qrels caches, `work/m9reserve`. No
  repo-wide content search across `results/` or `work/`. Never overwrite `results/perquery.json`.
- Every review brief names files, forbids recursive searches and web searches naming a reserved
  dataset, states the essential-only rule and the goal, and its access log is audited afterwards.
- Exact search makes quality numbers. ANN recall and nDCG under ANN are deployment behavior and stay
  labelled as such.
- BEIR-15 and reserved rows are descriptive: no superiority, equivalence or significance claims.
- Commit and push rules follow the user's global instructions: commit or push only when Dylan asks in
  that turn.

## Mac pitfalls

- 24 GB unified memory: one MPS job at a time, both MPS watermark variables set, encode batch 64,
  `nohup` for overnight runs (auto-memory `m5-pro-benchmarking-constraints`).
- Stella loads only in `.venv-mac` (transformers 4.57). Pass
  `config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}`. Pin fp32.
- Stella's tokenizer pads to 512 unless `no_padding()` is called. Stella queries need the `s2p_query`
  prompt. Upstream FastEmbed returns Nano as float64; cast to float32.
- Qdrant: use the native macOS binary for latency; Docker (a Linux VM on macOS) only for memory-limit
  runs. `qdrant_client` 1.19.0 and `qdrant_edge_py` 0.8.0 are in `.venv-mac`. TurboQuant needs Qdrant
  1.18 or later.
- Local assets: Stella, bge-small and the Zero bundle are in the Hugging Face cache; Nano and the
  Stella document ONNX download from `Qdrant/`. Stella document vectors exist locally for FiQA and
  SciFact only (`artifacts/demo-stella/`); check they match the registered encoding before reuse.

## Data not on this Mac

- **E8's fit list is the blocker.** Use M8's cleaned list, `work/m8_trainq_texts.json`: 337,981
  queries, 21,327,603 bytes, sha256 `da0f208ef29bae3833ffead070eb34b61f390f9d72ed96d448d45ae6b52072c2`
  (`results/m8_trainq_manifest.json`). It is on the WSL box. Verify the hash after copying. Do **not**
  use M7's `work/trainq_texts.json` (349,934 queries, 4,582 protected-query hits, `m7/LEDGER.md:274`),
  and do not rebuild either list: the filter reads protected queries. If the file cannot be recovered,
  E8 stays blocked and the owner decides.
- The M20 archive (142.6 GiB) is on the box's D: drive. E2 does not need it: it encodes its own 1M
  MS MARCO subset on the Mac.

## What Runpod needs (for E8)

From the owner:
1. A new Runpod API key, stored outside git (for example `~/.runpod/config.toml` through
   `runpodctl config --apiKey ...`, or a `chmod 600` file). Never in git, logs or a pod.
2. This Mac's public key (`~/.ssh/id_ed25519.pub`) added under Runpod Settings, SSH Public Keys.
3. The current account balance and a spend cap for E8. The recorded cloud ceiling is $1,000
   (CLAUDE.md), and the three retained stopped pods bill about $0.41/h (`m20/STATUS.md`, open item 2),
   so the headroom must be read from the live balance, not assumed. Proposed cap: $40.
4. A ruling on the three retained pods: whether any volume holds the cleaned fit list or the six-set encodes
   (restart one with a GPU), or whether E8 uses a fresh pod.

Planned pod: one A100 80GB or RTX 4090, 200 GB container disk, no network volume. Estimated 3-5 GPU
hours for 11 configurations over ~272k documents plus the fit-list queries.

## Runpod state, checked 2026-09-30

- `runpodctl` 2.14.0 is installed; the key is in `~/.runpod/config.toml` (mode 600, outside git).
  `runpodctl me` works. Never print the config file.
- Balance $219.57. Current spend $0.417/h: the three stopped M13 pods with 500 GB volumes each
  (`runpodctl pod list --all`): `m13-gate-chain-20260911` (`wnzk8eeqrrkw4m`, no GPU),
  `m13-replacement-20260911` (`exulxoxelug5um`, A100), `m13-a100-benchmark` (`k3aee2m68765em`, A100).
  Pod billing since 2026-09-11 sums to $285.01 (`runpodctl billing pods`), so the $1,000 ceiling
  leaves roughly $700; E8's proposed cap is $40, which still needs the owner's number.
- **Possible E8 unblock without the WSL box:** `results/m13_build_record.json` pins
  `m8_manifest_sha256 = da0f208e...`, the same hash as the cleaned fit list, so the file may sit on
  one of these volumes. Checking means starting a pod, which costs money: ask the owner first, and
  prefer the GPU-less `m13-gate-chain` pod for a look.
