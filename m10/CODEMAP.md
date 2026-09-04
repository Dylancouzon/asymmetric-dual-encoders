# M10 code map

`m10src/` imports `m7src` and `m9src` and edits neither. `m8/CODEMAP.md` and `m9/CODEMAP.md` still apply.

| module | what | artifact |
|---|---|---|
| `rank_probe.py` | PCA-k projection of stella query vectors, retrieval retention on the CQA dev components; 1024/768/256-d heads | `results/m10_rank_probe_mac.json` |
| `rank_probe_mix.py` | the same from caches, with bases fit on NQ, the other component, mixtures, oracle | same file, `mixture_bases_1024d` |
| `head_width_probe.py` | frozen bge-small + ridge head to stella targets with 384 / 768 / 1152-d pooled features | `results/m10_head_width_probe_mac.json` |
| `head_width_parity.py` | ONNX export of the per-token three-layer head; fastembed serving parity | `results/m10_head_width_parity_mac.json`, `work/m10onnx/nano-3layer/` |
| `head_mlp_parity.py` | the same export/parity check for a per-token **nonlinear** head; `residual` mode (`W_lin·x + W₂·GELU(W₁·x+b₁)`, W₁ 1152→192) is arm G-MLP, `mlp` mode is the cut bottleneck form; proves the serving path and the parameter count (34.96M) | `results/m10_head_mlp_parity_box.json`, `work/m10onnx/nano-3layer-mlp/` |
| `scripts/m10_conjunct_arithmetic.py` | per-dataset retention each conjunct demands, uniform-retention planning proxies (0.025 quantile), per-dataset stress scenarios, fiqa's share of the avg-6 margin (comparator rows only) | `results/m10_conjunct_arithmetic.json` |
| `forms.py` | the 12 synthetic-query form prompts, output contract and parser (self-check in `__main__`) | — |
| `m10/report-draft.html` | source of the owner report artifact (https://claude.ai/code/artifact/fce61c94-5444-4c78-bb2e-46112cb7547a); republish from a session, never edit the live page | — |

## Pitfalls this milestone earned

1. **Stella runs on the Mac only in `.venv-mac`** (transformers 4.57); transformers 5.x breaks its remote code (`get_extended_attention_mask(..., device=)`). MPS: ~170 short queries/s, 20–100 documents/s by length.
2. **fastembed 0.8.0 custom models need `config.json` and `special_tokens_map.json` in the model directory**, and transformers 5.x fast tokenizers no longer write `special_tokens_map.json` — write it from `tok.special_tokens_map`.
3. **A per-token linear head over concatenated layer states is exactly reproduced by fastembed's mean pooling** (2e-7), because mean pooling is linear. A nonlinear head is not.
4. **PCA of teacher vectors is the reconstruction-optimal subspace, not a retrieval bound.** Say "the subspace L2 regression aims at", never "upper bound" (Codex pass 4).
