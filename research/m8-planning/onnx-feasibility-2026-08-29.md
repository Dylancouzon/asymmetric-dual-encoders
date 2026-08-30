# ONNX feasibility: stella_en_400M_v5 → fastembed (Sonnet fact-check, 2026-08-29)

*Verbatim report of a Sonnet web fact-check commissioned after Dylan's 2026-08-28 ruling that M8
must (eventually) be servable via ONNXRuntime in fastembed. Verdict: parity-verified stella export
is DAYS, not weeks, and not blocked.*

## 1. NovaSearch/stella_en_400M_v5 — existing ONNX exports

**No official ONNX in the repo** (no `onnx/` on `main`; same for the older dunzhang mirror).
Community activity, fragmented and unmerged:
- **Discussion #3 "Upload ONNX weights"** — Xenova (HF/Transformers.js maintainer) produced ONNX
  weights (ONNXSlim-simplified), PR **still open/unmerged**. Reported issues: IR-version-10
  incompatibility on older onnxruntime; TEI integration failure (output named
  `sentence_embedding`, TEI expects `last_hidden_state`).
- **Discussion #12** — an independent naive `optimum-cli export onnx` attempt **fails**:
  `Found an unsupported argument type c10::SymInt in the JIT tracer`. Unresolved as filed.
- **Discussion #11** — root cause of both: stella's default `config.json` sets
  `unpad_inputs: true` and `use_memory_efficient_attention: true`; the latter routes through an
  xformers CUDA-only kernel with no ONNX/CPU equivalent. Maintainer's documented workaround:
  instantiate with `config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}`
  before tracing.
- `onnx-community`: no stella conversion.
- `magiccodingman/stella_en_400m_v5_onnx` (a **dataset** repo): FP32 + INT8 1024-d variants, empty
  README, **no parity/MTEB validation — unverified quality**.
- TEI issue #359 (native serving) open since July 2024, no action.

**Net:** export achievable and done at least once by a credible party; **no first-party, merged,
parity-verified artifact exists**. stella_en_1.5B_v5: no onnx/ found; unverified, lower confidence.

## 2. Alibaba-NLP/gte-large-en-v1.5 (parent architecture)

**Ships first-party ONNX, multiple precisions** in `onnx/`: model.onnx 1.75 GB, fp16 873 MB,
int8/uint8/quantized ~446 MB, q4 387 MB, bnb4 361 MB; Transformers.js-tagged (standard
Xenova/optimum pipeline works).

**Key architectural fact:** gte-large-en-v1.5 defaults `unpad_inputs: false` and
`use_memory_efficient_attention: false` — the opposite of stella. Same custom `NewModel` class
(auto_map → Alibaba-NLP/new-impl, RoPE, 24 layers). With the xformers-only ops off, tracing hits
only standard ops and succeeds. **The blocker is a config-flag default, not an architectural
incompatibility.** Known family issue: gte-base-en-v1.5 discussion #12 — ONNX correct but *slower*
than PyTorch on long sequences on CPU (crossover effect, not a parity bug).

## 3. fastembed requirements for a new dense model

- `TextEmbedding.list_supported_models()`: neither stella nor any GTE-v1.5-family model present
  (only the old plain-BERT `thenlper/gte-large`).
- **Ad hoc route**: `TextEmbedding.add_custom_model(model=..., pooling=..., normalization=True,
  sources=ModelSource(hf=...), dim=..., model_file="onnx/model.onnx")` — requires an ONNX file on
  the HF repo or a local path; fastembed's loader is an onnxruntime.InferenceSession wrapper, so
  ONNX is mandatory on this path.
- **Official-list route** (CONTRIBUTING.md): PR with canonical reference vectors (Colab notebook
  provided; pattern of merged PR #129).

## 4. Sparse/BM25 support and non-transformer precedent

- **BM25 supported**: `Qdrant/bm25` via SparseTextEmbedding — a **pure Python/NumPy/mmh3
  implementation with `model_file: "mock.file"`, no ONNX graph at all**, registered as a bespoke
  class. This is the precedent for the table query encoder.
- The Gather+pool+normalize query encoder has two paths: (a) a genuinely tiny ONNX graph
  (Gather → ReduceSum → L2 norm — all standard ops, no attention, no trust_remote_code; buildable
  with onnx.helper or torch.onnx.export on a 3-line nn.Module) registered via
  add_custom_model/supported-list PR; or (b) a bespoke numpy class mirroring BM25's precedent.
  No architectural obstacle on either path.
- No other non-transformer dense model (model2vec/potion/static) found in fastembed's list.

## 5. Tooling and known blockers

- Standard: `optimum-cli export onnx --task feature-extraction` or direct torch.onnx.export on a
  trust_remote_code-loaded model. No first-class optimum `NewModelOnnxConfig` found — support
  rides on generic tracing + Xenova tooling.
- Confirmed blockers/fixes for this family:
  1. `use_memory_efficient_attention: true` → xformers CUDA kernel, no ONNX equivalent → **set
     False before export** (stella's default is the bad one; gte ships False and exports cleanly).
  2. `unpad_inputs: true` → data-dependent unpadding → set False.
  3. Naive export without the fixes → `c10::SymInt` JIT-tracer error.
  4. Output-head naming (`sentence_embedding` vs `last_hidden_state`) — wrapper/remap fix.
  5. Quantized ONNX variants need their own parity check; nobody publishes quantization-parity
     MTEB deltas — measure in-house.

## Difficulty verdict

**Days** for a working, self-verified stella_en_400M_v5 export: (1) re-export with the two flags
off, (2) build/verify the pooling+normalization wrapper matching stella's sentence-embedding head,
(3) in-house nDCG parity: ONNX-fp32 vs PyTorch-fp32 vs int8-ONNX on a dev component. Existence
proofs: gte-large-en-v1.5 (same class, exports first-party) and Xenova's own stella export. No
existing artifact is reusable as-is (unmerged/unvalidated). Wiring into fastembed's supported list
is a separate small low-risk task; the query-side table encoder has direct precedent (BM25 bespoke
class) and no obstacle.

(Source URLs in the agent transcript; key ones: NovaSearch/stella_en_400M_v5 discussions #3/#11/#12
and config.json; Alibaba-NLP/gte-large-en-v1.5 onnx/ tree and config.json; gte-base-en-v1.5
discussion #12; qdrant/fastembed sparse/bm25.py, CONTRIBUTING.md; optimum-onnx configuration docs;
optimum issue #555; magiccodingman/stella_en_400m_v5_onnx.)
