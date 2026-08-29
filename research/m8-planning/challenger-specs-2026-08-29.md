# Challenger Spec research — four candidate document encoders

Method: for every candidate, `huggingface_hub.hf_hub_download`/`HfApi.model_info` was used to pull
only small config/tokenizer/README files (never `.safetensors`/`.bin`) from the **main** branch,
then `transformers.AutoTokenizer.from_pretrained` was run locally (offline, no network) on the
downloaded files to get `len(tokenizer)` — never trusted from `config.json`'s `vocab_size` alone.
All local copies live under
`/tmp/claude-1000/-home-dylan-asymetric-dual-encoders/4b780c0d-9e69-4c4c-aec8-f957cdd22f3f/scratchpad/hf/`
(one subdir per repo, `/`→`__`). No file in `/home/dylan/asymetric-dual-encoders` was modified.

Cap under test: **233 MB** int8 (`vocab * dim` bytes) for the released query-side lookup table.

---

## 1. `ibm-granite/granite-embedding-english-r2`

- **Repo / revision**: `ibm-granite/granite-embedding-english-r2`, main-branch commit
  **`47ea694b257b703fee9253d75c2b1f2985180498`** (`HfApi.model_info(...).sha`).
- **Architecture**: `ModernBertModel` (config.json `architectures`). Natively supported by the
  installed transformers **4.57.6** (`'modernbert' in CONFIG_MAPPING` → `True`) — no `auto_map`
  field in `config.json`, so **no `trust_remote_code`** needed for model or tokenizer.
- **Dim**: **768**. `1_Pooling/config.json`: `"word_embedding_dimension": 768"`. `modules.json`
  lists only `Transformer` (idx0) → `Pooling` (idx1, path `1_Pooling`) — **no Dense module**, so
  `post_dense = None`.
- **Pooling**: **cls**. `1_Pooling/config.json`: `"pooling_mode_cls_token": true`, all other
  pooling-mode flags false. Confirmed by the README's raw-transformers snippet:
  `query_embeddings = model_output[0][:, 0]` (line ~122 of the card).
- **Query prefix**: **none**. The card's `sentence-transformers` usage example calls
  `model.encode(input_queries)` on raw query strings with no prompt/prefix argument, and there is
  no `config_sentence_transformers.json`/`prompts` entry in this repo's file list (none of the
  downloaded files declare a query prompt). `doc_prefix` likewise none.
- **Tokenizer**: `tokenizer_class` in `tokenizer_config.json` = `"PreTrainedTokenizer"` (BPE,
  ModernBERT/`answerdotai/ModernBERT-base` vocabulary). `config.json` reports `"vocab_size": 50368`.
  Loaded locally: `AutoTokenizer.from_pretrained(...)` → **`len(tokenizer) == 50368`** — **matches
  config.json exactly, no padding gap** (unlike the registry's existing BERT-family models, this
  one has zero dead rows).
- **Sequence-start token id**: **50281** (`[CLS]`). `config.json`: `"cls_token_id": 50281`,
  `"bos_token_id": 50281`; loader-confirmed `tok.cls_token_id == 50281`.
- **`trust_remote_code`**: **False**. No `auto_map` anywhere in `config.json`/`tokenizer_config.json`.
- **Optional-dep overrides**: **none needed.** Inspected the installed
  `transformers/models/modernbert/modeling_modernbert.py`: it only *optionally* imports
  `flash_attn` (`if is_flash_attn_2_available(): from flash_attn... `) and falls back to
  sdpa/eager automatically — no xformers dependency at all, no config flags need overriding.
- **Licence**: `apache-2.0` (YAML front matter `license: apache-2.0`; README body: `**License:**
  [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)`).
- **Retrieval quality**: **BEIR Retrieval (15) = 53.1** nDCG@10-avg, from the model's own README
  table (`granite-embedding-english-r2` row, "BEIR Retrieval (15)" column, 149M params, 768-dim).
  Paper: arXiv 2508.21085.

**int8 table size**: 50368 × 768 = 38,682,624 bytes = **38.68 MB**. Well under the 233 MB cap
(fp16 would be 77.4 MB, also under).

---

## 2. `Alibaba-NLP/gte-modernbert-base`

- **Repo / revision**: `Alibaba-NLP/gte-modernbert-base`, main-branch commit
  **`e7f32e3c00f91d699e8c43b53106206bcc72bb22`**.
- **Architecture**: `ModernBertModel` (same as granite; base model per its card is
  `answerdotai/ModernBERT-base`). Native, no `auto_map`, **no `trust_remote_code`**.
- **Dim**: **768**. `1_Pooling/config.json` `word_embedding_dimension: 768`. `modules.json`:
  `Transformer` → `Pooling` (`1_Pooling`) only, **no Dense module** → `post_dense = None`.
- **Pooling**: **cls**. `1_Pooling/config.json`: `"pooling_mode_cls_token": true` (also declares
  `"include_prompt": true`, but there is no prompt configured — see below).
- **Query prefix**: **none**. `config_sentence_transformers.json`: `"prompts": {}`,
  `"default_prompt_name": null` — confirmed empty, and its README documents no prefix convention
  (same as its siblings `gte-large-en-v1.5`/`gte-base-en-v1.5`, already in `REGISTRY` with
  `query_prefix=""`).
- **Tokenizer**: byte-identical `tokenizer.json` to granite's (`md5sum` match, both derive from
  `answerdotai/ModernBERT-base`); `tokenizer_class` = `"PreTrainedTokenizerFast"` here (granite's
  says `"PreTrainedTokenizer"` — cosmetic difference only, same underlying vocab file).
  `config.json` `"vocab_size": 50368`; loaded locally: **`len(tokenizer) == 50368`** — again
  **matches config.json exactly**, no padding gap.
- **Sequence-start token id**: **50281** (`[CLS]`), same as granite (`config.json`
  `cls_token_id: 50281`, confirmed by loader).
- **`trust_remote_code`**: **False**.
- **Optional-dep overrides**: **none needed**, same reasoning as granite (native ModernBERT path,
  flash_attn optional/fallback, no xformers).
- **Licence**: `apache-2.0` (YAML front matter).
- **Retrieval quality**: **BEIR(15) = 55.33** nDCG@10-avg, from the model's own README "BEIR(NDCG@10)"
  summary table (`Average(15)` column for `gte-modernbert-base`). Per-dataset breakdown also in
  the card (ArguAna 72.68, FiQA2018 48.81, SciFact 77.4, SCIDOCS 21.29, NFCorpus 40.9, etc — the 5
  overlapping with our BEIR subset are directly readable from that table if needed later).

**int8 table size**: 50368 × 768 = 38,682,624 bytes = **38.68 MB**. Well under cap.

---

## 3. `NovaSearch/stella_en_1.5B_v5`

- **Repo / revision**: `NovaSearch/stella_en_1.5B_v5`, main-branch commit
  **`7817065102fd9e1b031fe874e910c01f40b2f001`**.
- **Architecture**: `Qwen2ForCausalLM` base, but the sentence-embedding pipeline loads it through
  `config.json`'s `"auto_map": {"AutoModel": "modeling_qwen.Qwen2Model", ...}`, i.e. a **custom**
  bidirectional-attention `Qwen2Model` shipped as `modeling_qwen.py` **in this same repo**
  (confirmed present in the repo's file listing at this revision) — **same-repo remote code, so
  pinning `revision` does pin it.**
- **Dim**: **1024** via the published pipeline. `modules.json` wires
  `Transformer(1536-dim) → Pooling(mean) → Dense("2_Dense_1024")`, and `2_Dense_1024/config.json`:
  `{"in_features": 1536, "out_features": 1024, "bias": true, "activation_function":
  "torch.nn.modules.linear.Identity"}`. (The repo also ships 256/768/2048/4096/6144/8192-dim Dense
  heads as MRL alternates, not wired by default `modules.json`.)
- **Pooling**: **mean**. `1_Pooling/config.json`: `"pooling_mode_mean_tokens": true`, all others
  false, `word_embedding_dimension: 1536` (pre-Dense).
- **`post_dense`**: **`"2_Dense_1024"`** — required, exactly the pattern already flagged in the
  `Spec` field comment for `stella-400M-v5` (identity activation, bias=true, 1536→1024). Omitting
  it would encode a different (1536-dim, un-projected) model.
- **Query prefix** (verbatim, from `config_sentence_transformers.json`'s `prompts.s2p_query`,
  cross-checked against the README's usage example — identical string):
  `"Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "`
  — **byte-identical to the `query_prefix` already registered for `stella-400M-v5`.** `doc_prefix`
  is empty (README: "docs do not need any prompts").
- **Tokenizer**: Qwen2 BPE (`vocab.json` + `merges.txt` + `added_tokens.json`), `tokenizer_config.json`
  declares `"tokenizer_class": "Qwen2Tokenizer"` with its own `auto_map` to
  `tokenization_qwen.Qwen2Tokenizer`/`Qwen2TokenizerFast` (same-repo file, also pinned by revision).
  `config.json` `"vocab_size": 151646`. Loaded locally with `trust_remote_code=True`:
  **`len(tokenizer) == 151646`** — **matches config.json exactly, no padding gap.**
- **Sequence-start token**: **there is no CLS/BOS-equivalent token.** `config.json` declares
  `"bos_token_id": 151643` (same value as `eos_token_id`), but the tokenizer's own
  `tokenizer_config.json` sets `"bos_token": null`, and the loaded tokenizer object reports
  `bos_token_id == None`. `eos_token_id = pad_token_id = 151643` (`<|endoftext|>`). **This is a
  genuine discrepancy between the model config and the tokenizer config** — flag it rather than
  picking one silently. Since pooling is mean (not CLS-readout), the existing `cls_id=101`
  fallback-row convention doesn't map onto this tokenizer family at all; a degenerate-empty-query
  row would need its own definition (e.g. the eos/pad id 151643, or an all-zero row) — this needs
  a design decision, not a default.
- **`trust_remote_code`**: **True**, required for both model and tokenizer, **same-repo** in both
  cases (`modeling_qwen.py`, `tokenization_qwen.py` both committed at the pinned revision).
- **Optional-dep overrides**: **none needed** — inspected `modeling_qwen.py` directly: its only
  `flash_attn` import is commented out (lines 49–51), no `xformers` reference anywhere in the file
  (`grep -i xformers` → no match), and the base `Qwen2Model` class sets `_supports_sdpa = True`
  and defaults `is_causal=False` in its forward path (line ~970) — i.e. sdpa bidirectional
  attention works out of the box, unlike `stella-400M-v5`'s NewModel architecture which hard-requires
  the `unpad_inputs`/`use_memory_efficient_attention` config overrides already in the registry.
- **Licence**: `mit` (YAML front matter `license: mit`, line 5428 of the README).
- **Retrieval quality**: **MTEB-Retrieval/BEIR-15 average = 61.006**, computed directly from the
  model's own `model-index` YAML in the README (parsed with `yaml.safe_load`, averaged the 15
  entries whose `task.type == "Retrieval"`, using each dataset's `main_score`). Matches the
  widely-cited 61.01 figure. **Contamination note**: 2 of the 15 self-reported retrieval datasets
  are ArguAna (65.269) and FiQA2018 (60.48) — 2 of our 6 eval datasets, same overlap already
  flagged for `stella-400M-v5` in `REGISTRY`.

**int8 table size**: 151646 × 1024 = 155,285,504 bytes = **155.29 MB**. Under the 233 MB cap
(margin ≈ 78 MB), but **fp16 would be 310.6 MB — busts the cap**, so this candidate only survives
at int8.

---

## 4. `harrier-oss-v1-0.6b` → found as **`microsoft/harrier-oss-v1-0.6b`**

I was not certain of the org prefix from the name alone; searched the HF Hub (`api.list_models`)
and found the canonical model at `microsoft/harrier-oss-v1-0.6b` (created 2026-03-30, i.e. after
my knowledge cutoff — consistent with not recognizing it), with everything else in that search
result being third-party GGUF/ONNX/MLX re-uploads of it. Cross-checked with a live web search
(Microsoft Tech Community blog, MarkTechPost) confirming this is a real, current Microsoft release,
part of a 270M/0.6B/27B family. Treating `microsoft/harrier-oss-v1-0.6b` as the intended model —
**flagging this identification explicitly rather than silently assuming it**, per your instruction.

- **Repo / revision**: `microsoft/harrier-oss-v1-0.6b`, main-branch commit
  **`f9b9dc8d367d443f2479d27aa5d8d2850c0774ee`**.
- **Architecture**: `Qwen3Model` (config.json `architectures`), **native** to transformers 4.57.6
  (`'qwen3' in CONFIG_MAPPING` → `True`; `AutoConfig.from_pretrained(...)` resolves to
  `transformers.models.qwen3.configuration_qwen3.Qwen3Config` with zero remote code). No `auto_map`
  anywhere in `config.json` or `tokenizer_config.json`.
- **Dim**: **1024**. `1_Pooling/config.json`: `"word_embedding_dimension": 1024"`, matches
  `config.json` `hidden_size: 1024`. `modules.json`: `Transformer` → `Pooling` → `Normalize`
  (idx 2, `sentence_transformers.models.Normalize`) — **no Dense module**, `post_dense = None`.
- **Pooling**: **NEITHER cls NOR mean — last-token pooling.** `1_Pooling/config.json`:
  `"pooling_mode_lasttoken": true`, all of `pooling_mode_cls_token`/`mean_tokens`/etc. false.
  Confirmed explicitly by the model card's FAQ ("**3. What pooling strategy does this model use?**
  The model uses **last-token pooling** — the embedding of the last non-padding token is used as
  the sentence representation.") and by the raw-transformers `last_token_pool()` snippet in the
  README.
  **THIS DOES NOT FIT THE CURRENT `Spec.pooling` FIELD.** `encoders.py`'s docstring restricts
  `pooling` to `"cls" | "mean"`, and I checked the actual pooling implementation
  (`m7src/teacher.py:101-121`, `pool_project_normalize`) — it has exactly two branches
  (`if pooling == "cls": ... elif pooling == "mean": ... else: raise ValueError(f"unknown pooling
  {pooling!r}")`). **Registering this model with `pooling="last"` would type-check but would crash
  at encode time** (or silently do nothing useful) until `pool_project_normalize` gains a
  last-token-pool branch (attention-mask-aware: last hidden state at
  `attention_mask.sum(dim=1) - 1`, as the README's `last_token_pool()` implements). This is a real
  code change, not just a new `Spec` entry.
- **Query prefix**: this model uses **task-specific instructions**, not a single fixed prefix.
  `config_sentence_transformers.json` declares three named prompts:
  - `web_search_query`: `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "`
    (this is the one to use for retrieval; **verbatim, no trailing period after "query"** — note
    this differs subtly from stella's phrasing, which ends "...answer the query.\nQuery: " with a
    period)
  - `sts_query`: `"Instruct: Retrieve semantically similar text\nQuery: "`
  - `bitext_query`: `"Instruct: Retrieve parallel sentences\nQuery: "`
  `doc_prefix`: none — card FAQ #1: "there is no need to add instructions to the document side."
- **Tokenizer**: native `Qwen2Tokenizer`/`Qwen2TokenizerFast` (BPE, `vocab.json`+`merges.txt`), no
  `auto_map`. `config.json` reports `"vocab_size": 151936`. Loaded locally (no
  `trust_remote_code`): **`len(tokenizer) == 151669`** — **does NOT match config.json**: config's
  151936 is padded, true tokenizer size is smaller by **267 tokens** (151936 − 151669 = 267 dead
  rows if the padded figure were used) — this is exactly the `vocab` pitfall the `Spec` field
  comment warns about, just a bigger gap than the 6-row gte-large/stella-400M case.
  (Aside: loading threw a benign warning — "incorrect regex pattern... set `fix_mistral_regex=True`"
  — a known generic transformers warning for this tokenizer-regex shape, not model-specific; noting
  it for completeness, not treating it as blocking.)
- **Sequence-start token**: **none exists, and the model doesn't need one** — pooling reads the
  *last* token, not the first, and `tokenizer_config.json` sets `"bos_token": null` (no BOS at
  all). `eos_token_id = 151645` (`<|im_end|>`), `pad_token_id = 151643` (`<|endoftext|>`). **The
  entire "sequence-start token for a degenerate-empty-query fallback" concept doesn't transfer to
  this model** — it's a structural mismatch with the CLS-pooled-BERT assumption baked into
  `table.py`'s `CLS_ID=101` convention, worse than stella's case (stella at least still starts a
  bag with something at position 0; harrier's whole embedding lives at the *end* of the sequence,
  so a "start token" fallback row is close to meaningless for it as designed).
- **`trust_remote_code`**: **False** for both model and tokenizer (native Qwen3Model, native
  Qwen2Tokenizer, no `auto_map` present anywhere in this repo's config files).
- **Optional-dep overrides**: none apparent — no custom remote code exists to need them (native
  architecture); could not fully verify absence of any flash-attn/xformers issue at actual load
  time without downloading the 1.19 GB `model.safetensors`, which the task rules out. Flagging as
  **not fully verified** rather than asserting confidently.
- **Licence**: `mit` (YAML front matter `license: mit`).
- **Retrieval quality**: **could not establish a retrieval-only/BEIR number.** The model card
  publishes only an **overall Multilingual MTEB v2 Score of 69.0** for the 0.6B variant (mixed
  task types averaged together: retrieval, clustering, classification, STS, bitext mining,
  reranking — not retrieval alone), in a comparison table against the 270M (66.5) and 27B (74.3)
  siblings. Checked `HfApi.model_info(...).card_data` for a `model-index`/`eval_results` block
  (none — unlike stella, this card carries no per-task YAML results) and ran a live web search
  specifically for a BEIR/retrieval-only number — none is published anywhere found. **This is a
  genuine gap, listed below, not filled with an inferred number.**

**int8 table size**: 151669 × 1024 = 155,308,576 bytes = **155.31 MB**. Under the 233 MB cap
(margin ≈ 78 MB) — but **fp16 would be 310.6 MB, busting the cap**, same situation as
stella_en_1.5B_v5.

---

## Summary table

| Repo | Revision (main HEAD) | Dim | Vocab (true, `len(tokenizer)`) | int8 MB (`vocab×dim`) | Over 233 MB? | `trust_remote_code` | Licence |
|---|---|---|---|---|---|---|---|
| `ibm-granite/granite-embedding-english-r2` | `47ea694b257b703fee9253d75c2b1f2985180498` | 768 | 50,368 (= config's 50368, no gap) | 38.68 MB | No | False | Apache-2.0 |
| `Alibaba-NLP/gte-modernbert-base` | `e7f32e3c00f91d699e8c43b53106206bcc72bb22` | 768 | 50,368 (= config's 50368, no gap) | 38.68 MB | No | False | Apache-2.0 |
| `NovaSearch/stella_en_1.5B_v5` | `7817065102fd9e1b031fe874e910c01f40b2f001` | 1024 (via `2_Dense_1024`) | 151,646 (= config's 151646, no gap) | 155.29 MB | No (int8) / **Yes at fp16 (310.6 MB)** | True (same-repo `modeling_qwen.py`/`tokenization_qwen.py`) | MIT |
| `microsoft/harrier-oss-v1-0.6b` | `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee` | 1024 | 151,669 (config says 151936 — **267-row padding gap**) | 155.31 MB | No (int8) / **Yes at fp16 (310.6 MB)** | False (native Qwen3Model + native Qwen2Tokenizer) | MIT |

None of the four bust the 233 MB **int8** cap on arithmetic alone; the two Qwen-family models
(stella_en_1.5B_v5, harrier-oss-v1-0.6b) would bust it at **fp16** and only fit at int8, with
~78 MB of headroom each. The two ModernBERT-family models (granite, gte-modernbert-base) have
large headroom either way.

---

## What I could not establish

1. **harrier-oss-v1-0.6b's retrieval-only quality number.** Only a mixed-task overall MTEB v2
   score (69.0) is published by Microsoft; no BEIR/retrieval-only average exists in the model
   card, in `HfApi`'s card metadata, or in a web search. Would need either an independent probe or
   a wait for a task-level MTEB leaderboard breakdown.
2. **Whether `microsoft/harrier-oss-v1-0.6b` is definitely the model meant by "harrier-oss-v1-0.6b"
   in the brief.** It is the only plausible primary-source match on the Hub (everything else
   matching the name is a re-upload/quantization of this exact repo), and independent web sources
   corroborate it as a real, current Microsoft release, but I was not given a URL to confirm
   against, so flagging the identification rather than treating it as certain.
3. **Whether harrier's Qwen3Model architecture loads cleanly on this machine without any config
   override at actual weight-load time.** I did not download `model.safetensors` (1.19 GB, out of
   scope per the constraints), so I can state "no `auto_map`, no declared optional-dep flags" from
   the config files, but cannot rule out a runtime surprise (e.g. an attention-implementation
   default that prefers `flash_attention_2`) the way I could for the other three by reading actual
   modeling code.
4. **Last-token pooling is not implemented anywhere in this repo.** `m7src/teacher.py`'s
   `pool_project_normalize` only has `cls`/`mean` branches and raises on anything else — adding
   harrier requires a real code change (an attention-mask-aware last-token branch), not just a new
   `Spec`. Flagging this as a blocker to registering harrier, not a detail to be quiet about.
5. **A principled `cls_id`/degenerate-empty-query-row design for both Qwen-family candidates**
   (stella_en_1.5B_v5 and harrier-oss-v1-0.6b). Neither has a true BOS/CLS analogue the way BERT's
   101 does; stella's `config.json` and `tokenizer_config.json` even disagree with each other about
   whether a BOS token id exists at all (151643 vs `null`). This needs a decision from whoever
   designs the fallback row, not a default I should silently pick.
