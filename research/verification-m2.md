# M2 adversarial verification

Reviewer pass over `bench/*.py`, `results/quality.json`, and the claims in `CLAUDE.md`. Read-only: no model inference, no MPS, no bench scripts. Everything marked "verified" was recomputed here with numpy / pytrec_eval on CPU from the cached artifacts, or read out of the installed packages and the HF cache.

Environment actually installed (`.venv`, Python 3.12.13): `transformers 5.15.1`, `sentence-transformers 6.0.0`, `torch 2.13.0`, `datasets 5.0.1`, `pytrec-eval-terrier 0.5.10`, `numpy 2.5.2`, `peft 0.20.0`. **`qdrant-edge` is not installed**, `results/costs.json` does not exist, `table_websearch.npy` / `table_arguana.npy` / `table_scidocs.npy` do not exist, and `results/quality.json` contains no LightRetriever or OpenSearch rows. So M3/M4/M5 cannot run today, and every finding about those scripts is pre-emptive.

Reproduction of the M2 table: recomputing all 13 configs from the cached vectors with `core.evaluate` reproduces `results/quality.json` to 4 decimals. The retrieval and scoring core is doing what it says.

---

## BLOCKER

### B1. Two of the thirteen configs ran at reduced precision, and one of them is the #2 model

`bench/run_st.py:46` — `SentenceTransformer(hf_id, device=device)` passes no dtype. Under `transformers` 5.x the default is `dtype="auto"`, i.e. the checkpoint's own `config.json` dtype:

- `transformers/modeling_utils.py:3986` docstring: "dtype (`str` or `torch.dtype`, *optional*, defaults to `"auto"`)"
- `transformers/modeling_utils.py:4136-4137`: `if dtype is None: dtype = "auto"`

Verified by loading each model on CPU through the project's own venv:

| model | `config.json` dtype | dtype actually loaded | `max_seq_length` |
|---|---|---|---|
| `ibm-granite/granite-embedding-small-english-r2` | `bfloat16` | **torch.bfloat16** | 8192 |
| `thenlper/gte-small` | `float16` | **torch.float16** | 512 |
| `BAAI/bge-small-en-v1.5` | `float32` | torch.float32 | 512 |

Same result through `SentenceTransformer(..., device="cpu")`, so sentence-transformers does not override it. Every other baseline (`bge`, `e5`, `MiniLM`, `arctic-xs/s/m-v1.5`, `mdbr-leaf-ir`) has `float32` or no dtype in config and loads fp32. The statics are unaffected.

Consequence: granite-small-r2 — the model sitting 0.0014 behind the leader — was encoded in bf16 (8-bit mantissa) while its competitors were encoded in fp32. Its 5-dataset average is 0.4611 against an official MTEB 0.4638; official ordering puts granite (0.4638) **above** arctic-embed-m-v1.5 (0.4630), the harness puts it below. The reported #1/#2 ordering is an artifact of this.

Fix: `SentenceTransformer(hf_id, device=device, model_kwargs={"dtype": torch.float32})`, delete `artifacts/granite-small-r2/` and `artifacts/gte-small/`, re-encode both. Also record the loaded dtype per row (see M8/MI7) so this cannot recur silently. Same fix applies to `bench/measure_cost.py:59` and `bench/run_projection.py:40`.

### B2. The "matches official MTEB ≤0.001" claim is false

`CLAUDE.md:23` and `CLAUDE.md:47` claim "harness matches official MTEB ≤0.001 (one −0.006 outlier: granite ArguAna)" and "every overlapping number matches to ≤0.001 nDCG after the ArguAna fix".

Recomputed over all 58 overlapping cells in `results/mteb_official.json`:

- 8 cells exceed 0.001, not 1.
- mean |delta| 0.00056, and 13 cells exceed 0.0005.

| config | dataset | harness | official | delta |
|---|---|---|---|---|
| granite-small-r2 | arguana | 0.53818 | 0.54398 | −0.00580 |
| granite-small-r2 | fiqa | 0.40352 | 0.40808 | −0.00456 |
| granite-small-r2 | nfcorpus | 0.36742 | 0.37144 | −0.00402 |
| potion-retrieval-32M | fiqa | 0.19027 | 0.18761 | +0.00266 |
| arctic-embed-m-v1.5 | scifact | 0.71586 | 0.71838 | −0.00252 |
| granite-small-r2 | scifact | 0.75685 | 0.75494 | +0.00191 |
| granite-small-r2 | scidocs | 0.23931 | 0.24059 | −0.00128 |
| potion-base-8M | scifact | 0.50521 | 0.50640 | −0.00119 |

Granite misses on **all five** datasets, with mixed signs — the signature of a precision problem (B1), not a prefix or convention error. `potion-retrieval-32M` FiQA is the one that should worry you most: a static model is deterministic in fp32, so +0.0027 means something in the load or text path differs from the official run (candidate: the 38 empty FiQA docs, see MI6, or `model2vec.StaticModel` vs the sentence-transformers loader).

Fix: restate the claim as "50 of 58 overlapping cells within 0.001; granite-small-r2 off on all five (dtype, B1); potion-retrieval-32M FiQA +0.0027 unexplained". Re-run the granite/gte cells after B1 and re-check. Chase the potion FiQA cell before publishing any static-model number.

### B3. The headline orderings in `CLAUDE.md` are inside the noise floor

Paired bootstrap (B=10,000, queries resampled within each dataset, 5-dataset macro average) computed here from the cached vectors:

| comparison | delta | 95% CI | p |
|---|---|---|---|
| arctic-m-v1.5 − granite-small-r2 | +0.0014 | [−0.0062, +0.0089] | 0.72 |
| bge-small − leaf-ir-asym | +0.0010 | [−0.0063, +0.0083] | 0.79 |
| **leaf-ir-asym − mdbr-leaf-ir (symmetric)** | **+0.0017** | **[−0.0027, +0.0060]** | **0.45** |
| granite-small-r2 − leaf-ir-asym | +0.0085 | [+0.0012, +0.0160] | 0.026 |
| arctic-m-v1.5 − leaf-ir-asym | +0.0099 | [+0.0065, +0.0134] | <0.001 |
| leaf-ir-asym − arctic-embed-s | +0.0157 | [+0.0091, +0.0224] | <0.001 |

The resolution of this benchmark for two similar systems is **±0.007 nDCG on the 5-dataset macro average** (1.96 × SD of the paired bootstrap difference). A single model's absolute score carries ±0.013. Per dataset it is much worse: ±0.043 on SciFact (n=300), ±0.035 NFCorpus (323), ±0.029 FiQA (648), ±0.019 ArguAna (1406), ±0.014 SciDocs (1000).

So:

- "arctic-m-v1.5 > granite-small-r2" is a coin flip: 63.8% / 36.0% under the bootstrap, and B1 pushes it the other way.
- "bge-small > leaf-ir-asym" is not resolved.
- **"leaf-ir-asym beats leaf symmetric" is not resolved.** The published effect is +0.48 nDCG on BEIR-15; this subset resolves ±0.7. The benchmark structurally cannot confirm or refute the central asymmetric claim.
- "97.9% of its 109M teacher" has a CI of 97.1%–98.6%. Quote the range or drop the third digit.

Worse, the per-dataset asym-vs-symmetric deltas disagree with the LEAF paper's own signs on three of five (paper numbers from `research/landscape.md:81-82`):

| dataset | paper asym − sym | harness asym − sym | harness 95% CI |
|---|---|---|---|
| ArguAna | +0.6 | **−1.07** | [−1.91, −0.22] |
| FiQA | +2.9 | +2.48 | [+1.46, +3.53] |
| NFCorpus | +0.4 | **−0.46** | [−1.24, +0.24] |
| SciDocs | +0.6 | +0.57 | [+0.15, +0.97] |
| SciFact | +0.2 | **−0.66** | [−2.27, +0.87] |

The ArguAna reversal is statistically significant in the wrong direction. Either the long-query case genuinely breaks the teacher/student alignment (a real and interesting finding for edge query encoders) or the configuration differs from what MongoDB used. `bench/run_asym.py` does match the `mdbr-leaf-ir` model card exactly (leaf queries with `prompt_name="query"` = the BGE instruction, arctic-m-v1.5 documents with no prompt — verified against the cached README), but the published 54.03 comes from the separately packaged `MongoDB/mdbr-leaf-ir-asym` repo, which this benchmark never loads. Check that repo's modules against `mdbr-leaf-ir` + arctic-m before reporting the asymmetric number.

Fix: publish CIs next to every number, state the ±0.007 resolution once, and reduce the M2 finding to the three groupings the data actually supports (top cluster arctic-m / granite / bge / leaf-asym / leaf-sym / gte, then arctic-s, then MiniLM / arctic-xs / e5, then the statics).

---

## MAJOR

### M1. The 5-dataset subset reorders models relative to full BEIR

Eight of the benchmarked configs have published BEIR-15 averages (`mdbr-leaf-ir` model card, cross-checked in `research/landscape.md`). Correlation between this subset and BEIR-15 over those eight: **Spearman rho 0.548, Kendall tau 0.357.** Nine of 28 pairs invert.

| model | 5-ds subset | BEIR-15 | rank here | rank BEIR-15 |
|---|---|---|---|---|
| leaf-ir-asym | 45.26 | 54.03 | 3 | 1 |
| mdbr-leaf-ir | 45.08 | 53.55 | 4 | 2 |
| arctic-embed-s | 43.69 | 51.98 | 5 | 3 |
| bge-small-en-v1.5 | 45.36 | 51.65 | 2 | 4 |
| granite-small-r2 | 46.11 | 50.87 | 1 | 5 |
| arctic-embed-xs | 40.06 | 50.15 | 7 | 6 |
| e5-small-v2 | 39.64 | 49.04 | 8 | 7 |
| all-MiniLM-L6-v2 | 40.96 | 41.95 | 6 | 8 |

MiniLM is last on BEIR-15 by 7 points and sixth of eight here, ahead of two models that beat it by 7–8 points on the full suite. The subset's top model is BEIR-15's fifth. Part of this is definitional (the subset is 5 of the 15), which is exactly the point: this subset does not rank near-neighbours the way full BEIR does, so no claim of the form "X beats Y" should leave the repo without "on SciFact/NFCorpus/FiQA/ArguAna/SciDocs" attached.

Fix: state the correlation in the deliverable, and add TREC-COVID (171K docs, 50 queries — cheap on the query side) plus one general-QA set. If corpus size is the blocker, sample MS MARCO to 100K rather than skipping the whole general-web family.

### M2. The subset is skewed against the thing being measured

Three of five datasets are scientific text (SciFact, NFCorpus, SciDocs); the general-web/QA family (MS MARCO, NQ, HotpotQA, DBPedia, Quora) is entirely absent; the largest corpus is 57.6K docs. LightRetriever's own retention table (`research/lightretriever.md:359-368`) shows the zero-query-compute design at **115% of the full model on TREC-COVID and 107% on argument retrieval**, and at **87% on SciDocs and 87% on entity retrieval**. Two of its three best categories are excluded, and its single worst (Citation Prediction / SciDocs) is included. A skeptic will read the subset as chosen to make the lookup-table approach look bad, and a supporter will read the ArguAna inclusion as making it look good. Neither reading is available if the report does not say which way each dataset cuts.

Also: no BM25 baseline anywhere in the benchmark, despite `CLAUDE.md:35` naming it as a candidate route and LightRetriever's own table putting BM25 at 41.7 BEIR — above every static model measured here (34.1) and above three of the small transformers on the subset. BM25 is zero-query-compute, Qdrant-native, and free to run. Its absence is the largest hole in the "what does zero query compute cost you" frontier.

Fix: add BM25 (`rank_bm25` or a 30-line numpy TF-IDF over the same corpora, using the same `topk_run`), and add a per-dataset "which side does this dataset favour" column to the report.

### M3. OpenSearch query IDF default is 1.0 where the model card uses 0

`bench/run_opensearch.py:81`: `qv[i, col_pos[t]] = idf.get(t, 1.0)`.

The model card's own reference code (read from the cached `README.md`) builds the IDF vector as:

```python
idf_vector = [0]*tokenizer.vocab_size
for token,weight in idf.items():
    _id = tokenizer._convert_token_to_id_with_added_voc(token)
    idf_vector[_id]=weight
```

Default weight for a token absent from `idf.json` is **0**, not 1.0. IDF values in that file are roughly 1–10, so a query token missing from the table currently contributes a mid-range weight instead of nothing. `idf.json` is not in the local HF cache so I could not count the misses, but the fix is unconditional.

Fix: `idf.get(t, 0.0)`, and print `len(cols) - sum(t in idf for t in cols)` per dataset so the size of the effect is on the record.

Everything else in that file does match the card exactly, verified line by line against the cached README: `output*attention_mask` masking (equivalent to relu-flooring, since padding becomes 0 and relu already floors at 0), the v3 double-log `torch.log(1 + torch.log(1 + torch.relu(values)))`, special-token column zeroing (equivalent here because `run_opensearch.py:46` removes special ids from the restricted column set), and unique-token binary × IDF on the query side.

### M4. LightRetriever hybrid fusion weights are undocumented and unverified

`bench/core.py:70` `fuse_linear(run_a, run_b, w_a=0.7, w_b=0.3)`, called from `run_lightretriever.py:191`. The file docstring claims "Conventions verified against github.com/caskcsg/lightretriever (see research/lightretriever.md)". `research/lightretriever.md:143` records only: "Hybrid = linear sum of L2-normalized dense and sparse scores (§3.1: 'The hybrid similarity scores are linearly summed from normalized dense and sparse scores')". Grep of the whole `research/` tree finds no mention of 0.7/0.3, of min-max, or of per-query normalization.

Two problems. First, the paper says *normalized* scores linearly summed, which reads as equal weights over L2- or z-normalized score vectors, not a 70/30 min-max blend. Second, if 0.7/0.3 was picked by looking at these five test sets, the hybrid number is fitted to the test data and cannot be compared with anything.

Fix: read `eval/`'s fusion code in the LightRetriever repo and record the exact scheme and weights in `research/lightretriever.md` with a file:line, the way the rest of that document is sourced. If the weights genuinely are a free parameter, report the sweep, or fix them at 0.5/0.5 and say so.

### M5. Per-task instruction tables are an advantage no baseline gets

`run_lightretriever.py:32-39` builds one lookup table per dataset instruction, so LightRetriever effectively gets five task-specific query encoders. Every baseline gets one fixed prompt (or none). `research/lightretriever.md:77` already flags this as "a real quality question the paper does not isolate", and `run_lightretriever.py:178` does provide the honest control (`-dense-websearch`, a single MS MARCO-instruction table, matching what one edge device would actually ship).

The failure mode is presentational: if `-dense` (per-task) is the number that lands in the frontier plot, the plot compares five encoders against one. The edge deployment ships one table.

Fix: make `-dense-websearch` the headline LightRetriever point, report `-dense` as a separate "oracle instruction" upper bound, and label both in the plot. Note that `table_websearch.npy` does not exist yet, so today only the advantaged variant is buildable.

### M6. `measure_cost.py` inflates baseline disk size by up to ~10×

`bench/measure_cost.py:50`: `snapshot_download(hf_id, allow_patterns=["*.safetensors", "*.json", "*.txt", "*model*"])`, then `dir_size_mb` sums everything downloaded.

`*model*` is an fnmatch over the full relative path, so for `sentence-transformers/all-MiniLM-L6-v2` it matches `model.safetensors`, `pytorch_model.bin`, `rust_model.ot`, `tf_model.h5`, nine `onnx/model*.onnx` variants, and three `openvino/openvino_model*` files — verified by running the pattern set against that repo's file listing. A 23M-parameter model whose single fp32 checkpoint is ~90 MB would be reported at close to 1 GB, and the script downloads all of it.

That number goes straight into the deliverable's cost frontier against the 466 MB LightRetriever table, i.e. the comparison the whole project turns on, and it biases it the wrong way.

Fix: `allow_patterns=["*.safetensors", "*.json", "*.txt", "vocab*", "merges*", "1_Pooling/*", "2_Dense/*", "3_Normalize/*"]` and drop `*model*` entirely, or better, report `sum(p.numel()*itemsize)` from the loaded model plus the tokenizer files. Whatever you pick, print the file list alongside the total.

### M7. The Edge prototype compares two different metrics and calls it ANN recall loss

`bench/edge_prototype.py:90` builds the ANN run as `{p.payload["doc_id"]: score}` with no self-hit filter, while `edge_prototype.py:105` builds the exact reference through `topk_run`, which does drop `doc_id == query_id`. FiQA has 55 ids that are both a query id and a doc id (verified). So `edge_metrics` and `exact_metrics` are not the same measurement, and any gap between them mixes ANN recall loss with a scoring convention difference.

Second issue in the same function: `lat_lookup` starts the clock after tokenization (`edge_prototype.py:77-78`), while `measure_cost.py:61` times `model.encode(QUERY)` for the baselines, which includes tokenization. The LightRetriever path is the one whose tokenization is a larger share of a tiny total, so this understates it.

Third: `shards_load_s` is measured in a process that runs right after `build` wrote those same files, so the OS page cache is warm. That is not a cold start, which is the specific claim the architecture rests on (`instructions.md:15`).

Fix: apply the same `!= qid` filter to the ANN run; move `t1` above the tokenizer call; and measure cold start after `purge` or from a freshly copied directory, and label whether the number is warm or cold.

### M8. Artifact cache keys omit everything that changes the vectors

`core.save_vecs` / `core.load_vecs` key on `(model_slug, dataset, kind)` only. `run_st.py:52-56` reuses `doc_vecs.npy` whenever it exists, regardless of the passage prefix in `REGISTRY`, the loaded dtype, `max_seq_length`, or the transformers version. After the B1 fix, a re-run will silently reuse the bf16 granite vectors unless the directory is deleted by hand.

`bench/reeval.py` compounds it: it can only refresh slugs that have *both* `doc_vecs.npy` and `query_vecs.npy`, so `leaf-ir-asym` (which has neither, it is composed at eval time) keeps whatever `run_asym.py` last wrote while every other row gets updated. A change to `core.topk_run` or `core.evaluate` would leave that one row stale and no warning anywhere.

Fix: write a `meta.json` next to each vector file with dtype, prefix, max_seq_length, and package versions; refuse to reuse a cache whose meta does not match. Add `leaf-ir-asym` (and any future composed config) to `reeval.py` via the `run_asym.PAIRS` table so one re-score pass covers everything.

### M9. LightRetriever's sparse doc encoder masks padding; the reference does not

`run_lightretriever.py:88`: `lg.masked_fill_(~mask.bool().unsqueeze(-1), -torch.inf)` before `amax(1)`. `research/lightretriever.md:138` records the reference as `torch.log1p(torch.relu(torch.amax(lm_out.logits, dim=1)))` — no mask. With right padding and causal attention, the reference's padded positions carry real continuation logits and do contribute to the max.

Masking is almost certainly the more defensible choice, but it is a deviation, and it means the sparse and hybrid numbers will not line up with the paper's 47.3 / 52.1 for reasons unrelated to anything else. It also silently makes the result batch-composition-dependent in the reference and batch-independent here, which is worth stating as a fix rather than hiding.

Fix: record the deviation in `research/lightretriever.md` next to the quote it contradicts, and run both variants once on SciFact to size it. If the gap is large, the paper's sparse numbers may simply not be reproducible without replicating their batching.

---

## MINOR

### MI1. Last-token pooling depends on an unset `padding_side`

`run_lightretriever.py:83`: `last = mask.sum(1) - 1` is only correct for right padding. The adapter repo's `tokenizer_config.json` has `"padding_side": null` (read from the cache), so this relies on `Qwen2Tokenizer`'s class default being `"right"`. The reference implementation (`research/lightretriever.md:121-127`) explicitly handles both sides. One tokenizer_config change upstream, or one `tok.padding_side = "left"` anywhere, silently pools a pad position.

Fix: `tok.padding_side = "right"` right after `AutoTokenizer.from_pretrained`, or copy the two-branch `lasttoken_pooling` from the reference.

### MI2. `fuse_linear` normalizes over a truncated candidate list

`core.py:70-80` min-max normalizes each run over its own top-1000, so "absent from the other run" and "ranked 1000th in the other run" both map to 0. A doc ranked #1 dense and missing from sparse scores 0.7; a doc ranked #1000 dense and #1 sparse scores 0.3. That is a defensible convention but it is a tuning knob disguised as arithmetic, and it interacts with M4.

Fix: state the convention in the report, or fuse over the union of both candidate sets with the global min as the floor.

### MI3. Sparse ties are broken arbitrarily

`core.topk_run` uses `np.argpartition`, which picks an arbitrary set among equal scores. On the sparse runs most of the corpus scores exactly 0, so which zeros land in the top-1000 is nondeterministic across numpy versions. It cannot touch nDCG@10 (zeros never outrank a term match) but it can perturb recall@100 on a query with fewer than 100 term matches, and it makes the run files non-reproducible.

Fix: add a deterministic tiebreak (`np.lexsort` on `(-score, doc_index)`) for the sparse path, or note that only nDCG@10 is trustworthy there.

### MI4. `idf.json` key-type detection is a 50-key heuristic

`run_opensearch.py:27`: `if all(k.isdigit() for k in list(idf)[:50])`. If the file's first 50 keys happen to be numeric tokens, every id gets reinterpreted. The string branch is also silent about misses: `tok.convert_tokens_to_ids(k)` returns `unk_token_id` for an unknown key, so many keys can collapse onto id 100 with last-write-wins. Harmless today (unk is a special id and excluded from `cols`) but it will not stay harmless.

`bench/measure_cost.py:99` uses only the string branch, so if the file is id-keyed that script's IDF dict is garbage. It affects nothing but a dict lookup timing, so it is cosmetic — but the two files should not disagree about the format of the same file.

Fix: assert the branch that fired and the number of keys that resolved, and import `load_idf` into `measure_cost.py` instead of re-deriving it.

### MI5. `stage_docs` skip check can leave a dataset half-cached

`run_lightretriever.py:129` skips on `doc_vecs.npy` alone, but `sparse_cols.json` is written later (line 135). A run interrupted between those two writes leaves a dataset that `stage_eval` cannot read, and the skip prevents recovery. `run_opensearch.py` has the mirror problem: `cols` is computed fresh at line 46 and then silently overwritten by the cached file at line 75, so a changed query set produces a self-consistent but wrong result with no warning.

Fix: skip on the presence of all outputs, and assert `cols == json.loads(...)` when a cache is reused.

### MI6. 38 FiQA documents are empty and become zero vectors for the statics

Verified: `(title + " " + text).strip()` is empty for 38 of 57,638 FiQA docs. Transformers still emit a CLS vector, but `potion-base-8M`, `potion-retrieval-32M`, and `static-retrieval-mrl-en-v1` all have exactly 38 zero-norm rows in their cached FiQA doc matrices. No NaN (verified across every artifact: no NaN rows anywhere, no zero rows outside these three), so nothing crashes, and 38/57,638 cannot explain the static FiQA collapse (~0.19 vs ~0.40). Still, a zero vector silently ties with every other zero vector.

Fix: log the count and either drop those docs from all systems or keep them and say so.

### MI7. `results/quality.json` carries no provenance

`core.record` writes `{ndcg@10, recall@100, n_queries}` and nothing else. Nothing in the file distinguishes a bf16 run from an fp32 one, records the prefix used, the max_seq_length, or the package versions — which is exactly why B1 survived to the M2 sign-off. `record()` already takes an `extra` dict; `run_projection.py:77` is the only caller that uses it.

Fix: pass dtype, prefix, max_seq_length, `transformers.__version__`, and the git SHA through `extra` on every call.

### MI8. Max-sequence-length asymmetry, largest exactly where it matters

Read from each repo's `sentence_bert_config.json`: MiniLM 256, bge/e5/gte/arctic-xs/arctic-s/arctic-m/leaf 512, granite-r2 **8192**, potion and static-retrieval-mrl unbounded (no truncation at all), LightRetriever 512 both sides, OpenSearch 512 (a harness choice; the card's default is the tokenizer's `model_max_length`).

Each value is the model's own configured limit, which is what MTEB uses too, so MTEB comparability holds. But ArguAna's queries average 193 words: the statics see the whole query, MiniLM sees roughly half of it, and the frontier plot puts them on the same axis. `research/methodology.md:92` already anticipates this for MiniLM and it is worth carrying into the report.

Fix: report the effective truncation per system in the results table, and report ArguAna with and without a common 256-token cap so the reader can see how much of the spread is truncation.

### MI9. `edge_prototype.build` only works for numeric doc ids

`edge_prototype.py:50`: `Point(id=int(doc_ids[i]))`. FiQA ids are numeric strings so it works; NFCorpus (`MED-10`) and SciDocs (hex sha) would raise. Fine for a FiQA-only prototype, but the file's docstring presents it as the architecture prototype, so the limitation should be visible.

Fix: `uuid.uuid5(NAMESPACE, doc_id)` with the original in the payload (which is already stored).

---

## NOTE — checked and clean, or a caveat to carry

**N1. Restricted-column sparse scoring is exact.** Refuting the concern directly: `amax` over the sequence axis is per-column, `relu` and `log1p` are elementwise, so slicing columns of the head weight commutes with the whole pipeline. Verified empirically on a synthetic replication (B=4, L=37, H=1536, V=151936, 2202 columns, right-padding mask): `log1p(relu(masked_amax(hidden @ W.T)))[:, cols]` and `log1p(relu(masked_amax(hidden @ W[cols].T)))` differ by **exactly 0.0**, bitwise, in both bf16 and fp32. Same for the resulting query dot products. The only residual risk is a different GEMM tiling on MPS for a `[K,H]` versus `[V,H]` weight; worth one 32-doc spot check, not worth worrying about.

Two real caveats though. It stops being exact the moment any column-coupling step is added — L2-normalizing the sparse vector, top-k term pruning, a global threshold, a FLOPs statistic. And it makes the artifact query-set-dependent: the doc sparse matrices in `artifacts/*/{ds}/doc_sparse.npy` cannot be reused for a different query set, cannot be used to report index size or nnz/doc, and cannot be shipped. Measured density over the restricted columns is 417/2317 on FiQA and 1850/19186 on ArguAna, which says nothing about the full-vocab sparsity a Qdrant sparse index would actually store. If the deliverable wants to compare inverted-index size, that needs a separate full-vocab pass over a few thousand docs.

**N2. fp16 vector storage is not a fairness problem.** Dense vectors are L2-normalized so components live in [−1, 1]; fp16 gives ~5e-4 relative per component and the dot-product error lands around 1e-4, two orders below the ±0.007 resolution. It is applied uniformly to every system by `core.save_vecs`, so it cannot favour one. The lookup tables are also safe: max |value| across the three built tables is 36.25 (fp16 overflows at 65504), zero inf, zero NaN, and cos(fp16-table mean, bf16-table mean) on a 12-token query is 0.9999998. The fp16 table cast is free.

**N3. The self-hit drop is correct and safe.** `core.topk_run:59` drops `doc_ids[i] != qid`, matching BEIR's `ignore_identical_ids`. Verified: ArguAna has 1298 of 1406 queries whose id is also a corpus doc id, FiQA has 55 — and in **neither** dataset does any qrels entry mark `doc_id == query_id` as relevant, so the drop never removes a judged positive. BEIR retrieves top_k+1 and filters, this filters after top-1000, which is indistinguishable at nDCG@10 and recall@100.

**N4. Data loading and query counts are correct.** Verified against BEIR: 300 / 323 / 648 / 1406 / 1000 test queries, matching `n_queries` in `quality.json`; no qrels query id missing from the queries file; no duplicate doc ids; SciDocs' 25,000 explicit level-0 judgments are handled correctly by pytrec_eval (treated as non-relevant, queries still counted); no query has zero positives. The `(title + " " + text).strip()` join matches the MTEB source quoted in `research/methodology.md:47`.

**N5. Prefix conventions are empirically validated.** Ten of the twelve configs with official numbers land within 0.0005 on the 5-dataset average (`mdbr-leaf-ir` and `arctic-embed-s` at 0.0000, `all-MiniLM-L6-v2` 0.0000, `static-retrieval-mrl` 0.0000, `potion-base-8M` 0.0000, `e5-small-v2` and `arctic-embed-xs` 0.0001, `bge-small` 0.0001, `arctic-m-v1.5` 0.0005, `gte-small` 0.0005). That is stronger evidence than reading model cards. Granite is the exception and it is B1.

**N6. LightRetriever table construction matches the repo.** No BOS for Qwen (the adapter tokenizer declares `bos_token: <|bos|>` with `add_bos_token: false`, and `research/lightretriever.md:48` records that the reference only prepends bos when `encode("")` yields one); `[prompt_ids] + [token] + [eos]` with `eos = <|endoftext|>`; EOS hidden state from `model.model` (post-final-norm, correct for the tied LM head); no attention mask, matching the reference; `V = len(tok) = 151666`, matching the built tables' shape. `queries_from_table` computes an unweighted mean over the raw id list, so repeated tokens count once per occurrence — the same semantics as `EmbeddingBag(mode='mean')`. The prompt string, including the trailing space in `"\nQuery: "`, is byte-identical to `cache_emb_bag.ipynb`. Per-task instructions match the paper's §A.7 table for all five datasets. `edge_prototype.py:83` correctly re-expands duplicate token ids after Qdrant's dedup on `retrieve`.

One property worth knowing: table row norms range from 15.5 to 129.5 across the vocabulary. Since the query vector is the plain mean of rows, large-norm rows dominate — the effective term weighting is baked into the row norms. That is the reference behaviour, and the per-row absmax int8 quantization at `run_lightretriever.py:174-175` preserves it, so the int8 size lever is measuring the right thing.

**N7. The doc-side cost is missing from the frontier, and it is large.** The frontier as specified in `CLAUDE.md:11` plots query-side cost only. Computed from the cached artifacts, the fp32 dense index for all 100,785 subset docs:

| system | dim | index (MB) |
|---|---|---|
| LightRetriever qwen2.5-1.5b | 1536 | 619.2 |
| arctic-embed-m-v1.5 / mdbr-leaf-ir | 768 | 309.6 |
| static-retrieval-mrl-en-v1 | 1024 | 412.8 |
| bge-small / arctic-embed-s | 384 | 154.8 |

The near-zero query encoder is paid for with a **4× larger document index** than bge-small, before any hybrid sparse index. For a Qdrant customer that is the recurring cost, and it is the number the current deliverable does not show. Add index bytes per document and indexing throughput (docs/s on the same machine) to the frontier, or the report answers only half the question it poses. The 1.5B doc encoder is also roughly 45× the FLOPs of a 33M one per document, which is a real cloud bill even under "unlimited indexing compute".

**N8. `record()` is not concurrency safe.** `core.py:98-102` is a read-modify-write of a single JSON file. Two runners in parallel lose one row silently. `CLAUDE.md:48` already mandates strictly sequential jobs for memory reasons, so this is latent, but it deserves a one-line comment at the function rather than living in the decision log.

---

## What this benchmark cannot tell Qdrant

Worth stating explicitly in the deliverable, because each of these will be the first question asked:

1. **Nothing about ANN behaviour per system.** All quality numbers are brute force by design (`CLAUDE.md:16`). Bag-of-token query vectors live in the convex hull of the lookup table's rows, a very different distribution from a transformer's output manifold, so HNSW recall at a fixed `ef` could differ systematically between the symmetric and lookup-table query sides. The Edge prototype measures this for exactly one configuration on one dataset, and M7 says that measurement is currently confounded. Until recall-at-fixed-ef is measured per system, the report cannot claim the quality ranking survives ANN.

2. **Nothing above 57.6K documents.** The whole subset is 100,785 docs. The claim in `instructions.md:12` is about an edge client backed by object storage, where the interesting regimes are shard size, segment count, and quantization at millions of vectors. A 4× dimension difference (N7) matters far more at 10M docs than at 100K.

3. **Nothing about filtered search, multitenancy, or payload-heavy collections** — where Qdrant actually differentiates and where a two-collection edge design would meet its real constraints.

4. **Nothing about reasoning, code, or multilingual retrieval.** LightRetriever loses to BM25 on BRIGHT (11.8 vs 14.5, `research/lightretriever.md:370`) and the paper concedes the point. Any recommendation must be scoped to short keyword-ish queries over English prose.

5. **Nothing about query compositionality**, which `instructions.md:44` names as the central weakness. Five nDCG@10 averages cannot separate "the bag of tokens lost word order" from "the model is smaller". The measurable version is BRIGHT, per `research/lightretriever.md:440`.

6. **Nothing about the operational cost of an instruction change.** One table per instruction (M5), at 444 MB and 10–25 minutes per build for the 1.5B backbone. A deployment that wants a second task ships a second table. That is a headline architectural constraint and it is currently a footnote.

7. **The static-model comparison is closer than the framing suggests.** `potion-retrieval-32M` at 0.3443 is also tokenize-lookup-average at query time. The only difference from LightRetriever's query side is that potion uses lookup on the document side too, which is precisely the paper's A1 ablation (−13.8 BEIR, `research/lightretriever.md:376-380`). So `potion-retrieval-32M` vs `lightretriever-dense-websearch` is the single most informative cell in the whole matrix — it isolates the value of the expensive document encoder at fixed query cost — and it does not exist yet.

---

## Suggested order of work

1. Fix B1 (one kwarg in three files), delete `artifacts/granite-small-r2/` and `artifacts/gte-small/`, re-encode, re-run `reeval.py`.
2. Rewrite the two claims in `CLAUDE.md` per B2 and B3, with the ±0.007 resolution and the bootstrap CIs.
3. Chase `potion-retrieval-32M` FiQA (+0.0027 on a deterministic model) before any static number ships.
4. Fix M3 (one default), M6 (one pattern list), M7 (three small edits) before M4/M5 produce numbers.
5. Source the fusion weights (M4) before running `stage_eval`, since `-hybrid` and `-hybrid-websearch` both depend on them.
6. Add BM25 and one large/general dataset (M1, M2). Both are cheap and both are the first thing a reviewer will ask for.
7. Add index bytes/doc and indexing throughput to the frontier (N7).
