# M11 code map — and the reusable ONNX-port checklist

Read `STATUS.md` first. This file is what a future session needs to (a) port nano (M13) and (b) **port a
different model to ONNX without rediscovering T3's traps**. Numbers live in `results/m11_*.json`;
the T3 narrative is `PLANNING.md` §T3. Nothing here restates them.

Known future consumers: nano's port (M13, blocked on M10) and M15's image model.

## Layout

| file | what | reusable? |
|---|---|---|
| `release/zero_encoder.py` | the shipped `zero` query path — numpy + tokenizers, no torch | model-specific |
| `release/export_onnx.py` | `zero`: hand-builds two graphs from a lookup table, 11 checks | pattern only |
| `release/export_doc.py` | stella doc tower: torch export + checks on frozen real fixtures | **template — copy it** |
| `release/push_doc.py` | build → 5 gates → create private → upload → verify → public | **template — copy it** |
| `release/push.py` | same for `zero`; 8 gates. Pre-existing repo, so no private-first ordering | template |
| `release/verify_bundle.py`, `verify_tokenizer.py` | `zero` gates 4 and 7 | verify_tokenizer generalises |
| `release/test_gates.py` | 14 checks proving `zero`'s gates refuse 13 planted breakages | pattern only |
| `release/doc_fixtures.json` | 259 real nq-250k passages, six length strata | regenerate per tokenizer |

A hand-built graph (`export_onnx.py`) and a torch export (`export_doc.py`) share nothing but the
gate structure. Do not try to unify them.

## The ONNX-port checklist

Every line cost something in T3 or T2. Evidence: `results/m11_doc_fp16_gpu.json`,
`results/m11_doc_export.json`, `PLANNING.md` §T2/§T3.

**Precision**

1. **Validate reduced precision on the execution provider it exists for.** ORT has no fast CPU fp16
   kernels: it up-converts, so a CPU gate measures fp32 arithmetic on fp16 weights. stella's fp16
   graph passed CPU parity at cos 0.99999923 and was wrong on **255/259 passages on CUDA**
   (min-cos 0.662). The same up-conversion is why fp16 is ~10x SLOWER on CPU — one cause, two
   symptoms, and testing only on CPU reads them backwards.
2. `convert_float_to_float16` casts an exported graph and **cannot control accumulation dtype**
   inside attention. If you need fp16, control dtype at export, not after.
3. Its `node_block_list` matches names **exactly** — derive the list from the graph (e.g. nodes
   after the last `LayerNormalization`), never hard-code it. A rename silently blocks nothing and
   you get an all-fp16 head plus a wasted multi-GB export.
4. `op_block_list` is by op TYPE and would block every MatMul in the backbone. Not a substitute.
5. Its output needs repair before it will load: duplicate node names sharing an output tensor
   (`repair()` drops the byte-identical clones) and appended casts breaking topological order
   (stable Kahn — the level-by-level form "works" but churns 3440 positions instead of 41).

**Correctness of the comparison itself**

6. **A dot product is not a cosine** unless both sides are unit-norm — which fp16 output is not
   (norms 0.9995–1.0004). Same graph: dot 0.99954 (fails 1e-4) vs true cosine 0.99999940 (passes).
   Always `dot / (‖a‖·‖b‖)`, and record the norm ranges so the assumption is visible.
7. **Assert exact state-dict keys**, never "the first rank-2 tensor". `2_Dense_1024` has a
   `linear.bias` that rank-selection drops, and the square matrix keeps a wrong or transposed key
   shape-valid — a torch reference built the same way certifies the error.
8. **Run `onnx.checker` if you claim you ran it.** `export_onnx.py` check 1 was labelled "checker
   passes" with `not custom` as its only predicate, so `zero`'s live gate asserted something it
   never tested. Assert the opset explicitly too.
9. **A dtype census proves what a filename claims.** A copied fp32 graph renamed `model_fp16.onnx`
   passes opset/domain/IO checks perfectly. Count initializers by dtype and name the exceptions.
10. **A present-but-failing graph must fail the run.** Scoring only the primary graph let
    `PASS  fp16 shippable: False` exit zero with the bad file still staged.

**Fixtures**

11. **Real text, length-stratified, frozen to a file, strata re-asserted on load.** Synthetic word
    salad (the M9 n=40) certifies nothing. Include the truncation boundary explicitly — 511/512/513
    — or the "we tested truncation" claim is vacuous. A replacement file of short passages must be
    refused, not trusted.
12. **Don't load a 1.6 GB corpus beside a torch model and a multi-GB ORT session.** Freeze the
    fixtures once; the push gate then needs no corpus at all.

**fastembed**

13. **`PoolingType.DISABLED` + `normalization=False` serves an already-pooled graph** — measured
    bit-identical to direct ORT (max-abs 0.00e+00). You do NOT need a per-token graph for a model
    whose head sits after pooling. Believing otherwise costs a duplicate of the whole model.
14. **`parallel>1` cannot work for `add_custom_model`**: the worker builds `OnnxTextEmbedding`,
    which cannot resolve a runtime-registered name. Register the model natively instead — an entry
    in `supported_onnx_models` costs ~13 lines and `parallel` then works. A card teaching
    `add_custom_model` also breaks the day the model ships natively (`already registered`).
15. **fastembed truncates at `min(model_max_length, max_length)` from the shipped files.** Ship the
    length your index was built at, or long documents silently fail to reproduce it. There is no
    API override (qdrant/fastembed#689). Ship `tokenizer.json` `padding: null` so fastembed
    installs its own dynamic padding.

**Publishing**

16. **New repo: `exist_ok=False`, id as a constant, private → upload → verify → public.** A
    free-form `--repo-id` is a typo-to-publication path. (`zero` could not use this: its repo was
    already public, spending the ordering guarantee — `STATUS.md`.)
17. **Verify against a hash snapshot captured after the last gate**, not against the staging dir as
    it stands at verification time, which compares changed bytes with themselves.
18. **Verify big files by LFS oid** — it IS the content sha256, so no multi-GB re-download.
19. **Execute the card's code against the staged bytes, offline**, and refuse anything that would
    reach the Hub instead: a surviving `snapshot_download`, a `TextEmbedding` built from a literal
    repo id, or any un-redirected `TextEmbedding(` call. Do NOT assert a substitution count — a
    card may legitimately have no block of a given kind, and every negative fixture is such a card.

**Serving and gate hygiene**

20. **Gate the SERVING path, not just the graph.** ONNX parity says the graph is right; it says
    nothing about what the library feeds it. Here the two agreed to the digit (4.47e-08), which is
    the result you want — but it is a measurement, not an assumption. (`verify_fastembed.py`)
21. **Parity fixtures must exceed the truncation limit.** A tokenizer rule only bites past it, so a
    fixture set of ordinary short queries passes under the WRONG rule. The negative control here
    (truncation 8000 restored) is invisible to 1,024 real dev queries and 44x the threshold on one
    long input.
22. **A fork carries the same `__version__` as the release it branched from.** Key result files on
    the import path, or a fork run silently overwrites the stock one and both look identical.
23. **The card gate needs the FastEmbed branch on `PYTHONPATH` and the sibling model in the HF
    cache**, because the cards name built-in models and one card embeds the other model. Both are
    box-local assumptions (`FASTEMBED_FORK` is a hard-coded path); on another machine gate 6 fails
    for reasons unrelated to the bundle.
24. **Silence a multiprocess probe at the fd level.** Worker processes write to the inherited fd,
    so `contextlib.redirect_stderr` does not reach them; a flood of worker tracebacks filled the
    pipe and deadlocked the run. `os.dup2` to `/dev/null` around the block.

## Gotchas specific to stella

- Crashes without `config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}`,
  and they must land on the **config** — transformers 4.57 forwards unknown `from_pretrained`
  kwargs to `__init__` and raises.
- Its card frontmatter declares `license: mit`; lineage `Alibaba-NLP/gte-large-en-v1.5` is
  Apache-2.0. Ship no Apache text unless you redistribute `modeling.py`.
- **Document-only.** Queries need the `s2p_query` prompt; fastembed's `query_embed` adds no prefix,
  so symmetric use is silently wrong.
