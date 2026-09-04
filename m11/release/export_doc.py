"""Export stella_en_400M_v5's DOCUMENT path to ONNX and verify it on REAL passages. M11a T3.

    .venv/bin/python m11/release/export_doc.py --make-fixtures   # once; writes doc_fixtures.json
    .venv/bin/python m11/release/export_doc.py --export --check  # build work/m11onnx/stella-doc
    .venv/bin/python m11/release/export_doc.py --check --onnx-dir DIR [--n N]

`m9src/export_doc_model.py` is left untouched as the historical record. It is not reused because
its evidence does not survive contact: parity ran on n=40 of word salad from a 20-word vocabulary
and never on real text; the fp16 graph was never in the comparison at all (`main()` builds its
session on the fp32 path); its `fastembed_local` block cannot have been written by the function it
names; and what it records as `min-cos` is a bare dot product, `(ref * got).sum(1)`, which is only
a cosine when BOTH sides are unit-norm -- precisely what an fp16 graph is not (Codex, 2026-09-03).
See `m11/PLANNING.md` §T3.

Two graphs, opset 17, both from ONE loaded torch module so they cannot disagree about weights:

  model.onnx        fp32, self-contained: masked mean -> 2_Dense_1024 -> L2.  (b, 1024)
  model_fp16.onnx   fp16 backbone, fp32 head. Same signature. Shipped only if it passes §11.4.

There is no `model_tokens.onnx`. The M9 note claiming fastembed "has no slot for a dense layer
after pooling" is wrong: `PoolingType.DISABLED` with `normalization=False` serves an
already-pooled graph unchanged (`fastembed/common/model_description.py`,
`custom_text_embedding.py`). Check 5 measures that route rather than trusting it. Believing the
M9 note would have cost a third 1.75 GB graph, its export, its upload and its parity row.
"""
import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
OUT = REPO / "work/m11onnx/stella-doc"
FIXTURES = SRC / "doc_fixtures.json"
RESULT = REPO / "results/m11_doc_export.json"

REPO_ID = "NovaSearch/stella_en_400M_v5"
REVISION = "ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20"
DENSE = "2_Dense_1024"
# NewModel asserts on xformers unless BOTH are off, and they must land on the CONFIG: transformers
# 4.57 forwards unknown kwargs from from_pretrained to __init__ and raises.
CONFIG_KWARGS = {"use_memory_efficient_attention": False, "unpad_inputs": False}
MAX_LEN = 512               # FREEZE.json:encoder_spec -- the length the document index was built at

GRAPHS = ("model.onnx", "model_fp16.onnx")
IO = {"inputs": ["input_ids", "attention_mask"], "outputs": ["embedding"]}

# Fixture strata, by RAW (untruncated) token count under the pinned tokenizer. `boundary` and
# `over` are the point: truncation at 512 is the rule the index was built with, so 511 must pass
# through whole and 513 must come back as 512.
STRATA = {"tiny": (1, 32), "short": (32, 128), "mid": (128, 384), "near": (384, 511),
          "boundary": (511, 514), "over": (514, 10 ** 9)}

# The dtype census proves what the filenames only claim: a copied fp32 graph renamed
# `model_fp16.onnx` passes opset/domain/IO checks perfectly. fp32 must be all-float; fp16 must be
# half everywhere EXCEPT the Dense parameters, which the block list keeps in fp32.
DTYPE_INVARIANT = {"model.onnx": {"float16": 0, "float32_min": 1},
                   "model_fp16.onnx": {"float16_min": 1, "float32_exact": 2}}

# Mandate §11.4: the tolerance a port must meet.
TOL_COS, TOL_ABS = 1e-4, 1e-3
# fp32 graph vs the torch module is the same arithmetic twice, so it is held far tighter.
TOL_COS_SELF, TOL_ABS_SELF = 1e-6, 1e-5

# The `2_Dense_1024` state dict, by exact key. Selecting "the first rank-2 tensor" is what the M9
# builder did; the matrix is SQUARE, so a wrong key or a transpose stays shape-valid and a torch
# reference built the same way would certify it (Codex, 2026-09-03).
DENSE_KEYS = {"linear.weight": (1024, 1024), "linear.bias": (1024,)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def snapshot():
    d = (Path.home() / ".cache/huggingface/hub/models--NovaSearch--stella_en_400M_v5/snapshots"
         / REVISION)
    if not d.exists():
        sys.exit(f"REFUSED: pinned revision not cached at {d}; the tokenizer and weights must "
                 "come from the pinned revision, not from whatever is on the Hub today")
    return d


def tokenizer(truncate=True):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(snapshot() / "tokenizer.json"))
    tk.no_padding()
    tk.no_truncation()
    if truncate:
        tk.enable_truncation(max_length=MAX_LEN)
    return tk


# ---------------------------------------------------------------- fixtures

def make_fixtures(n_per_stratum=48, seed=0):
    """Scan nq-250k ONCE and freeze a length-stratified set of REAL passages to a small JSON.

    Frozen rather than resampled so every recorded number is reproducible and the push gate needs
    no corpus at all. nq-250k only: hotpotqa.json is 1.6 GB (5.2M strings) and would have to sit
    in RAM beside the torch model and a 1.75 GB ORT session, and it contributes nothing above 445
    tokens anyway -- the truncation boundary exists in NQ or nowhere (Codex, 2026-09-03).
    """
    texts = json.loads((REPO / "work/dev/nq-250k.json").read_text())["doc_texts"]
    tk = tokenizer(truncate=False)
    r = random.Random(seed)
    idx = list(range(len(texts)))
    r.shuffle(idx)

    lens = {}
    for i in idx:
        lens[i] = len(tk.encode(texts[i]).ids)

    # The boundary stratum is the point of the exercise: truncation at 512 is the rule the index
    # was built with, so 511 must pass through whole and 513 must come back as 512.
    picked, out = set(), {}
    for name, (lo, hi) in STRATA.items():
        cand = [i for i in idx if lo <= lens[i] < hi and i not in picked]
        take = cand[:n_per_stratum]
        if not take:
            sys.exit(f"REFUSED: stratum {name} [{lo},{hi}) is empty; the stratification claim "
                     "would be vacuous")
        picked.update(take)
        out[name] = [texts[i] for i in take]
        print(f"  {name:9s} [{lo},{hi})  {len(take):3d} passages  "
              f"raw lengths {min(lens[i] for i in take)}-{max(lens[i] for i in take)}")

    FIXTURES.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  wrote {FIXTURES}  ({FIXTURES.stat().st_size/1e3:.0f} kB)")


def fixtures(n=None):
    """-> (texts, stratum labels). `n` takes a deterministic slice of each stratum for the gate.

    The strata are ASSERTED, not trusted. Without this a replacement file of 259 short passages,
    or six mislabeled strata, passes every downstream check while silently losing the
    boundary/truncation coverage the parity claim rests on (Codex, 2026-09-03).
    """
    if not FIXTURES.exists():
        sys.exit(f"REFUSED: {FIXTURES} missing; run --make-fixtures once")
    d = json.loads(FIXTURES.read_text())
    if set(d) != set(STRATA):
        sys.exit(f"REFUSED: {FIXTURES} strata are {sorted(d)}, expected {sorted(STRATA)}")
    tk = tokenizer(truncate=False)
    for name, (lo, hi) in STRATA.items():
        raw = [len(tk.encode(t).ids) for t in d[name]]
        if not raw:
            sys.exit(f"REFUSED: stratum {name} is empty")
        if not all(lo <= L < hi for L in raw):
            sys.exit(f"REFUSED: stratum {name} holds lengths {min(raw)}-{max(raw)}, "
                     f"outside its declared [{lo},{hi})")
    per = None if n is None else max(1, n // len(d))
    texts, labels = [], []
    for name, ts in d.items():
        ts = ts if per is None else ts[:per]
        texts += ts
        labels += [name] * len(ts)
    return texts, labels


def encode(texts, batch=8):
    """Tokenize at the FROZEN rule (truncate 512, pad to batch-longest) -> [(ids, mask), ...]."""
    import numpy as np
    tk = tokenizer()
    encs = [tk.encode(t) for t in texts]
    out = []
    for i in range(0, len(encs), batch):
        ch = encs[i:i + batch]
        L = max(len(e.ids) for e in ch)
        ids = np.zeros((len(ch), L), "int64")
        am = np.zeros((len(ch), L), "int64")
        for j, e in enumerate(ch):
            ids[j, :len(e.ids)] = e.ids
            am[j, :len(e.ids)] = 1
        out.append((ids, am))
    return out


# ---------------------------------------------------------------- the torch reference

def build():
    """-> (module, dim). Masked mean -> 2_Dense_1024 -> L2, the head the index was built with."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(REPO_ID, revision=REVISION, trust_remote_code=True)
    for k, v in CONFIG_KWARGS.items():
        setattr(cfg, k, v)
    backbone = AutoModel.from_pretrained(REPO_ID, revision=REVISION, config=cfg,
                                         trust_remote_code=True).eval()

    dcfg = json.loads(Path(hf_hub_download(REPO_ID, f"{DENSE}/config.json",
                                           revision=REVISION)).read_text())
    if not dcfg.get("activation_function", "").endswith("Identity"):
        sys.exit(f"REFUSED: {DENSE} activation is {dcfg.get('activation_function')!r}, not "
                 "Identity; the shipped head would not be the head the index was built with")
    sd = load_file(hf_hub_download(REPO_ID, f"{DENSE}/model.safetensors", revision=REVISION))
    got = {k: tuple(v.shape) for k, v in sd.items()}
    if got != DENSE_KEYS:
        sys.exit(f"REFUSED: {DENSE} state dict is {got}, expected {DENSE_KEYS}; selecting the "
                 "weight by rank would silently accept a different or transposed tensor")
    W, B = sd["linear.weight"], sd["linear.bias"]

    dense = torch.nn.Linear(W.shape[1], W.shape[0], bias=True)
    with torch.no_grad():
        dense.weight.copy_(W)
        dense.bias.copy_(B)

    class DocEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.b, self.dense = backbone, dense

        def forward(self, input_ids, attention_mask):
            h = self.b(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(self.dense(v), dim=-1, eps=1e-12)

    return DocEncoder().eval(), int(W.shape[0])


# ---------------------------------------------------------------- fp16

def head_nodes(graph):
    """Everything after the last LayerNormalization: mask broadcast, masked mean, Dense, L2.

    Derived, never hard-coded. Node names are assigned by the exporter, and
    `convert_float_to_float16` matches `node.name in node_block_list` EXACTLY -- a rename means
    nothing is blocked, the conversion silently reverts to an all-fp16 head, and the 1.75 GB
    export is wasted (Fable, 2026-09-03).
    """
    i = max(k for k, n in enumerate(graph.node) if n.op_type == "LayerNormalization")
    return [n.name for n in graph.node[i + 1:]]


def repair(graph):
    """`convert_float_to_float16` emits an UNLOADABLE graph here; both defects are mechanical.

    1. It inserts one cast-to-fp32 node per CONSUMER of a blocked node's input, all sharing a name
       AND an output tensor -> `two nodes with same node name`, two producers for one tensor. The
       clones are byte-identical, so keeping the first is exact.
    2. It APPENDS those casts, so a blocked node can consume a tensor produced later in the list
       -> `Nodes in a graph must be topologically sorted`.

    Neither is a precision change. Measured on this graph: 3 clones, 41 positions reordered.
    """
    seen, keep, dropped = {}, [], []
    for n in graph.node:
        b = n.SerializeToString()
        if n.name in seen:
            if seen[n.name] != b:
                sys.exit(f"REFUSED: two DIFFERENT nodes share the name {n.name!r}")
            dropped.append(n.name)
            continue
        seen[n.name] = b
        keep.append(n)

    # Stable Kahn: `ready` updates AS each node is emitted, so a node whose producer sits just
    # ahead of it in the same pass still goes out in place. Deferring that to the next pass
    # reorders the graph level-by-level instead -- 3440 positions moved rather than 41, for the
    # same valid result. Only the appended casts should move.
    ready = {i.name for i in graph.input} | {i.name for i in graph.initializer} | {""}
    ordered, pending = [], keep
    while pending:
        rest = []
        for n in pending:
            if all(x in ready for x in n.input):
                ordered.append(n)
                ready.update(n.output)
            else:
                rest.append(n)
        if len(rest) == len(pending):
            sys.exit(f"REFUSED: {len(rest)} nodes have unsatisfiable inputs, e.g. "
                     f"{rest[0].name} <- {list(rest[0].input)}")
        pending = rest
    moved = sum(1 for a, b in zip(keep, ordered) if a.name != b.name)
    del graph.node[:]
    graph.node.extend(ordered)
    return {"clone_nodes_dropped": sorted(set(dropped)), "positions_reordered": moved}


# ---------------------------------------------------------------- export

def export(out, fp16=False):
    """`fp16` is OFF by default and the graph it builds is NOT a release artifact.

    It passes CPU parity (cos 0.99999923 on all 259 fixtures) and is still unusable: ONNX Runtime
    has no fast CPU fp16 kernels, so it up-converts to fp32 -- which is why that number looks good
    AND why fp16 runs ~10x slower there. On CUDA, where it runs in real fp16, it disagrees with the
    fp32 reference on 255 of 259 passages (min-cos 0.662), at every length from 7 tokens up.
    `results/m11_doc_fp16_gpu.json`. Kept as an experimental build path so the next attempt starts
    from the measurement rather than repeating it.
    """
    import copy
    import onnx
    import torch
    from onnxruntime.transformers.float16 import convert_float_to_float16

    out.mkdir(parents=True, exist_ok=True)
    model, dim = build()
    texts, _ = fixtures()
    ids, am = encode(texts[:4], batch=4)[0]

    t = time.time()
    torch.onnx.export(model, (torch.from_numpy(ids), torch.from_numpy(am)),
                      str(out / "model.onnx"),
                      input_names=IO["inputs"], output_names=IO["outputs"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"},
                                    "attention_mask": {0: "b", 1: "s"},
                                    "embedding": {0: "b"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"  exported model.onnx ({time.time()-t:.0f}s)", flush=True)
    del model
    if not fp16:
        print("  fp16 NOT built (--fp16 to opt in; it is not a release artifact, see the "
              "docstring and results/m11_doc_fp16_gpu.json)", flush=True)
        return {"dim": dim, "fp16_built": False}

    g = onnx.load(str(out / "model.onnx"))
    blk = head_nodes(g.graph)
    m16 = convert_float_to_float16(copy.deepcopy(g), keep_io_types=True, node_block_list=blk)
    rep = repair(m16.graph)
    onnx.save(m16, str(out / "model_fp16.onnx"))
    onnx.checker.check_model(str(out / "model_fp16.onnx"), full_check=True)
    print(f"  converted model_fp16.onnx  head kept fp32 ({len(blk)} nodes), "
          f"{len(rep['clone_nodes_dropped'])} clones dropped, "
          f"{rep['positions_reordered']} reordered", flush=True)
    return {"dim": dim, "fp16_built": True, "fp16_blocked_nodes": blk, "fp16_repair": rep}


# ---------------------------------------------------------------- checks

def _sess(path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run(sess, batches):
    import numpy as np
    return np.concatenate([sess.run(None, {"input_ids": i, "attention_mask": m})[0]
                           for i, m in batches])


def _cos(ref, got):
    """Cosine, not a dot product. They differ exactly when a side is not unit-norm, which is the
    fp16 graph's situation (norms 0.9995-1.0004) -- so the metric choice flips the verdict."""
    import numpy as np
    na = np.linalg.norm(ref, axis=1)
    nb = np.linalg.norm(got, axis=1)
    return (ref * got).sum(1) / (na * nb), na, nb


def check_graphs(onnx_dir):
    """Static: the claim 'opset 17, standard ops only, self-contained' is asserted, not narrated.

    `onnx.checker` is actually RUN here. `export_onnx.py`'s check 1 labels itself 'checker passes'
    but only inspects domains (line 202), so `zero`'s gate 8 asserts a checker pass it never
    performs -- fixed separately (Codex, 2026-09-03).
    """
    import onnx
    res = {}
    for name in GRAPHS:
        p = Path(onnx_dir) / name
        if not p.exists():
            res[name] = {"present": False}
            continue
        onnx.checker.check_model(str(p), full_check=True)
        m = onnx.load(str(p), load_external_data=False)
        ops = [(o.domain, o.version) for o in m.opset_import]
        custom = sorted({n.domain for n in m.graph.node} - {"", "ai.onnx", "ai.onnx.ml"})
        ext = [i.name for i in m.graph.initializer if i.HasField("data_location")
               and i.data_location == onnx.TensorProto.EXTERNAL]
        io = {"inputs": [i.name for i in m.graph.input],
              "outputs": [o.name for o in m.graph.output]}

        n16 = sum(1 for i in m.graph.initializer if i.data_type == onnx.TensorProto.FLOAT16)
        n32 = sum(1 for i in m.graph.initializer if i.data_type == onnx.TensorProto.FLOAT)
        want = DTYPE_INVARIANT[name]
        dt_ok = (n16 == want["float16"] if "float16" in want else n16 >= want["float16_min"])
        dt_ok = dt_ok and (n32 == want["float32_exact"] if "float32_exact" in want
                           else n32 >= want["float32_min"])
        fp32_names = sorted(i.name for i in m.graph.initializer
                            if i.data_type == onnx.TensorProto.FLOAT)
        if name == "model_fp16.onnx" and dt_ok:
            dt_ok = fp32_names == ["dense.bias", "dense.weight"]

        ok = (ops == [("", 17)] and not custom and not ext and io == IO and dt_ok)
        res[name] = {"present": True, "opset": ops, "custom_domain_ops": custom,
                     "external_data_initializers": ext, "io": io, "n_nodes": len(m.graph.node),
                     "initializers": {"float16": n16, "float32": n32},
                     "float32_initializer_names": fp32_names if len(fp32_names) <= 4 else "many",
                     "dtype_invariant_ok": bool(dt_ok),
                     "bytes": p.stat().st_size, "sha256": sha256(p), "pass": bool(ok)}
        print(f"  {name:18s} opset {ops}  {len(m.graph.node)} nodes  "
              f"{p.stat().st_size/1e6:.0f} MB  -> {'PASS' if ok else 'FAIL'}", flush=True)
    return res


def check_parity(onnx_dir, n=None):
    import numpy as np
    import torch

    texts, labels = fixtures(n)
    batches = encode(texts, batch=8)
    lens = sorted(int(m[j].sum()) for _, m in batches for j in range(m.shape[0]))
    res = {"n_fixtures": len(texts), "strata": {s: labels.count(s) for s in set(labels)},
           "encoded_lengths": {"min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1],
                               "n_truncated_to_512": sum(1 for L in lens if L == MAX_LEN)}}

    model, dim = build()
    with torch.inference_mode():
        ref = np.concatenate([model(torch.from_numpy(i), torch.from_numpy(m)).numpy()
                              for i, m in batches])
    del model
    res["dim"] = dim

    for name, tc, ta in (("model.onnx", TOL_COS_SELF, TOL_ABS_SELF),
                         ("model_fp16.onnx", TOL_COS, TOL_ABS)):
        p = Path(onnx_dir) / name
        if not p.exists():
            continue
        s = _sess(p)
        t = time.time()
        got = _run(s, batches)
        secs = time.time() - t
        del s
        cos, na, nb = _cos(ref, got)
        mx = float(np.abs(ref - got).max())
        res[f"parity_{name}"] = {
            "min_cos": float(cos.min()), "mean_cos": float(cos.mean()), "max_abs": mx,
            "min_dot": float((ref * got).sum(1).min()),
            "ref_norms": [float(na.min()), float(na.max())],
            "out_norms": [float(nb.min()), float(nb.max())],
            "all_finite": bool(np.isfinite(got).all()),
            "seconds_for_n": round(secs, 1), "tol_cos": tc, "tol_abs": ta,
            "pass": bool(cos.min() >= 1 - tc and mx <= ta and np.isfinite(got).all())}
        print(f"  {name:18s} cos {cos.min():.8f}  max-abs {mx:.3e}  norms "
              f"{nb.min():.6f}-{nb.max():.6f}  {secs:.0f}s  -> "
              f"{'PASS' if res[f'parity_{name}']['pass'] else 'FAIL'}", flush=True)

    # batch invariance, per shipped graph: padded positions must contribute nothing to either the
    # masked sum or its denominator. Run on the longest and shortest fixtures together.
    probe = [texts[0], texts[len(texts) // 2], texts[-1]]
    singles = [encode([t], batch=1) for t in probe]
    for name in GRAPHS:
        p = Path(onnx_dir) / name
        if not p.exists():
            continue
        s = _sess(p)
        alone = np.concatenate([_run(s, x) for x in singles])
        ragged = _run(s, encode(probe, batch=len(probe)))
        del s
        res[f"batch_invariance_{name}"] = {
            "max_abs": float(np.abs(alone - ragged).max()),
            "lengths": [int(x[0][1].sum()) for x in singles],
            "pass": bool(np.abs(alone - ragged).max() <= TOL_ABS)}
        print(f"  batch invariance {name:18s} max-abs "
              f"{res[f'batch_invariance_{name}']['max_abs']:.3e}", flush=True)
    return res


def check(onnx_dir, n=None):
    res = {"fixtures_sha256": sha256(FIXTURES), "graphs": check_graphs(onnx_dir)}
    res.update(check_parity(onnx_dir, n))
    shipped = [g for g in GRAPHS if res["graphs"].get(g, {}).get("present")]
    res["fp16_shippable"] = bool(res.get("parity_model_fp16.onnx", {}).get("pass")
                                 and res["graphs"].get("model_fp16.onnx", {}).get("pass"))
    # A PRESENT graph that fails its parity must fail the RUN. Scoring only model.onnx let
    # `PASS  fp16 shippable: False` exit zero while leaving the failed file in the output dir,
    # where any gate treating exit status as approval would upload it (Codex, 2026-09-03).
    res["pass"] = bool(
        res["graphs"]["model.onnx"]["pass"]
        and res["parity_model.onnx"]["pass"]
        and all(res[f"batch_invariance_{g}"]["pass"] for g in shipped)
        and ("model_fp16.onnx" not in shipped or res["fp16_shippable"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-fixtures", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--fp16", action="store_true",
                    help="also build model_fp16.onnx -- EXPERIMENTAL, not released; it is wrong "
                         "on CUDA (results/m11_doc_fp16_gpu.json)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--onnx-dir", default=str(OUT))
    ap.add_argument("--n", type=int, default=None, help="subset per stratum; default all")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    if not (a.make_fixtures or a.export or a.check):
        ap.error("pass --make-fixtures, --export and/or --check")

    t0 = time.time()
    if a.make_fixtures:
        make_fixtures()
    meta = export(Path(a.onnx_dir), fp16=a.fp16) if a.export else {}
    if not a.check:
        return
    res = check(a.onnx_dir, a.n)
    res.update(meta)
    res.update({"_what": "M11a T3: stella_en_400M_v5 document path -> ONNX, verified on REAL "
                         "passages with a true cosine. Supersedes results/m9_doc_export.json.",
                "repo": REPO_ID, "revision": REVISION, "dense": DENSE,
                "config_kwargs": CONFIG_KWARGS, "max_length": MAX_LEN,
                "fixtures": str(FIXTURES.relative_to(REPO)), "onnx_dir": str(a.onnx_dir),
                "seconds": round(time.time() - t0, 1),
                "host": {"platform": platform.platform(), "python": platform.python_version()}})
    if not a.no_write:
        RESULT.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(f"  wrote {RESULT}")
    print(f"\n{'PASS' if res['pass'] else 'FAIL'}   fp16 shippable: {res['fp16_shippable']}")
    if not res["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
