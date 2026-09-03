"""Export `zero`'s query path to ONNX, and check the graph against the numpy encoder.

The rule (`zero_encoder.py:_encode_ids`): a token appearing c times carries TOTAL weight sqrt(c),
so the pooled vector is `sum_u sqrt(c_u)*row_u / sum_u sqrt(c_u)`, L2-normalized, with a fallback
to the normalized [CLS] row for a degenerate bag.

ONNX has no per-row `Unique` -- without an axis it flattens the batch, with one it uniques whole
slices -- so the graph uses the per-occurrence identity instead:

    sum_u sqrt(c_u)*row_u == sum_i row_{t_i}/sqrt(c_{t_i})
    sum_u sqrt(c_u)       == sum_i 1/sqrt(c_{t_i})

Counts come from an all-pairs comparison: Equal(ids[:,:,None], ids[:,None,:]) -> (b,s,s), with the
KEY axis masked so padding contributes to no count, then ReduceSum over it. Padded positions must
contribute zero to BOTH the count and the weight; masking only the weight passes every ordinary
query and fails only on a literal "[PAD]" in the text, which is why that fixture is in the check.

Two graphs from one table:
  model.onnx         (b,1024)  pooled + normalized, with the fallback -- for a direct ORT caller
  model_tokens.onnx  (b,s,1024) row_{t_i}/sqrt(c_{t_i}) -- for fastembed, whose masked mean and
                     normalize recover the same direction (the 1/n it divides by is annihilated)

  .venv/bin/python m11/release/export_onnx.py            # export + check
  .venv/bin/python m11/release/export_onnx.py --check    # check what is already exported
  .venv/bin/python m11/release/export_onnx.py --check --onnx-dir DIR --bundle DIR --no-write

The last form is what `push.py`'s ONNX gate runs against the STAGED files: it re-derives every
number below rather than reading a recorded verdict, because a `"pass": true` in a JSON binds the
file to a claim and not to the arithmetic.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

BUNDLE = REPO / "work/release/zero-v1"
OUT = REPO / "work/m11onnx/zero-v1"
RESULT = REPO / "results/m11_zero_export.json"
OPSET = 17
EPS = 1e-6


def const(name, arr):
    from onnx import helper, numpy_helper
    return helper.make_node("Constant", [], [name],
                            value=numpy_helper.from_array(np.asarray(arr), name + "_v"))


def build_graph(rows_int8, scale, fallback, tokens_only, mask_counts=True):
    """The graph, standard domain only. `tokens_only` emits the per-token rows instead of pooling.

    `mask_counts=False` builds the DEFECTIVE variant the plan warns about -- weight masked, count
    axis not -- so `main` can prove check 5 is load-bearing rather than merely green."""
    from onnx import TensorProto, helper, numpy_helper

    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"])
    am = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "s"])
    out_shape = ["b", "s", rows_int8.shape[1]] if tokens_only else ["b", rows_int8.shape[1]]
    out = helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, out_shape)

    inits = [numpy_helper.from_array(rows_int8, "TABLE"),
             numpy_helper.from_array(scale.astype(np.float32), "SCALE")]
    N = []

    def node(op, i, o, **kw):
        N.append(helper.make_node(op, i, o, **kw))

    N += [const("AX1", np.array([1], np.int64)), const("AX2", np.array([2], np.int64)),
          const("ONE", np.float32(1.0))]
    if not tokens_only:
        N.append(const("EPSC", np.float32(EPS)))

    node("Cast", ["attention_mask"], ["maskf"], to=TensorProto.FLOAT)          # (b,s)

    # counts, padding excluded from the KEY axis
    node("Unsqueeze", ["input_ids", "AX2"], ["idsq"])                          # (b,s,1)
    node("Unsqueeze", ["input_ids", "AX1"], ["idsk"])                          # (b,1,s)
    node("Equal", ["idsq", "idsk"], ["eq"])                                    # (b,s,s)
    node("Cast", ["eq"], ["eqf"], to=TensorProto.FLOAT)
    if mask_counts:
        node("Unsqueeze", ["maskf", "AX1"], ["maskk"])                         # (b,1,s)
        node("Mul", ["eqf", "maskk"], ["eqm"])
    else:
        node("Identity", ["eqf"], ["eqm"])
    node("ReduceSum", ["eqm", "AX2"], ["cnt"], keepdims=0)                      # (b,s)

    # w = mask / sqrt(max(c,1)) -- masked on the QUERY axis too
    node("Max", ["cnt", "ONE"], ["cnt1"])
    node("Sqrt", ["cnt1"], ["rc"])
    node("Div", ["maskf", "rc"], ["w"])                                        # (b,s)

    # dequantized rows for each position
    node("Gather", ["TABLE", "input_ids"], ["rq"], axis=0)                     # (b,s,d) int8
    node("Cast", ["rq"], ["rqf"], to=TensorProto.FLOAT)
    node("Gather", ["SCALE", "input_ids"], ["sc"], axis=0)                     # (b,s)
    node("Unsqueeze", ["sc", "AX2"], ["scq"])
    node("Mul", ["rqf", "scq"], ["rows"])                                      # (b,s,d)

    node("Unsqueeze", ["w", "AX2"], ["wq"])
    node("Mul", ["rows", "wq"], ["weighted"])                                  # (b,s,d)

    if tokens_only:
        N.append(helper.make_node("Identity", ["weighted"], ["embeddings"]))
    else:
        node("ReduceSum", ["weighted", "AX1"], ["num"], keepdims=0)             # (b,d)
        node("ReduceSum", ["w", "AX1"], ["den0"], keepdims=1)                   # (b,1)
        node("Max", ["den0", "EPSC"], ["den"])
        node("Div", ["num", "den"], ["vec"])
        node("Mul", ["vec", "vec"], ["sq"])
        node("ReduceSum", ["sq", "AX1"], ["ss"], keepdims=1)
        node("Sqrt", ["ss"], ["nrm"])                                           # (b,1)
        node("Max", ["nrm", "EPSC"], ["nrmc"])
        node("Div", ["vec", "nrmc"], ["unit"])
        # degenerate bag -> the normalized [CLS] row, exactly as the numpy encoder does
        N.append(const("FALLBACK", fallback.astype(np.float32).reshape(1, -1)))
        node("LessOrEqual", ["nrm", "EPSC"], ["degen"])                         # (b,1)
        node("Where", ["degen", "FALLBACK", "unit"], ["embeddings"])

    g = helper.make_graph(N, "zero-query", [ids, am], [out], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = 9
    return m


def export():
    import onnx
    from zero_encoder import ZeroQueryEncoder

    enc = ZeroQueryEncoder(BUNDLE, variant="int8")
    z = np.load(BUNDLE / "model.npz")
    rows_int8, scale = z["rows_int8"], z["int8_scale"]
    fb = enc._fallback

    OUT.mkdir(parents=True, exist_ok=True)
    for name, tokens_only in (("model.onnx", False), ("model_tokens.onnx", True)):
        m = build_graph(rows_int8, scale, fb, tokens_only)
        onnx.checker.check_model(m, full_check=True)
        onnx.save(m, str(OUT / name))
        mb = (OUT / name).stat().st_size / 1e6
        print(f"  wrote {name}  {mb:.1f} MB  opset {OPSET}")
    return enc


def ops_and_domains(path):
    import onnx
    m = onnx.load(str(path))
    return ({n.op_type for n in m.graph.node},
            {n.domain for n in m.graph.node} | {i.domain for i in m.opset_import})


def tokenize(enc, texts):
    """(ids, mask) padded to the batch longest -- what an ONNX caller has to supply."""
    encs = [enc.tokenizer.encode(t) for t in texts]
    L = max((len(e.ids) for e in encs), default=1) or 1
    ids = np.zeros((len(texts), L), np.int64)
    mask = np.zeros((len(texts), L), np.int64)
    for i, e in enumerate(encs):
        ids[i, :len(e.ids)] = e.ids
        mask[i, :len(e.ids)] = 1
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="skip the export, check what exists")
    ap.add_argument("--onnx-dir", default=None, help="where the graphs are (default work/m11onnx)")
    ap.add_argument("--bundle", default=None, help="where the table and tokenizer are")
    ap.add_argument("--no-write", action="store_true", help="do not write the result JSON")
    a = ap.parse_args()
    global BUNDLE, OUT
    if a.bundle:
        BUNDLE = Path(a.bundle)
    if a.onnx_dir:
        OUT = Path(a.onnx_dir)

    import onnxruntime as ort
    from zero_encoder import ZeroQueryEncoder

    enc = ZeroQueryEncoder(BUNDLE, variant="int8") if a.check else export()
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = {n: ort.InferenceSession(str(OUT / n), so, providers=["CPUExecutionProvider"])
            for n in ("model.onnx", "model_tokens.onnx")}

    def run(name, texts):
        ids, mask = tokenize(enc, texts)
        return sess[name].run(None, {"input_ids": ids, "attention_mask": mask})[0], mask

    checks, res = [], {}

    # 1 -- standard ops only
    ops, doms = ops_and_domains(OUT / "model.onnx")
    ops2, doms2 = ops_and_domains(OUT / "model_tokens.onnx")
    custom = (doms | doms2) - {""}
    checks.append(("1 standard domain only, opset 17, checker passes", not custom,
                   f"{len(ops | ops2)} op types, domains {sorted(doms | doms2)}"))

    # 2 -- parity against the numpy encoder on real dev queries
    dev = json.loads((REPO / "work/dev/heldout-train.json").read_text())["q_texts"]
    qs = dev[:1024]
    got = np.concatenate([run("model.onnx", qs[i:i + 64])[0] for i in range(0, len(qs), 64)])
    want = enc.encode(qs)
    dev_abs = float(np.abs(got - want).max())
    dev_cos = float((got * want).sum(1).min())
    res["dev_queries"] = {"n": len(qs), "max_abs": dev_abs, "min_cos": dev_cos}
    checks.append((f"2 parity on {len(qs)} real dev queries", dev_abs <= 1e-5 and dev_cos >= 1 - 1e-6,
                   f"max-abs {dev_abs:.3e}   min-cos {dev_cos:.9f}"))

    # 3 -- batch invariance: alone vs beside a long query in a padded batch
    long_q = " ".join(["retrieval augmented generation"] * 200)
    solo = run("model.onnx", [qs[0]])[0][0]
    inbatch = run("model.onnx", [qs[0], long_q])[0][0]
    inv = float(np.abs(solo - inbatch).max())
    res["batch_invariance_max_abs"] = inv
    checks.append(("3 batch invariance (padded beside a long query)", inv <= 1e-6, f"{inv:.3e}"))

    # 4 -- model_tokens + masked mean + L2 == model.onnx
    tv, mask = run("model_tokens.onnx", qs[:64])
    pooled = (tv * mask[:, :, None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), EPS)
    d4 = float(np.abs(pooled - run("model.onnx", qs[:64])[0]).max())
    res["tokens_route_max_abs"] = d4
    checks.append(("4 model_tokens + masked mean + L2 == model.onnx", d4 <= 1e-6, f"{d4:.3e}"))

    # 5 -- literal [PAD]: the fixture that separates count-masking from weight-masking
    pad_fx = ["[PAD]", "[PAD] [PAD] hello", "[CLS] [SEP] [UNK] [MASK]"]
    g5 = run("model.onnx", pad_fx)[0]
    d5 = float(np.abs(g5 - enc.encode(pad_fx)).max())
    res["pad_fixtures_max_abs"] = d5
    checks.append(("5 literal [PAD] / special tokens", d5 <= 1e-5, f"{d5:.3e}"))

    # 6 -- edge cases
    edge = ["", "   ", "a", "the the the the the the", "zzzqx",
            " ".join(["x"] * 511), " ".join(["x"] * 512), " ".join(["x"] * 513),
            "🜛 ᚠᚢᚦ 𐎠𐎢", "electroencephalographically " + "a" * 120,
            " ".join(["retrieval augmented generation"] * 400),
            "0 0 0 0 1 1 1 2", "one two three two one"]
    worst, worst_t = 0.0, None
    for t in edge:                              # one at a time: b=1 is a distinct path
        d = float(np.abs(run("model.onnx", [t])[0][0] - enc.encode(t)[0]).max())
        if d > worst:
            worst, worst_t = d, t
    res["edge_cases"] = {"n": len(edge), "max_abs": worst, "worst": (worst_t or "")[:40]}
    checks.append((f"6 {len(edge)} edge cases at b=1", worst <= 1e-5,
                   f"max-abs {worst:.3e} on {(worst_t or '')[:28]!r}"))

    # 7 -- permutation invariance: a bag has no order
    perm = "alpha beta gamma delta beta alpha"
    words = perm.split()
    variants = [" ".join(words[i:] + words[:i]) for i in range(len(words))]
    pv = run("model.onnx", variants)[0]
    d7 = float(np.abs(pv - pv[0]).max())
    res["permutation_max_abs"] = d7
    checks.append(("7 permutations of one bag agree", d7 <= 1e-6, f"{d7:.3e}"))

    # 9 -- the fallback branch. No tokenized text reaches it (add_special_tokens means even "" is
    # [CLS][SEP]), so it is only reachable by an all-masked row -- which is exactly why it would
    # otherwise ship untested.
    zi = np.zeros((2, 4), np.int64)
    zm = np.zeros((2, 4), np.int64)
    zm[1, :2] = 1
    zi[1, :2] = [101, 102]
    g9 = sess["model.onnx"].run(None, {"input_ids": zi, "attention_mask": zm})[0]
    d9 = float(np.abs(g9[0] - enc._fallback).max())
    res["fallback_max_abs"] = d9
    checks.append(("9 all-masked row returns the [CLS] fallback", d9 <= 1e-6, f"{d9:.3e}"))

    # 10 -- negative control: the defective graph the plan warns about must FAIL check 5, so that
    # check 5 passing means something. Weight masked, count axis not.
    import onnx
    bad_p = OUT / "_defective_count_mask.onnx"
    z = np.load(BUNDLE / "model.npz")
    onnx.save(build_graph(z["rows_int8"], z["int8_scale"], enc._fallback, False, mask_counts=False),
              str(bad_p))
    bad = ort.InferenceSession(str(bad_p), so, providers=["CPUExecutionProvider"])
    bi, bm = tokenize(enc, pad_fx)
    bad_out = bad.run(None, {"input_ids": bi, "attention_mask": bm})[0]
    bad_d = float(np.abs(bad_out - enc.encode(pad_fx)).max())
    bad_dev = float(np.abs(bad.run(None, dict(zip(("input_ids", "attention_mask"),
                                                  tokenize(enc, qs[:64]))))[0]
                           - enc.encode(qs[:64])).max())
    bad_p.unlink()
    res["negative_control"] = {"pad_fixtures_max_abs": bad_d, "dev_queries_max_abs": bad_dev}
    checks.append(("10 count-mask is load-bearing (defective graph fails 5)",
                   bad_d > 1e-5, f"defective: [PAD] {bad_d:.3e} vs dev {bad_dev:.3e}"))

    # 8 -- cost: the S x S count term is not free
    cost = {}
    for label, text in (("s=8", qs[0]), ("s=512", " ".join(["retrieval augmented"] * 400))):
        ids, mask = tokenize(enc, [text])
        sess["model.onnx"].run(None, {"input_ids": ids, "attention_mask": mask})
        t0 = time.perf_counter()
        for _ in range(50):
            sess["model.onnx"].run(None, {"input_ids": ids, "attention_mask": mask})
        cost[label] = (time.perf_counter() - t0) / 50 * 1e3
    res["latency_ms_1thread"] = cost
    checks.append(("8 cost row measured", True,
                   f"s=8 {cost['s=8']:.3f} ms   s=512 {cost['s=512']:.3f} ms"))

    ok = True
    print()
    for name, passed, detail in checks:
        ok &= bool(passed)
        print(f"{'PASS' if passed else 'FAIL'}  {name:52s}  {detail}")

    import hashlib
    res.update({
        "pass": bool(ok), "opset": OPSET, "eps": EPS,
        "source_table_sha256": json.loads((BUNDLE / "config.json").read_text())
            ["source_table_sha256"],
        "artifacts": {n: {"relpath": str(OUT.joinpath(n).resolve().relative_to(REPO)),
                          "bytes": (OUT / n).stat().st_size,
                          "sha256": hashlib.sha256((OUT / n).read_bytes()).hexdigest()}
                      for n in ("model.onnx", "model_tokens.onnx")},
        "checks": [{"name": n, "pass": bool(p), "detail": d} for n, p, d in checks],
    })
    if not a.no_write:
        RESULT.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(f"\n{'EXPORT OK' if ok else 'EXPORT FAILED'} -> {RESULT.relative_to(REPO)}")
    else:
        print(f"\n{'ONNX OK' if ok else 'ONNX FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
