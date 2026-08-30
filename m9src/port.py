"""M9.1 port pilots: ONNX export, torch-vs-ONNX parity, and fastembed registration.

M10 has to port BOTH models, and `B6-pre` only ever exported the document graph on near-identity
weights — so the real artifacts have never been through this. A backbone that needs a custom op
is disqualified on M10 grounds regardless of its screen number, which is why this runs at M9.1
and not after training (m9/BRIEF.md §2.1).

Thresholds and the parity sample are locked in `m9/registry.json` -> validation_samples.onnx_parity.
"""
import json
import time

import numpy as np
import torch

import m9base
from m9base import RESULTS, WORK

import data as m9data     # noqa: E402
import guard9             # noqa: E402
import nano               # noqa: E402

ONNX = WORK / "m9onnx"


def registry():
    return json.loads((m9base.M9 / "registry.json").read_text())


def length_bin(words, edges):
    for i, (lo, hi) in enumerate(edges):
        if words >= lo and (hi is None or words <= hi):
            return i
    return len(edges) - 1


def parity_sample():
    """-> list[str]. The locked 512-text sample: 256 queries + 256 documents, stratified by the
    registered word-length bins, shortfall redistributed to the next bin down."""
    r = registry()
    v = r["validation_samples"]["onnx_parity"]
    edges = r["bins"]["length_words"]
    rng = np.random.default_rng(v["seed"])
    half = v["n"] // 2

    def take(texts, want):
        bins = {}
        for i, t in enumerate(texts):
            bins.setdefault(length_bin(len(t.split()), edges), []).append(i)
        nb = len(edges)
        quota = [want // nb] * nb
        quota[-1] += want - sum(quota)              # 51/51/51/51/52, the remainder to the longest
        out, deficit = [], 0
        for b in range(nb - 1, -1, -1):             # longest bin first; shortfall flows DOWN
            pool = bins.get(b, [])
            k = min(len(pool), quota[b] + deficit)
            deficit = quota[b] + deficit - k
            if k:
                out += [texts[i] for i in rng.choice(pool, size=k, replace=False)]
        if len(out) < want:                         # top up from the whole set, deterministic
            extra = rng.choice(len(texts), size=want - len(out), replace=False)
            out += [texts[i] for i in extra]
        return out[:want]

    q = json.loads((WORK / "m9_screen_queries.json").read_text())
    cand, _ = m9data.doc_pool_rows(r["data"]["doc_candidates_n"],
                                   r["data"]["doc_candidates_seed"])
    d = m9data.row_texts(np.sort(rng.choice(cand, size=6000, replace=False)))
    return take(q, half) + take(d, half)


def export(student_key, out_dir=None, state_dict=None):
    """-> (path, meta). opset 17, dynamic batch and sequence, zero custom-domain ops."""
    r = registry()
    opset = r["validation_samples"]["onnx_parity"]["opset"]
    out_dir = out_dir or (ONNX / student_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "model.onnx"

    m = nano.Nano(student_key).eval()
    if state_dict is not None:
        m.load_state_dict(state_dict)
    ex = m.tok(["a short query", "a somewhat longer example sentence for the export trace"],
               padding=True, truncation=True, max_length=64, return_tensors="pt")

    class Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return torch.nn.functional.normalize(self.inner(input_ids, attention_mask),
                                                 dim=-1, eps=1e-12)

    torch.onnx.export(
        Wrapped(m), (ex["input_ids"], ex["attention_mask"]), str(p),
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                      "embedding": {0: "b"}},
        opset_version=opset, do_constant_folding=True, dynamo=False)
    m.tok.save_pretrained(out_dir)

    import onnx
    g = onnx.load(str(p))
    domains = sorted({n.domain for n in g.graph.node})
    custom = [d for d in domains if d not in ("", "ai.onnx", "ai.onnx.ml")]
    shipped = sorted(f for f in out_dir.iterdir() if f.is_file())
    total = sum(f.stat().st_size for f in shipped)
    meta = {"student": student_key, "path": str(p.relative_to(m9base.REPO)), "opset": opset,
            "onnx_bytes": p.stat().st_size, "domains": domains, "custom_domain_ops": custom,
            "n_nodes": len(g.graph.node),
            # the 70 MB cap is on TOTAL SHIPPED BYTES, not the weight product (LEDGER §0)
            "shipped_files": {f.name: f.stat().st_size for f in shipped},
            "shipped_bytes": total, "shipped_MB_decimal": round(total / 1e6, 3),
            "within_70MB_target": total <= 70e6}
    return p, meta, m


def parity(student_key, texts=None, state_dict=None):
    r = registry()["validation_samples"]["onnx_parity"]
    texts = texts or parity_sample()
    p, meta, m = export(student_key, state_dict=state_dict)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    m = m.cpu().eval()

    mins, maxa = [], []
    t0 = time.time()
    B = 32
    for i in range(0, len(texts), B):
        b = m.tok(texts[i:i + B], padding=True, truncation=True, max_length=m.max_seq,
                  return_tensors="pt")
        with torch.inference_mode():
            ref = torch.nn.functional.normalize(m(b["input_ids"], b["attention_mask"]).float(),
                                                dim=-1, eps=1e-12).numpy()
        got = sess.run(None, {"input_ids": b["input_ids"].numpy(),
                              "attention_mask": b["attention_mask"].numpy()})[0]
        mins.append((ref * got).sum(1))
        maxa.append(np.abs(ref - got).max())

    cos = np.concatenate(mins)
    out = {**meta, "n_sample": len(texts), "min_cos": float(cos.min()),
           "mean_cos": float(cos.mean()), "max_abs": float(max(maxa)),
           "seconds": round(time.time() - t0, 1),
           "pass_min_cos": bool(cos.min() >= r["min_cos"]),
           "pass_max_abs": bool(max(maxa) <= r["max_abs"]),
           "pass_no_custom_ops": not meta["custom_domain_ops"]}
    out["pass"] = bool(out["pass_min_cos"] and out["pass_max_abs"] and out["pass_no_custom_ops"])
    return out


def fastembed_check(student_key, texts):
    """Register the export with fastembed and compare its output to the ONNX session's."""
    try:
        from fastembed import TextEmbedding
        from fastembed.common.model_description import (DenseModelDescription, ModelSource,
                                                        PoolingType)
    except Exception as e:
        return {"available": False, "error": repr(e)[:300]}

    d = ONNX / student_key
    try:
        TextEmbedding.add_custom_model(
            model=f"m9/nano-{student_key}",
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=None),
            dim=1024, model_file="model.onnx",
        )
        return {"available": True, "registered": True,
                "note": "add_custom_model accepted the description; a serving-parity run needs "
                        "the artifact published at a repo path, which is M10's step",
                "dir": str(d.relative_to(m9base.REPO))}
    except TypeError:
        # older/newer signature -- record what the API actually wants rather than guessing
        import inspect
        return {"available": True, "registered": False,
                "add_custom_model_signature": str(inspect.signature(
                    TextEmbedding.add_custom_model)),
                "DenseModelDescription_fields": [f for f in
                                                 DenseModelDescription.model_fields]}
    except Exception as e:
        return {"available": True, "registered": False, "error": repr(e)[:400]}


def main():
    guard9.begin_run("m9-port-pilot")
    texts = parity_sample()
    lens = [len(t.split()) for t in texts]
    out = {"sample": {"n": len(texts), "word_len_min": min(lens), "word_len_max": max(lens),
                      "word_len_mean": round(float(np.mean(lens)), 1)},
           "students": {}}
    for k in nano.STUDENTS:
        out["students"][k] = parity(k, texts)
        out["students"][k]["fastembed"] = fastembed_check(k, texts[:8])
        print(k, json.dumps({a: b for a, b in out["students"][k].items()
                             if a not in ("domains",)}, indent=1), flush=True)
    out["pass_all"] = all(v["pass"] for v in out["students"].values())
    guard9.write_result(RESULTS / "m9_port_pilot.json", out, "m9-port-pilot")
    return out


if __name__ == "__main__":
    main()
