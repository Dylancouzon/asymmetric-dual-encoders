"""Can the DOCUMENT model be ported? Export stella-400M to ONNX and check it end to end.

M10 ships both halves of the pair, and the document half has never been exported. `B6-pre` did
export a document graph at opset 17 with zero custom-domain ops and parity 0.99999994 — but on
**near-identity weights**, so it proved the harness, not the artifact. stella-400M carries custom
remote code (`NewModel`, an xformers assert, `unpad_inputs`), and the mandate is explicit that a
backbone needing custom ops is disqualified on M10 grounds alone. So: find out now.

Device-independent by construction — a graph either exports with standard ops or it does not — and
it feeds no quality decision, so it belongs off the training box.

    python m9src/export_doc_model.py
    python m9src/export_doc_model.py --serve      # also try the fastembed local serving route
"""
import argparse
import json
import platform
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "work" / "m9onnx" / "stella-400M-doc"

REPO_ID = "NovaSearch/stella_en_400M_v5"
REVISION = "ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20"
DENSE = "2_Dense_1024"
# NewModel asserts on xformers unless both are off; these must land on the CONFIG, not on
# from_pretrained (transformers 4.57 forwards unknown kwargs to __init__ and raises).
CONFIG_KWARGS = {"use_memory_efficient_attention": False, "unpad_inputs": False}
WORDS = ("retrieval embedding vector index document query passage neural token model ranking "
         "corpus semantic dense sparse encoder decoder attention transformer").split()


def synth(nwords, n, seed=0):
    import random
    r = random.Random(seed)
    return [" ".join(r.choice(WORDS) for _ in range(nwords)) for _ in range(n)]


def build():
    """-> (wrapped torch module, tokenizer, output dim). Mean pool -> Dense(1024) -> L2."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(REPO_ID, revision=REVISION)
    cfg = AutoConfig.from_pretrained(REPO_ID, revision=REVISION, trust_remote_code=True)
    for k, v in CONFIG_KWARGS.items():
        setattr(cfg, k, v)
    backbone = AutoModel.from_pretrained(REPO_ID, revision=REVISION, config=cfg,
                                         trust_remote_code=True).eval()
    dcfg = json.loads(Path(hf_hub_download(REPO_ID, f"{DENSE}/config.json",
                                           revision=REVISION)).read_text())
    assert dcfg.get("activation_function", "").endswith("Identity"), dcfg
    sd = load_file(hf_hub_download(REPO_ID, f"{DENSE}/model.safetensors", revision=REVISION))
    W = next(v for k, v in sd.items() if v.dim() == 2)
    b = next((v for k, v in sd.items() if v.dim() == 1), None)

    class DocEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.b = backbone
            self.dense = torch.nn.Linear(W.shape[1], W.shape[0], bias=b is not None)
            with torch.no_grad():
                self.dense.weight.copy_(W)
                if b is not None:
                    self.dense.bias.copy_(b)

        def forward(self, input_ids, attention_mask):
            h = self.b(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(self.dense(v), dim=-1, eps=1e-12)

    return DocEncoder().eval(), tok, W.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="also try fastembed's local model route")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    import numpy as np
    import torch

    out = {"_what": "can the DOCUMENT half of the pair be ported? stella-400M -> ONNX. B6-pre "
                    "only ever exported a document graph on near-identity weights, so the real "
                    "artifact has never been through this. Feeds no quality decision.",
           "repo": REPO_ID, "revision": REVISION, "opset": a.opset,
           "host": {"platform": platform.platform(), "machine": platform.machine(),
                    "python": platform.python_version()}}
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        model, tok, dim = build()
        out["dim"] = int(dim)
        out["params"] = sum(p.numel() for p in model.parameters())
        print(f"loaded stella-400M + {DENSE}: {out['params']/1e6:.0f}M params, dim {dim} "
              f"({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        out["load_error"] = repr(e)[:500]
        print("LOAD FAILED:", repr(e)[:300])
        (REPO / "results" / "m9_doc_export.json").write_text(json.dumps(out, indent=2))
        return out

    p32 = OUT / "model.onnx"
    ex = tok(synth(12, 2) + synth(200, 1), padding=True, truncation=True, max_length=512,
             return_tensors="pt")
    try:
        t1 = time.time()
        torch.onnx.export(model, (ex["input_ids"], ex["attention_mask"]), str(p32),
                          input_names=["input_ids", "attention_mask"],
                          output_names=["embedding"],
                          dynamic_axes={"input_ids": {0: "b", 1: "s"},
                                        "attention_mask": {0: "b", 1: "s"},
                                        "embedding": {0: "b"}},
                          opset_version=a.opset, do_constant_folding=True, dynamo=False)
        tok.save_pretrained(OUT)
        out["export_seconds"] = round(time.time() - t1, 1)
    except Exception as e:
        out["export_error"] = repr(e)[:800]
        print("EXPORT FAILED:", repr(e)[:400])
        (REPO / "results" / "m9_doc_export.json").write_text(json.dumps(out, indent=2))
        return out

    import onnx
    g = onnx.load(str(p32))
    domains = sorted({n.domain for n in g.graph.node})
    custom = [d for d in domains if d not in ("", "ai.onnx", "ai.onnx.ml")]
    out.update({"onnx_bytes": p32.stat().st_size, "n_nodes": len(g.graph.node),
                "domains": domains, "custom_domain_ops": custom,
                "pass_no_custom_ops": not custom})
    print(f"exported: {len(g.graph.node)} nodes, {p32.stat().st_size/1e6:.0f} MB, "
          f"domains {domains}", flush=True)

    try:
        import copy
        from onnxruntime.transformers.float16 import convert_float_to_float16
        p16 = OUT / "model_fp16.onnx"
        onnx.save(convert_float_to_float16(copy.deepcopy(g), keep_io_types=True), str(p16))
        out["onnx_fp16_bytes"] = p16.stat().st_size
    except Exception as e:
        out["fp16_error"] = repr(e)[:300]

    # parity: torch fp32 vs the exported graph, over short AND long inputs so the dynamic
    # sequence axis is actually exercised
    import onnxruntime as ort
    sess = ort.InferenceSession(str(p32), providers=["CPUExecutionProvider"])
    cos, mx = [], 0.0
    for nw in (5, 12, 40, 120, 400):
        b = tok(synth(nw, 8, seed=nw), padding=True, truncation=True, max_length=512,
                return_tensors="pt")
        with torch.inference_mode():
            ref = model(b["input_ids"], b["attention_mask"]).numpy()
        got = sess.run(None, {"input_ids": b["input_ids"].numpy(),
                              "attention_mask": b["attention_mask"].numpy()})[0]
        cos.append((ref * got).sum(1))
        mx = max(mx, float(np.abs(ref - got).max()))
    c = np.concatenate(cos)
    out.update({"parity_min_cos": float(c.min()), "parity_mean_cos": float(c.mean()),
                "parity_max_abs": mx, "parity_n": int(c.size),
                "pass_parity": bool(c.min() >= 1 - 1e-4 and mx <= 1e-3)})
    out["pass"] = bool(out["pass_no_custom_ops"] and out["pass_parity"])
    print(f"parity: min-cos {c.min():.8f}, max-abs {mx:.2e}  -> "
          f"{'PASS' if out['pass'] else 'FAIL'}", flush=True)

    if a.serve:
        out["fastembed_local"] = try_fastembed()

    out["seconds"] = round(time.time() - t0, 1)
    (REPO / "results" / "m9_doc_export.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ("domains",)}, indent=1))
    return out


def try_fastembed():
    """Does fastembed serve a LOCAL custom model directory? M10's port needs that route to work,
    and the M9 port pilot could only register a description, not serve one."""
    d = REPO / "work" / "m9onnx" / "nano-minilm-l6"
    if not (d / "model_fp16.onnx").exists():
        return {"skipped": "run m9src/edge_cost.py first -- it writes the nano graphs"}
    try:
        import numpy as np
        import onnxruntime as ort
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType
        name = "qdrant/nano-minilm-l6"
        TextEmbedding.add_custom_model(
            model=name, pooling=PoolingType.MEAN, normalization=True,
            sources=ModelSource(hf=name), dim=1024, model_file="model_fp16.onnx",
            description="M9 nano", license="mit", size_in_gb=0.05)
        emb = TextEmbedding(model_name=name, specific_model_path=str(d))
        v = np.asarray(list(emb.embed(["a short query", "another query about retrieval"])))
        sess = ort.InferenceSession(str(d / "model_fp16.onnx"),
                                    providers=["CPUExecutionProvider"])
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(str(d))
        b = tk(["a short query", "another query about retrieval"], padding=True,
               return_tensors="np")
        ref = sess.run(None, {"input_ids": b["input_ids"].astype("int64"),
                              "attention_mask": b["attention_mask"].astype("int64")})[0]
        cos = float((v * ref).sum(1).min())
        return {"served": True, "min_cos_vs_onnxruntime": cos, "shape": list(v.shape),
                "pass": bool(cos >= 0.999)}
    except Exception as e:
        return {"served": False, "error": repr(e)[:500]}


if __name__ == "__main__":
    main()
