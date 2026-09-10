"""M10.0-a3 (box, CPU): can fastembed serve a NONLINEAR per-token head exactly?

`m10/EXPLORED.md` closed "a nonlinear head" as having no fastembed serving path. That is true of a
head applied AFTER pooling (fastembed pools the graph's per-token output itself, and a nonlinearity
does not commute with the mean). It is NOT true of a head applied PER TOKEN before pooling: any
per-token function is exported as the graph's token output and fastembed's masked mean + normalize
reproduces the training-time `normalize(mean_t(head(x_t)))` exactly, because the pooling is the
same linear op on both sides. Two forms, both over the concatenated states of layers 12/8/4:
  mlp       Linear(1152->k) -> GELU -> Linear(k->1024)            (rank of the output <= k)
  residual  Linear(1152->1024)(x) + Linear(k->1024)(GELU(Linear(1152->k)(x)))
            -- the anchor's full-rank linear head plus a rank-k nonlinear correction; this is the
            G-MLP arm after the Codex pass of 2026-09-04 (the bottleneck form capped rank at 512).
Random weights (parity does not depend on them); records the parameter count against the 35M cap.
Writes results/m10_head_mlp_parity_box.json (mode and k in the record). Diagnostic; read by no rule.
Usage: head_mlp_parity.py [mlp|residual] [k]
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "m10_head_mlp_parity_box.json"
DIR = REPO / "work" / "m10onnx" / "nano-3layer-mlp"   # overwritten per run; the JSON records which form
STUDENT = "BAAI/bge-small-en-v1.5"
LAYERS = (12, 8, 4)
MODE = sys.argv[1] if len(sys.argv) > 1 else "residual"
HIDDEN = int(sys.argv[2]) if len(sys.argv) > 2 else (192 if MODE == "residual" else 512)


class ResidualHead(torch.nn.Module):
    """Full-rank linear path plus a rank-k GELU correction, applied per token."""
    def __init__(self, d_in, k, d_out):
        super().__init__()
        self.lin = torch.nn.Linear(d_in, d_out)
        self.down = torch.nn.Linear(d_in, k)
        self.up = torch.nn.Linear(k, d_out)

    def forward(self, x):
        return self.lin(x) + self.up(torch.nn.functional.gelu(self.down(x)))
NAME = "qdrant/nano-3layer-mlp-parity-check"


class PerTokenMLP(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, input_ids, attention_mask):
        hs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True).hidden_states
        tok = torch.cat([hs[l] for l in LAYERS], dim=-1)          # [b, s, 1152]
        return self.head(tok)                                      # [b, s, 1024] per token, nonlinear


def reference(model, tok, texts):
    """Training-time form: per-token MLP, THEN masked mean, THEN normalize."""
    b = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.inference_mode():
        t = model(b["input_ids"], b["attention_mask"])
        m = b["attention_mask"].unsqueeze(-1).to(t.dtype)
        v = (t * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(v, dim=-1).numpy()


def main():
    from transformers import AutoModel, AutoTokenizer
    from datasets import load_dataset
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(STUDENT)
    backbone = AutoModel.from_pretrained(STUDENT, dtype=torch.float32).eval()
    if MODE == "residual":
        head = ResidualHead(384 * len(LAYERS), HIDDEN, 1024)
    else:
        head = torch.nn.Sequential(torch.nn.Linear(384 * len(LAYERS), HIDDEN), torch.nn.GELU(),
                                   torch.nn.Linear(HIDDEN, 1024))
    model = PerTokenMLP(backbone, head).eval()
    DIR.mkdir(parents=True, exist_ok=True)
    ex = tok(["a short query", "a somewhat longer example sentence for the export trace"],
             padding=True, truncation=True, max_length=64, return_tensors="pt")
    p = DIR / "model.onnx"
    torch.onnx.export(model, (ex["input_ids"], ex["attention_mask"]), str(p),
                      input_names=["input_ids", "attention_mask"], output_names=["token_embeddings"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                    "token_embeddings": {0: "b", 1: "s"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)
    tok.save_pretrained(DIR)
    backbone.config.save_pretrained(DIR)
    (DIR / "special_tokens_map.json").write_text(json.dumps(
        {k: (v if isinstance(v, str) else str(v)) for k, v in tok.special_tokens_map.items()}, indent=1))
    import onnx
    g = onnx.load(str(p))
    custom = sorted({n.domain for n in g.graph.node} - {"", "ai.onnx", "ai.onnx.ml"})
    n_params = sum(q.numel() for q in model.parameters())
    n_head = sum(q.numel() for q in head.parameters())

    qs = list(load_dataset("google-research-datasets/nq_open", split="train")["question"])[:56]
    texts = qs + [" ".join(qs[i:i + 12]) for i in range(0, 48, 12)] + \
            ["Instruct: retrieve", "?", "a " * 300, "The quick brown fox jumps over the lazy dog."]
    ref = reference(model, tok, texts)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    b = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    toks = sess.run(None, {"input_ids": b["input_ids"].numpy(), "attention_mask": b["attention_mask"].numpy()})[0]
    m = b["attention_mask"].numpy()[..., None].astype(np.float32)
    manual = (toks * m).sum(1) / np.maximum(m.sum(1), 1e-9)
    manual = manual / np.linalg.norm(manual, axis=1, keepdims=True)
    cos_manual = (ref * manual).sum(1)

    res = {"_what": __doc__.strip(), "student": STUDENT, "layers": list(LAYERS), "mode": MODE, "k": HIDDEN,
           "head": (f"Linear(1152->1024)(x) + Linear({HIDDEN}->1024)(GELU(Linear(1152->{HIDDEN})(x))), per token"
                    if MODE == "residual" else f"1152->{HIDDEN}->GELU->1024, per token"),
           "opset": 17, "custom_domain_ops": custom, "n_nodes": len(g.graph.node), "onnx_bytes": p.stat().st_size,
           "params_total": int(n_params), "params_head": int(n_head), "under_35M_cap": bool(n_params <= 35_000_000),
           "n_texts": len(texts),
           "ort_then_manual_mean_pool": {"min_cos": float(cos_manual.min()), "max_abs": float(np.abs(ref - manual).max())}}
    try:
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType
        TextEmbedding.add_custom_model(model=NAME, pooling=PoolingType.MEAN, normalization=True,
                                       sources=ModelSource(hf=NAME), dim=1024, model_file="model.onnx",
                                       description="M10 per-token MLP head parity check",
                                       license="mit", size_in_gb=round(p.stat().st_size / 1e9, 4))
        t0 = time.time()
        fe = TextEmbedding(model_name=NAME, specific_model_path=str(DIR), threads=4)
        got = np.stack(list(fe.embed(texts, batch_size=32)))
        cos = (ref * got).sum(1)
        res["fastembed"] = {"version": __import__("fastembed").__version__, "served": True,
                            "min_cos": float(cos.min()), "mean_cos": float(cos.mean()),
                            "max_abs": float(np.abs(ref - got).max()), "seconds": round(time.time() - t0, 1),
                            "pass_min_cos_1e-4": bool(cos.min() >= 1 - 1e-4), "pass_max_abs_1e-3": bool(np.abs(ref - got).max() <= 1e-3)}
    except Exception as e:
        res["fastembed"] = {"served": False, "error": repr(e)[:500]}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "_what"}, indent=1))


if __name__ == "__main__":
    main()
