"""M10.0-a2 (Mac, CPU): can fastembed serve the three-layer pooled head exactly?

The M10 default student concatenates masked-mean-pooled hidden states of layers 12, 8 and 4
(1152-d) before a Linear(1152 -> 1024) head. Mean pooling is linear, so applying the head PER
TOKEN over the concatenated per-token states and letting fastembed mean-pool afterwards is the
same vector as pooling first (identical masking required). This exports that per-token graph
(opset 17), registers it as a fastembed custom model with MEAN pooling + normalization, and
compares fastembed's output to the in-graph pool-then-head reference. Random head weights: the
check is about the serving path, not quality. Writes results/m10_head_width_parity_mac.json.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "m10_head_width_parity_mac.json"
DIR = REPO / "work" / "m10onnx" / "nano-3layer"
STUDENT = "BAAI/bge-small-en-v1.5"
LAYERS = (12, 8, 4)
NAME = "qdrant/nano-3layer-parity-check"


class PerTokenHead(torch.nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, input_ids, attention_mask):
        hs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True).hidden_states
        tok = torch.cat([hs[l] for l in LAYERS], dim=-1)          # [b, s, 1152]
        return self.head(tok)                                      # [b, s, 1024] per token


def pooled_reference(model, tok, texts):
    """Pool the three layers FIRST, then the head, then normalize -- the training-time form."""
    b = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.inference_mode():
        hs = model.backbone(**b, output_hidden_states=True).hidden_states
        m = b["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
        feat = torch.cat([(hs[l] * m).sum(1) / m.sum(1).clamp(min=1e-9) for l in LAYERS], -1)
        v = model.head(feat)
        return torch.nn.functional.normalize(v, dim=-1).numpy(), b


def main():
    from transformers import AutoModel, AutoTokenizer
    from datasets import load_dataset
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(STUDENT)
    backbone = AutoModel.from_pretrained(STUDENT, dtype=torch.float32).eval()
    head = torch.nn.Linear(384 * len(LAYERS), 1024)
    model = PerTokenHead(backbone, head).eval()
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
    backbone.config.save_pretrained(DIR)   # fastembed reads config.json from the model dir
    # transformers 5.x fast tokenizers no longer write special_tokens_map.json; fastembed requires it
    (DIR / "special_tokens_map.json").write_text(json.dumps(
        {k: (v if isinstance(v, str) else str(v)) for k, v in tok.special_tokens_map.items()}, indent=1))
    import onnx
    g = onnx.load(str(p))
    custom = sorted({n.domain for n in g.graph.node} - {"", "ai.onnx", "ai.onnx.ml"})
    n_params = sum(p_.numel() for p_ in model.parameters())

    qs = list(load_dataset("google-research-datasets/nq_open", split="train")["question"])[:56]
    texts = qs + [" ".join(qs[i:i + 12]) for i in range(0, 48, 12)] + \
            ["Instruct: retrieve", "?", "a " * 300, "The quick brown fox jumps over the lazy dog."]
    ref, _ = pooled_reference(model, tok, texts)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    b = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    toks = sess.run(None, {"input_ids": b["input_ids"].numpy(), "attention_mask": b["attention_mask"].numpy()})[0]
    m = b["attention_mask"].numpy()[..., None].astype(np.float32)
    manual = (toks * m).sum(1) / np.maximum(m.sum(1), 1e-9)
    manual = manual / np.linalg.norm(manual, axis=1, keepdims=True)
    cos_manual = (ref * manual).sum(1)

    res = {"_what": __doc__.strip(), "student": STUDENT, "layers": list(LAYERS), "opset": 17,
           "custom_domain_ops": custom, "n_nodes": len(g.graph.node), "onnx_bytes": p.stat().st_size,
           "params_total": int(n_params), "params_head": int(sum(p_.numel() for p_ in head.parameters())),
           "n_texts": len(texts),
           "ort_then_manual_mean_pool": {"min_cos": float(cos_manual.min()), "max_abs": float(np.abs(ref - manual).max())}}
    try:
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType
        TextEmbedding.add_custom_model(model=NAME, pooling=PoolingType.MEAN, normalization=True,
                                       sources=ModelSource(hf=NAME), dim=1024, model_file="model.onnx",
                                       description="M10 three-layer per-token head parity check",
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
