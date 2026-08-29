"""B6-pre: can the teacher plus a doc-side head export to ONE ONNX file? (LEDGER §9, §11.3.)

E3's hard condition, in Dylan's words: the doc-side head is approved **only** if it "fuses into
the doc ONNX graph as plain nodes -- one served file, no custom pipeline". So this is a binary
feasibility gate, and it is a gate on D1's existence: if it fails, D1 is closed, B6's quality arm
never runs, and the doc-side-head row comes off the Stage-S menu. That is the registered
no-survivor outcome, not a disappointment to be worked around.

WHAT IS ACTUALLY BEING TESTED. Not "can a doc-side head be exported" -- a `Linear` obviously can.
The question is whether the head fuses into the TEACHER's graph, so that what a user downloads and
serves is one file whose output is already the mapped, renormalized document vector. Two things
therefore have to hold at once:
  1. **stella exports at all.** Its blocker is documented as two config flags (`unpad_inputs`,
     `use_memory_efficient_attention`), both already set False in this project's `Spec`
     (`research/m8-planning/onnx-feasibility-2026-08-29.md`). It runs under `trust_remote_code`,
     so the export path goes through vendored modelling code rather than a native architecture --
     which is exactly why this is measured rather than assumed.
  2. **the composition stays plain.** The head must appear as MatMul/Add/activation and the final
     L2 normalize as ordinary ops. If the exporter emits a custom op, a subgraph the runtime
     cannot fold, or requires a Python pre/post step, E3's condition fails even though a file
     was produced.

PARITY IS THE ACCEPTANCE CRITERION, not the mere existence of a file. The exported graph must
reproduce the torch forward of the SAME composed module within the §11.4 tolerances. A graph that
exports and then disagrees is worse than no graph.

Run with `--head none` to isolate whether a failure is the teacher's export or the composition's.
"""
import argparse
import json
import sys
import time

import numpy as np

import m8base
import probe_guard

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_b6_pre.json"
EXPORT_DIR = m8base.WORK / "onnx"
COS_TOL = 1e-4          # LEDGER §11.4 vector/cosine tolerance for the parity fixtures
MAXABS_TOL = 1e-3

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Rosacea treatments and kits for performing them.",
    "What is the difference between a list and a tuple in Python?",
    "SARS-CoV-2 spike protein binding affinity to ACE2 receptors.",
    "a",
    "",
]


class Composed:
    """Teacher -> pool -> (optional doc-side head) -> L2 normalize, as ONE torch module."""

    def __new__(cls, spec, head, device):
        import torch
        import torch.nn as nn
        from teacher import load_post_dense, load_teacher

        _, backbone = load_teacher(dtype=torch.float32, device=device)
        dense = load_post_dense(spec, device)

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.pooling = spec.pooling
                # stella's published Dense head, folded in as an ordinary Linear so the exporter
                # sees a MatMul rather than a sentence-transformers module.
                self.dense = None
                if dense is not None:
                    w, b = dense
                    lin = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
                    with torch.no_grad():
                        lin.weight.copy_(torch.as_tensor(w))
                        if b is not None:
                            lin.bias.copy_(torch.as_tensor(b))
                    self.dense = lin.to(device)
                # The D1 candidate itself: a square linear map over the document vector. Identity-
                # initialized so a parity failure is unambiguously the EXPORT's fault and not a
                # random map's numerics.
                self.head = None
                if head == "linear":
                    h = nn.Linear(spec.dim, spec.dim, bias=True)
                    with torch.no_grad():
                        h.weight.copy_(torch.eye(spec.dim))
                        h.bias.zero_()
                    self.head = h.to(device)
                elif head == "mlp":
                    h = nn.Sequential(nn.Linear(spec.dim, spec.dim), nn.GELU(),
                                      nn.Linear(spec.dim, spec.dim))
                    with torch.no_grad():
                        h[0].weight.copy_(torch.eye(spec.dim)); h[0].bias.zero_()
                        h[2].weight.copy_(torch.eye(spec.dim)); h[2].bias.zero_()
                    self.head = h.to(device)

            def forward(self, input_ids, attention_mask):
                h = self.backbone(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state
                if self.pooling == "cls":
                    v = h[:, 0]
                else:
                    m = attention_mask.unsqueeze(-1).to(h.dtype)
                    v = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)
                if self.dense is not None:
                    v = self.dense(v)
                if self.head is not None:
                    v = self.head(v)
                return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return _M().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["linear", "mlp", "none"], default="linear")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    import torch
    import encoders
    from table import get_tokenizer

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"                     # export on CPU: deterministic and avoids CUDA-op fallbacks
    spec = encoders.active()
    tok = get_tokenizer()
    t0 = time.time()

    out = {"_note": __doc__.strip().splitlines()[0],
           "encoder": {"name": spec.name, "repo": spec.repo, "revision": spec.revision,
                       "trust_remote_code": spec.trust_remote_code, "dim": spec.dim},
           "head": a.head, "opset": a.opset,
           "criterion": f"one file, plain nodes, and parity within cosine {COS_TOL} / "
                        f"max-abs {MAXABS_TOL} against the torch forward of the SAME module",
           "stages": {}}

    try:
        model = Composed(spec, a.head, device)
        out["stages"]["compose"] = "ok"
    except Exception as e:                                          # noqa: BLE001
        out["stages"]["compose"] = f"FAILED: {type(e).__name__}: {e}"
        out["pass"] = False
        out["verdict"] = "composition failed before export was attempted"
        probe_guard.write_result(OUT, out, "B6-pre")
        print(json.dumps(out["stages"], indent=2))
        return 1

    enc = tok(TEXTS, padding=True, truncation=True, max_length=512, return_tensors="pt")
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    with torch.no_grad():
        ref = model(ids, mask).cpu().numpy()
    out["stages"]["torch_forward"] = f"ok, {ref.shape}"

    path = EXPORT_DIR / f"{spec.name}-doc-{a.head}.onnx"
    try:
        torch.onnx.export(
            model, (ids, mask), str(path),
            input_names=["input_ids", "attention_mask"], output_names=["embedding"],
            dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                          "embedding": {0: "b"}},
            opset_version=a.opset, do_constant_folding=True)
        out["stages"]["export"] = f"ok, {path.stat().st_size/1e6:.1f} MB, one file"
    except Exception as e:                                          # noqa: BLE001
        out["stages"]["export"] = f"FAILED: {type(e).__name__}: {str(e)[:600]}"
        out["pass"] = False
        out["verdict"] = ("stella does not export under this path. E3's condition is not met by "
                          "this route; D1's registered no-survivor outcome applies unless another "
                          "export route is registered and tried.")
        probe_guard.write_result(OUT, out, "B6-pre")
        print(json.dumps(out["stages"], indent=2))
        return 1

    try:
        import onnx
        m = onnx.load(str(path))
        ops = {}
        for n in m.graph.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        custom = sorted({n.op_type for n in m.graph.node
                         if n.domain not in ("", "ai.onnx", "ai.onnx.ml")})
        out["stages"]["graph"] = {"n_nodes": len(m.graph.node), "op_histogram": ops,
                                  "custom_domain_ops": custom}
        out["plain_nodes_only"] = not custom
    except Exception as e:                                          # noqa: BLE001
        out["stages"]["graph"] = f"could not inspect: {type(e).__name__}: {e}"
        out["plain_nodes_only"] = None

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(["embedding"], {"input_ids": ids.cpu().numpy(),
                                       "attention_mask": mask.cpu().numpy()})[0]
        cos = float(np.min(np.sum(ref * got, axis=1)
                           / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1) + 1e-12)))
        maxabs = float(np.max(np.abs(ref - got)))
        out["parity"] = {"min_cosine": cos, "max_abs": maxabs,
                         "cos_tol": COS_TOL, "maxabs_tol": MAXABS_TOL,
                         "pass": bool(1 - cos <= COS_TOL and maxabs <= MAXABS_TOL)}
        out["stages"]["runtime"] = "ok"
    except Exception as e:                                          # noqa: BLE001
        out["stages"]["runtime"] = f"FAILED: {type(e).__name__}: {str(e)[:400]}"
        out["parity"] = {"pass": False}

    out["pass"] = bool(out.get("parity", {}).get("pass") and out.get("plain_nodes_only") is not False)
    out["verdict"] = ("E3's condition is MET by this route: one file, plain nodes, parity within "
                      "tolerance. D1 stays on the Stage-S menu and B6's quality arm may be "
                      "registered." if out["pass"] else
                      "E3's condition is NOT met by this route. D1's registered no-survivor "
                      "outcome applies unless another route is registered and tried.")
    out["seconds"] = round(time.time() - t0, 1)
    probe_guard.write_result(OUT, out, "B6-pre")
    print(json.dumps({"stages": out["stages"], "parity": out.get("parity"),
                      "plain_nodes_only": out.get("plain_nodes_only"),
                      "pass": out["pass"], "verdict": out["verdict"]}, indent=2, default=str))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
