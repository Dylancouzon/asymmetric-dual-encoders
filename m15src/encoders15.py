"""M15 query encoders on the Mac: Zero (NumPy) and ONNX Runtime CPU for the transformers.

One ORT session setup for every transformer (E1 protocol, `m15/MEASUREMENTS.md`): intra-op
threads 4, inter-op 1, full graph optimization, dynamic padding, fp32.
"""
import json
import sys
from pathlib import Path

import numpy as np

import roster_ids as I
from common import REPO, sha_file

NANO_REPO, NANO_REVISION = "Qdrant/constella-nano", "6bb167dc6f60d3992602235b8e8aaa374a309168"
NANO_ONNX_SHA = "9ba0acf57b71dc31bc5512c5445078a797fa51cf3e85587d6b8a506bfc55dbc2"


def session(path, threads=4):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class OnnxEncoder:
    """Tokenize -> ONNX -> (mean or cls pooling, unless the graph pools) -> L2 normalize."""

    def __init__(self, name, directory, pooling, prefix="", max_length=512, expect_sha=None,
                 threads=4):
        from tokenizers import Tokenizer
        d = Path(directory)
        self.name, self.prefix, self.pooling = name, prefix, pooling
        self.onnx_sha256 = sha_file(d / "model.onnx")
        if expect_sha and self.onnx_sha256 != expect_sha:
            raise RuntimeError(f"{name}: model.onnx hash {self.onnx_sha256} != {expect_sha}")
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_length)
        self.tok.no_padding()
        self.sess = session(d / "model.onnx", threads)
        self.inputs = [i.name for i in self.sess.get_inputs()]

    def _one(self, text):
        enc = self.tok.encode(self.prefix + text)
        ids = np.asarray([enc.ids], dtype=np.int64)
        mask = np.ones_like(ids)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(None, feed)[0].astype(np.float32)
        if out.ndim == 3:
            out = out[0].mean(0) if self.pooling == "mean" else out[0, 0]
        else:
            out = out[0]
        return out / max(float(np.linalg.norm(out)), 1e-12)

    def encode(self, texts):
        texts = [texts] if isinstance(texts, str) else list(texts)
        return np.stack([self._one(t) for t in texts]).astype(np.float32)


class ZeroEncoder:
    def __init__(self):
        from huggingface_hub import snapshot_download
        sys.path.insert(0, str(REPO / "m11" / "release"))
        from zero_encoder import ZeroQueryEncoder
        self.dir = Path(snapshot_download(I.ZERO_HUB_REPO, revision=I.ZERO_HUB_REVISION))
        frozen = json.loads((REPO / "m7" / "FREEZE.json").read_text())["table_sha256"]
        if sha_file(self.dir / "model.npz") != frozen:
            raise RuntimeError("Zero table differs from the M7 freeze")
        self.model = ZeroQueryEncoder(self.dir, variant="int8")
        self.name = "zero"

    def encode(self, texts):
        return self.model.encode(texts)


def nano():
    from huggingface_hub import snapshot_download
    d = snapshot_download(NANO_REPO, revision=NANO_REVISION)
    return OnnxEncoder("nano", d, "mean", expect_sha=NANO_ONNX_SHA)


STELLA_ONNX_REPO = "Qdrant/stella-en-400M-v5-doc-onnx"
STELLA_ONNX_REVISION = "293ec490f8e8c56ca468f84c84b8095bac954254"
EXPORTS = REPO / "work" / "m15" / "onnx"


def stella_query():
    from huggingface_hub import snapshot_download
    d = snapshot_download(STELLA_ONNX_REPO, revision=STELLA_ONNX_REVISION)
    return OnnxEncoder("stella-query", d, "mean", prefix=I.STELLA_QUERY_PROMPT)


def export_sentence_transformer(name, repo, revision):
    """Export the whole SentenceTransformer pipeline (pooling, Dense, Normalize) to one graph."""
    import torch
    from sentence_transformers import SentenceTransformer
    out = EXPORTS / name
    out.mkdir(parents=True, exist_ok=True)
    st = SentenceTransformer(repo, revision=revision, device="cpu",
                             model_kwargs={"dtype": torch.float32}).eval()

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.st = st

        def forward(self, input_ids, attention_mask):
            return self.st({"input_ids": input_ids,
                            "attention_mask": attention_mask})["sentence_embedding"]

    batch = st.tokenizer(["a short query", "a somewhat longer query text"], padding=True,
                         return_tensors="pt")
    with torch.inference_mode():
        torch.onnx.export(Wrap().eval(), (batch["input_ids"], batch["attention_mask"]),
                          str(out / "model.onnx"), opset_version=17, dynamo=False,
                          input_names=["input_ids", "attention_mask"], output_names=["embedding"],
                          dynamic_axes={"input_ids": {0: "b", 1: "s"},
                                        "attention_mask": {0: "b", 1: "s"},
                                        "embedding": {0: "b"}})
    st.tokenizer.save_pretrained(str(out))
    return st


def bge():
    return OnnxEncoder("bge-small", EXPORTS / "bge-small", "cls", prefix=I.BGE_PREFIX)


def leaf():
    return OnnxEncoder("leaf-query", EXPORTS / "leaf-query", "mean",
                       prefix="Represent this sentence for searching relevant passages: ")


def make(name):
    return {"zero": ZeroEncoder, "nano": nano, "stella-query": stella_query, "bge-small": bge,
            "leaf-query": leaf}[name]()
