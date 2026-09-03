"""Standalone query encoder for the `zero` lookup table. numpy + tokenizers, no torch.

This is the whole query path. It is a vocab x dim table of vectors: tokenize the query,
gather one row per token, take a count-saturated weighted mean, L2 normalize. There is no
transformer and no matrix multiply -- encoding a query is a gather and a sum.

The output lives in the document space of the frozen teacher (NovaSearch/stella_en_400M_v5,
revision pinned in config.json), so it is only meaningful against document vectors produced
by that exact encoder. Cosine similarity is the score.

Conformance: this file reproduces the frozen training-time query path (m7src/table.py
`encode_pooled`) to < 1e-5 max-abs on the release fixtures; see m11/release/test_conformance.py.
"""
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

EPS = 1e-6


class ZeroQueryEncoder:
    """The released query encoder.

    variant: "int8" is the artifact the published numbers were measured on (31 MB);
             "fp16" is the same table before quantization (62 MB), included for reference --
             int8 was measured quality-free against it (upper bound 0.00013 nDCG@10).
    """

    def __init__(self, model_dir, variant="int8"):
        d = Path(model_dir)
        self.config = json.loads((d / "config.json").read_text())
        pre = self.config["preproc"]
        if pre["pool_mode"] != "sqrt" or pre["prefix"] != "" or not pre["add_special_tokens"]:
            raise ValueError(f"this file implements the frozen M7 rule only, got {pre}")
        self.max_length = int(pre["max_length"])
        self.fallback_id = int(self.config["fallback_token_id"])

        z = np.load(d / "model.npz")
        if variant == "int8":
            self.rows = z["rows_int8"].astype(np.float32) * z["int8_scale"][:, None]
        elif variant == "fp16":
            self.rows = z["rows_fp16"].astype(np.float32)
        else:
            raise ValueError(f"variant must be 'int8' or 'fp16', got {variant!r}")
        self.variant = variant

        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.max_length)
        # stella's tokenizer.json ships with padding-to-512 enabled. Padding would put ~500
        # [PAD] rows into every bag; the frozen path (transformers, padding off) never sees one.
        self.tokenizer.no_padding()
        self._fallback = self._normalize(self.rows[self.fallback_id])

    @property
    def dim(self):
        return self.rows.shape[1]

    @staticmethod
    def _normalize(v):
        n = float(np.linalg.norm(v))
        if n <= EPS:                       # degenerate row: fall back to e_0
            e0 = np.zeros_like(v)
            e0[0] = 1.0
            return e0
        return v / n

    def encode(self, texts):
        """texts: str or list[str] -> float32 array (n, dim), L2-normalized."""
        if isinstance(texts, str):
            texts = [texts]
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, enc in enumerate(self.tokenizer.encode_batch(texts)):
            out[i] = self._encode_ids(enc.ids)
        return out

    def _encode_ids(self, ids):
        if not ids:
            return self._fallback
        uniq, counts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
        # count saturation: a token seen c times carries TOTAL weight sqrt(c), not c. The
        # denominator cancels under the final L2 normalize; it is kept so the intermediate
        # stays in the released rule's range and the degeneracy threshold means the same thing.
        w = np.sqrt(counts, dtype=np.float32)
        vec = (self.rows[uniq] * w[:, None]).sum(0) / max(float(w.sum()), EPS)
        if float(np.linalg.norm(vec)) <= EPS:
            return self._fallback
        return self._normalize(vec).astype(np.float32)
