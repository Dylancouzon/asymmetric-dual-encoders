"""The standalone M17 numpy query loader: eager fp32 rows vs resident int8 codes/scales.

Same query path as the shipped v1 encoder (`m11/release/zero_encoder.py`) — tokenize, gather
one row per unique token id, sqrt-count weighted mean, L2 normalize — with one implementation
difference under test:

  * `eager_fp32`   the whole table is expanded to float32 at load, as v1 does today;
  * `resident_int8` the int8 codes and per-row scales stay resident and ONLY the query's unique
    rows are dequantized, per query.

Both modes do float32 arithmetic in the same order over the same `np.unique`-sorted ids and
counts, keep [CLS]/[SEP] as ordinary rows, keep the empty/degenerate fallback and keep the
frozen 512-token truncation, so `parity()` is expected to be exact — the registry's
`int8_resident_loading.parity_max_abs` is 1e-6, not a tolerance for a different rule.

**This file does not edit the frozen M11 release implementation** and changes no public API.

RSS is reported separately from weight bytes: `weight_bytes` is what the arrays cost, while
`rss_kb` is what the process holds, which also carries numpy, the tokenizer and the interpreter.
Array-size arithmetic is not an RSS measurement (m17/CODEMAP.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

EPS = 1e-6
MODES = ("eager_fp32", "resident_int8")


class M17QueryEncoder:
    """numpy + tokenizers, no torch. `mode` selects the loading strategy under comparison."""

    def __init__(self, model_dir, variant="int8", mode="resident_int8"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        d = Path(model_dir)
        self.config = json.loads((d / "config.json").read_text())
        pre = self.config["preproc"]
        if pre["pool_mode"] != "sqrt" or pre["prefix"] != "" or not pre["add_special_tokens"]:
            raise ValueError(f"this loader implements the frozen M7 query rule only, got {pre}")
        self.max_length = int(pre["max_length"])
        self.fallback_id = int(self.config["fallback_token_id"])
        self.variant, self.mode = variant, mode

        z = np.load(d / "model.npz")
        if variant == "int8":
            self.codes = z["rows_int8"]
            self.scales = z["int8_scale"].astype(np.float32)
        elif variant == "fp16":
            if mode == "resident_int8":
                raise ValueError("resident_int8 needs the int8 variant's codes and scales")
            self.codes, self.scales = None, None
        else:
            raise ValueError(f"variant must be 'int8' or 'fp16', got {variant!r}")

        if mode == "eager_fp32":
            self.rows = (z["rows_fp16"].astype(np.float32) if variant == "fp16"
                         else self.codes.astype(np.float32) * self.scales[:, None])
            self.codes = self.scales = None
            n_rows = self.rows.shape[0]
        else:
            self.rows = None
            n_rows = self.codes.shape[0]

        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        n = self.tokenizer.get_vocab_size(with_added_tokens=True)
        if n != n_rows:
            raise ValueError(f"tokenizer has {n} tokens but the table has {n_rows} rows; a token "
                             "id outside the table would index off the end")
        self.tokenizer.enable_truncation(max_length=self.max_length)
        # stella's tokenizer.json ships padding-to-512 enabled; padding would put ~500 [PAD]
        # rows into every bag. The frozen path never sees one.
        self.tokenizer.no_padding()
        self._fallback = self._normalize(self._gather(np.asarray([self.fallback_id]))[0])

    @property
    def dim(self):
        return int(self.rows.shape[1] if self.rows is not None else self.codes.shape[1])

    @property
    def weight_bytes(self):
        """Bytes the TABLE occupies in this mode. Not the process RSS (see `rss_kb`)."""
        if self.rows is not None:
            return int(self.rows.nbytes)
        return int(self.codes.nbytes + self.scales.nbytes)

    def _gather(self, uniq):
        """The only difference between the two modes: when dequantization happens."""
        if self.rows is not None:
            return self.rows[uniq]
        return self.codes[uniq].astype(np.float32) * self.scales[uniq, None]

    @staticmethod
    def _normalize(v):
        n = float(np.linalg.norm(v))
        if n <= EPS:
            e0 = np.zeros_like(v)
            e0[0] = 1.0
            return e0
        return v / n

    def encode(self, texts):
        """texts: str or list[str] -> float32 (n, dim), L2-normalized."""
        if isinstance(texts, str):
            texts = [texts]
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, enc in enumerate(self.tokenizer.encode_batch(list(texts))):
            out[i] = self._encode_ids(enc.ids)
        return out

    def _encode_ids(self, ids):
        if not ids:
            return self._fallback
        uniq, counts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
        w = np.sqrt(counts, dtype=np.float32)
        vec = (self._gather(uniq) * w[:, None]).sum(0) / max(float(w.sum()), EPS)
        if float(np.linalg.norm(vec)) <= EPS:
            return self._fallback
        return self._normalize(vec).astype(np.float32)


def rss_kb():
    """Process RSS in KiB, or None where /proc is unavailable. Measured, never derived."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def parity(model_dir, texts=None, float_rows=None, tol=None):
    """Resident-int8 vs eager-fp32 on the same artifact, and both vs the exported float rows.

    `float_rows` is the pre-quantization effective float32 table. The int8 comparison against it
    is a QUANTIZATION measurement (it will not be 1e-6); the loader-vs-loader comparison is the
    parity the registry bounds.
    """
    from common import registry
    tol = tol if tol is not None else float(registry()["int8_resident_loading"]["parity_max_abs"])
    texts = list(texts or DEFAULT_FIXTURES)
    a = M17QueryEncoder(model_dir, variant="int8", mode="eager_fp32")
    b = M17QueryEncoder(model_dir, variant="int8", mode="resident_int8")
    va, vb = a.encode(texts), b.encode(texts)
    dev = float(np.abs(va - vb).max())
    out = {"loader_parity_max_abs": dev, "tolerance": tol, "pass": dev <= tol,
           "weight_bytes": {"eager_fp32": a.weight_bytes, "resident_int8": b.weight_bytes},
           "rss_kb_after_load": rss_kb(), "n_fixtures": len(texts)}
    if float_rows is not None:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
        ref = _reference_encode(np.asarray(float_rows, dtype=np.float32),
                                Path(model_dir) / "tokenizer.json", cfg, texts)
        out["vs_exported_float_rows_max_abs"] = float(np.abs(ref - va).max())
        out["_note_quantization"] = ("the float-row comparison measures int8 quantization, not "
                                     "loader parity; only loader_parity_max_abs is bounded by "
                                     "int8_resident_loading.parity_max_abs")
    return out


def _reference_encode(rows, tokenizer_json, cfg, texts):
    tok = Tokenizer.from_file(str(tokenizer_json))
    tok.enable_truncation(max_length=int(cfg["preproc"]["max_length"]))
    tok.no_padding()
    fb_id = int(cfg["fallback_token_id"])
    out = np.empty((len(texts), rows.shape[1]), dtype=np.float32)
    for i, enc in enumerate(tok.encode_batch(list(texts))):
        ids = enc.ids
        if not ids:
            out[i] = M17QueryEncoder._normalize(rows[fb_id])
            continue
        uniq, counts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
        w = np.sqrt(counts, dtype=np.float32)
        v = (rows[uniq] * w[:, None]).sum(0) / max(float(w.sum()), EPS)
        out[i] = (M17QueryEncoder._normalize(v) if float(np.linalg.norm(v)) > EPS
                  else M17QueryEncoder._normalize(rows[fb_id]))
    return out


DEFAULT_FIXTURES = [
    "what is a lookup table", "", "   ", "the the the the the the",
    "s3 bucket policy", "k8s ingress", "s3://bucket/key", "s3_client s3-compatible xs3",
    "C++ .NET C#", "[CLS] [SEP] [UNK] [MASK]", "[PAD] [PAD] hello",
    "electroencephalographically " + "a" * 120, "​ \t\n",
    " ".join(["retrieval augmented generation"] * 400),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    rep = parity(args.bundle)
    print(json.dumps(rep, indent=1, sort_keys=True))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1, sort_keys=True))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
