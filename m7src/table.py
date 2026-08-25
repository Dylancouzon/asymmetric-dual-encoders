"""The lookup-table query encoder: the released artifact and the one query-preprocessing rule.

Query time is tokenize -> gather rows -> weighted average -> L2 normalize. No transformer.

Preprocessing is frozen as a `Preproc` value object so the training path, the dev path, the
final scorer, and the released model card all share one definition. Double prefix application
is a hard error, not a silent fix.
"""
import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from teacher import QUERY_PREFIX, TEACHER, TEACHER_REV

EPS = 1e-6


@dataclass(frozen=True)
class Preproc:
    """The frozen query-preprocessing rule. `prefix` has exactly two mandated values."""
    prefix: str = ""
    add_special_tokens: bool = True
    max_length: int = 512

    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


NO_PREFIX = Preproc(prefix="")
WITH_PREFIX = Preproc(prefix=QUERY_PREFIX)


def get_tokenizer(model_id=TEACHER, revision=TEACHER_REV):
    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def tokenize(tok, texts, pre: Preproc):
    """Byte-for-byte conformance: ids are exactly tok(pre.prefix + text). Multiplicity kept."""
    if pre.prefix:
        for t in texts:
            if t.startswith(pre.prefix):
                raise ValueError("prefix already applied to the input text; double application "
                                 "is forbidden by the M7 preprocessing rule")
    enc = tok([pre.prefix + t for t in texts], add_special_tokens=pre.add_special_tokens,
              truncation=True, max_length=pre.max_length)
    return enc["input_ids"]


def ragged(ids_list, device="cpu"):
    """-> (flat_ids int64, offsets int64) for F.embedding_bag; no padding is ever introduced."""
    lens = [len(x) for x in ids_list]
    flat = torch.tensor([i for x in ids_list for i in x], dtype=torch.long, device=device)
    off = torch.zeros(len(ids_list), dtype=torch.long, device=device)
    if len(ids_list) > 1:
        off[1:] = torch.tensor(lens[:-1], dtype=torch.long, device=device).cumsum(0)
    return flat, off, torch.tensor(lens, dtype=torch.long, device=device)


class QueryTable(nn.Module):
    """vocab x dim rows + optional positive bounded per-token scalar weights (softplus)."""

    def __init__(self, rows_init, weight_init=None, learned_weights=True):
        super().__init__()
        self.rows = nn.Parameter(torch.as_tensor(rows_init, dtype=torch.float32).clone())
        self.learned_weights = learned_weights
        v = self.rows.shape[0]
        if learned_weights:
            w0 = torch.ones(v) if weight_init is None else torch.as_tensor(weight_init, dtype=torch.float32)
            # invert softplus so softplus(w_raw) == w0 at init
            self.w_raw = nn.Parameter(torch.log(torch.expm1(w0.clamp_min(1e-4))))
        else:
            self.register_buffer("w_raw", torch.zeros(0))

    @property
    def vocab(self):
        return self.rows.shape[0]

    @property
    def dim(self):
        return self.rows.shape[1]

    def token_weights(self):
        if not self.learned_weights:
            return None
        return F.softplus(self.w_raw)

    def forward(self, flat_ids, offsets, lens):
        """Weighted mean over token occurrences, then L2 normalize. Length-normalized by
        construction (divide by the weight sum), so long queries are not down-weighted."""
        w = self.token_weights()
        psw = None if w is None else w[flat_ids]
        s = F.embedding_bag(flat_ids, self.rows, offsets, mode="sum", per_sample_weights=psw)
        denom = (lens.to(s.dtype) if psw is None
                 else torch.zeros_like(s[:, 0]).index_add_(0, _bag_index(offsets, lens), psw))
        mean = s / denom.clamp_min(EPS).unsqueeze(1)
        norm = mean.norm(dim=1, keepdim=True)
        fb = self.fallback_vector().to(mean.dtype)
        out = torch.where(norm > EPS, mean / norm.clamp_min(EPS), fb.expand_as(mean))
        return out

    def fallback_vector(self):
        """Deterministic behavior for empty queries / near-zero-norm sums: the [CLS] row,
        L2-normalized; e_0 if that row is itself degenerate."""
        r = self.rows[0]
        n = r.norm()
        e0 = torch.zeros(self.dim, device=self.rows.device, dtype=self.rows.dtype)
        e0[0] = 1.0
        # torch.where, not a Python branch: keeps this graph-safe and avoids a device sync
        return torch.where(n > EPS, r / n.clamp_min(EPS), e0)

    @torch.no_grad()
    def encode(self, texts, pre: Preproc, tok=None, batch=1024, device=None):
        tok = tok or get_tokenizer()
        device = device or self.rows.device
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for lo in range(0, len(texts), batch):
            ids = tokenize(tok, texts[lo:lo + batch], pre)
            f, o, l = ragged(ids, device)
            out[lo:lo + batch] = self(f, o, l).float().cpu().numpy()
        return out


def _bag_index(offsets, lens):
    """Row index per flat token, for summing per_sample_weights per bag."""
    return torch.repeat_interleave(torch.arange(len(lens), device=lens.device), lens)


# ---- released artifact: quantization + (de)serialization -------------------------------

def quantize_int8(rows):
    """Symmetric per-row absmax, no calibration set (the M3 recipe, quality-free for LR)."""
    r = np.asarray(rows, dtype=np.float32)
    scale = np.abs(r).max(axis=1) / 127.0
    scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    q = np.rint(r / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q, scale):
    return q.astype(np.float32) * scale[:, None]


def save_table(path, model: QueryTable, pre: Preproc, meta=None):
    rows = model.rows.detach().float().cpu().numpy()
    w = model.token_weights()
    w = None if w is None else w.detach().float().cpu().numpy()
    q, scale = quantize_int8(rows)
    np.savez(path, rows_fp16=rows.astype(np.float16), rows_int8=q, int8_scale=scale,
             token_weights=(np.zeros(0, dtype=np.float32) if w is None else w))
    (path.parent / (path.stem + ".meta.json")).write_text(json.dumps(
        {"preproc": asdict(pre), "preproc_fingerprint": pre.fingerprint(),
         "teacher": TEACHER, "teacher_revision": TEACHER_REV,
         "vocab": int(rows.shape[0]), "dim": int(rows.shape[1]),
         "learned_weights": bool(w is not None), **(meta or {})}, indent=1, sort_keys=True))


def load_table(path, variant="fp16", device="cuda"):
    z = np.load(path)
    rows = z["rows_fp16"].astype(np.float32) if variant == "fp16" else \
        dequantize_int8(z["rows_int8"], z["int8_scale"])
    w = z["token_weights"]
    m = QueryTable(rows, weight_init=(w if w.size else None), learned_weights=bool(w.size))
    return m.to(device).eval()
