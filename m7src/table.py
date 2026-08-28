"""The lookup-table query encoder: the released artifact and the one query-preprocessing rule.

Query time is tokenize -> gather rows -> weighted average -> L2 normalize. No transformer.

Preprocessing is frozen as a `Preproc` value object so the training path, the dev path, the
final scorer, and the released model card all share one definition. Double prefix application
is a hard error, not a silent fix.
"""
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer

from teacher import QUERY_PREFIX, TEACHER, TEACHER_REV

EPS = 1e-6
# The bge/BERT WordPiece [CLS] id. Row 0 is [PAD], NOT [CLS] -- an earlier version of the
# degenerate-query fallback documented itself as "the [CLS] row" while actually using row 0, and
# the conformance test compared encode("") against fallback_vector() itself, so the circularity
# hid the mismatch. The model card states this id.
CLS_ID = 101


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

    def __init__(self, rows_init, weight_init=None, learned_weights=True, fallback_id=CLS_ID):
        super().__init__()
        self.fallback_id = int(fallback_id)
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
        """Deterministic behavior for empty queries / near-zero-norm sums: the [CLS] row
        (id 101 in this vocab), L2-normalized; e_0 if that row is itself degenerate."""
        r = self.rows[min(self.fallback_id, self.rows.shape[0] - 1)]
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


# ---- count saturation (capacity lever #4, protocol in m7/LEDGER.md 2026-08-27) -----------
#
# `absorb_check` proves multiplicity-dependent pooling is one of only two query-side transforms
# that are NOT absorbable into the rows, so this is real capacity and costs no bytes. It is a
# PROBE for now: `Preproc` deliberately does not carry the mode, because adding a field would
# change the frozen preprocessing fingerprint of every table already on disk. If a mode is
# adopted, Preproc grows `pool_mode` and the artifacts are re-saved with it.
POOL_MODES = ("mean", "binary", "cap2", "sqrt")


def occurrence_weights(ids_list, mode, device="cpu"):
    """Per-occurrence weights so a token appearing c times contributes TOTAL weight f(c):
    mean f(c)=c (the current released rule) | binary 1 | cap2 min(2,c) | sqrt sqrt(c).
    The weighted-mean denominator is the same sum, so length normalization is unchanged."""
    if mode not in POOL_MODES:
        raise KeyError(f"unknown pool mode {mode!r}; known {POOL_MODES}")
    if mode == "mean":
        return torch.ones(sum(len(x) for x in ids_list), dtype=torch.float32, device=device)
    ws = []
    for ids in ids_list:
        _, inv, c = np.unique(np.asarray(ids, dtype=np.int64), return_inverse=True,
                             return_counts=True)
        cc = c[inv].astype(np.float32)
        ws.append({"binary": 1.0 / cc, "cap2": np.minimum(2.0, cc) / cc,
                   "sqrt": np.sqrt(cc) / cc}[mode])
    flat = np.concatenate(ws) if ws else np.zeros(0, dtype=np.float32)
    return torch.from_numpy(flat.astype(np.float32)).to(device)


@torch.no_grad()
def encode_pooled(model: QueryTable, texts, pre: Preproc, mode="mean", tok=None, batch=1024,
                  device=None):
    """`QueryTable.encode` with a count-saturation rule applied on top of the table's own
    per-token weights. mode='mean' reproduces `forward` (asserted in the probe's smoke)."""
    tok = tok or get_tokenizer()
    device = device or model.rows.device
    base_w = model.token_weights()
    out = np.empty((len(texts), model.dim), dtype=np.float32)
    for lo in range(0, len(texts), batch):
        ids = tokenize(tok, texts[lo:lo + batch], pre)
        flat, off, lens = ragged(ids, device)
        psw = occurrence_weights(ids, mode, device=device)
        if base_w is not None:
            psw = psw * base_w[flat]
        s = F.embedding_bag(flat, model.rows, off, mode="sum", per_sample_weights=psw)
        denom = torch.zeros_like(s[:, 0]).index_add_(0, _bag_index(off, lens), psw)
        mean = s / denom.clamp_min(EPS).unsqueeze(1)
        norm = mean.norm(dim=1, keepdim=True)
        fb = model.fallback_vector().to(mean.dtype)
        out[lo:lo + batch] = torch.where(norm > EPS, mean / norm.clamp_min(EPS),
                                         fb.expand_as(mean)).float().cpu().numpy()
    return out


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


def apply_unseen_policy(rows, updates, policy, init_rows=None, min_updates=1):
    """What a row that training never (or barely) touched should contain.

    The six are scientific, biomedical, financial and argumentative; the training mix is
    Wikipedia and e-commerce. Rarely-updated rows are therefore exactly the rows the six are
    most likely to hit, and leaving them at a differently-scaled init makes those queries land
    in a different regime from trained ones. Policies are compared on dev.
      init             leave at initialization (the default the mandate names)
      mean_of_trained  replace with the mean of the trained rows (same regime, no information)
      zero             contribute nothing to the bag
    """
    rows = np.array(rows, dtype=np.float32, copy=True)
    cold = np.asarray(updates) < min_updates
    if policy == "init":
        if init_rows is not None:
            rows[cold] = np.asarray(init_rows, dtype=np.float32)[cold]
    elif policy == "mean_of_trained":
        warm = ~cold
        if warm.any():
            rows[cold] = rows[warm].mean(0)
    elif policy == "zero":
        rows[cold] = 0.0
    else:
        raise KeyError(policy)
    return rows, int(cold.sum())


def save_table(path, model: QueryTable, pre: Preproc, meta=None, updates=None, rows_override=None):
    rows = (np.asarray(rows_override, dtype=np.float32) if rows_override is not None
            else model.rows.detach().float().cpu().numpy())
    w = model.token_weights()
    w = None if w is None else w.detach().float().cpu().numpy()
    q, scale = quantize_int8(rows)
    np.savez(path, rows_fp16=rows.astype(np.float16), rows_int8=q, int8_scale=scale,
             token_weights=(np.zeros(0, dtype=np.float32) if w is None else w),
             updates=(np.zeros(0, dtype=np.int64) if updates is None
                      else np.asarray(updates, dtype=np.int64)))
    (path.parent / (path.stem + ".meta.json")).write_text(json.dumps(
        {"preproc": asdict(pre), "preproc_fingerprint": pre.fingerprint(),
         "teacher": TEACHER, "teacher_revision": TEACHER_REV,
         "vocab": int(rows.shape[0]), "dim": int(rows.shape[1]),
         "learned_weights": bool(w is not None), **(meta or {})}, indent=1, sort_keys=True))


def save_release(path, model: QueryTable, pre: Preproc, meta=None, device="cuda"):
    """The RELEASE export (Codex MINOR-int8-weights): learned per-token weights are FOLDED into
    the rows, so the shipped int8 artifact is self-contained -- no fp32 weight vector multiplies
    the quantized rows at query time. Folding is exact for retrieval: per-row absmax int8 codes
    are scale-invariant (the scale just multiplies by w), and the weight-sum division in forward
    is a per-query positive scalar, absorbed by the final L2 normalize. Training checkpoints keep
    save_table's unfolded shape (a folded table cannot resume training).

    Self-verifying: encodes a fixture through the live model and the loaded artifact and refuses
    to write on max-abs deviation > 5e-3 (fp16 storage of folded rows is the only difference).
    G4 int8-equivalence must be measured on THIS artifact, not the training checkpoint."""
    w = model.token_weights()
    rows = model.rows.detach().float().cpu().numpy()
    if w is not None:
        rows = w.detach().float().cpu().numpy()[:, None] * rows
    folded = QueryTable(rows, learned_weights=False,
                        fallback_id=model.fallback_id).to(device).eval()
    fixture = ["what is a lookup table", "protein folding market impact",
               "argue both sides of a covid tax", "zzzqx", ""]
    a = model.encode(fixture, pre)
    b = folded.encode(fixture, pre)
    dev = float(np.abs(a - b).max())
    if dev > 5e-3:
        raise AssertionError(f"folded release deviates from the live model: max abs {dev}")
    save_table(path, folded, pre, meta={**(meta or {}), "weights_folded": True,
                                        "fold_max_abs_dev": dev})
    # review #2 BLOCKER 2 addendum: verify the SERIALIZED artifact, not just the in-memory fold
    reloaded = load_table(path, variant="fp16", device=device)
    dev_ser = float(np.abs(folded.encode(fixture, pre) - reloaded.encode(fixture, pre)).max())
    if dev_ser > 5e-3:
        raise AssertionError(f"serialized fp16 release deviates from the fold: max abs {dev_ser}")
    return dev


def ensure_release(npz_path, device="cuda"):
    """The released (weights-folded) sibling of a training checkpoint, created on demand and
    cached next to it. Everything that judges or freezes 'the released artifact' -- G4, freeze,
    the final run -- must go through this, never the raw training npz (review #2 BLOCKER 2)."""
    npz_path = Path(npz_path)
    rel = npz_path.with_name(npz_path.stem + ".release.npz")
    if rel.exists() and rel.stat().st_mtime >= npz_path.stat().st_mtime:
        return rel
    meta = read_meta(npz_path)
    m = load_table(npz_path, variant="fp16", device=device)
    save_release(rel, m, Preproc(**meta["preproc"]), meta={"source": npz_path.name}, device=device)
    return rel


def load_table(path, variant="fp16", device="cuda"):
    z = np.load(path)
    rows = z["rows_fp16"].astype(np.float32) if variant == "fp16" else \
        dequantize_int8(z["rows_int8"], z["int8_scale"])
    w = z["token_weights"]
    m = QueryTable(rows, weight_init=(w if w.size else None), learned_weights=bool(w.size))
    return m.to(device).eval()


def read_meta(path):
    """The table's own recipe. final_run reads preprocessing from here, never from a CLI flag."""
    return json.loads((Path(path).parent / (Path(path).stem + ".meta.json")).read_text())
