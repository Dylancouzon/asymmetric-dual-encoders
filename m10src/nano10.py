"""The M10 student: a PER-TOKEN head over concatenated layers, pooled AFTER the head.

M9's student was `mean_t(h_t) -> Linear`, one layer wide. M10's diagnosis (`m9/FINDINGS.md`,
`m10/PLANNING.md` §9) is that a 384-wide linear head cannot be pushed past ~90-93% retention once
queries are diverse, so the head widens to three concatenated layers (1152-d; four = 1536-d,
family G) and moves BEFORE the pool:

    training      normalize(mean_t(head(x_t)))
    serving       the ONNX graph emits head(x_t) per token; fastembed does the masked mean and
                  the normalize itself

Those are the same function, and for a LINEAR head that is not an approximation -- pooling and the
head commute, so `head(mean_t x_t) == mean_t(head(x_t))` exactly. Two consequences the code below
depends on, both asserted by `m10src/test_nano10.py` rather than believed:

1. **The ridge warm start can be solved on POOLED features.** Fitting per-token would be 1152-d
   features x every token; fitting on the pooled 1152-d vector gives the identical head. That is
   why the M9 warm-start machinery ports unchanged to a per-token architecture.
2. **The serving parity is exact.** `m10src/student_parity.py` measures it end to end through
   fastembed (min-cos >= 0.99999988 on all six family-F heads) -- this module only has to not
   break the algebra.

G-MLP (family G) adds a per-token rank-192 GELU correction to the same linear path. It stays
exportable for the same reason -- it is applied per token, before the pool -- but it does NOT
commute with pooling, so its warm start is the three-solve recipe registered in
`m10/screen_registry.json` `warm_start.G-MLP`, not a pooled ridge.

`m9src/nano.py` is a `guard9` "train"-scope file and is not touched.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# feature layers per student (§Recipe): the last layer, then two-thirds, then one-third of depth;
# family G's fourth arm adds one more. Hard-coded rather than derived, because these exact indices
# are what the parity artifacts were measured on.
LAYERS = {
    "bge-small":  {3: (12, 8, 4), 4: (12, 8, 4, 2)},
    "MiniLM-L6":  {3: (6, 4, 2),  4: (6, 4, 2, 1)},
    "MiniLM-L12": {3: (12, 8, 4), 4: (12, 8, 4, 2)},
}
REPOS = {"bge-small": "BAAI/bge-small-en-v1.5",
         "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
         "MiniLM-L12": "sentence-transformers/all-MiniLM-L12-v2"}
OUT_DIM = 1024
CAP = 35_000_000


class MLPHead(nn.Module):
    """G-MLP: the anchor's full-rank linear path plus a rank-k GELU correction, per token."""

    def __init__(self, d_in, k, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.down = nn.Linear(d_in, k)
        self.up = nn.Linear(k, d_out)

    def forward(self, x):
        return self.lin(x) + self.up(F.gelu(self.down(x)))


class Nano10(nn.Module):
    """Backbone -> concat chosen hidden states -> head PER TOKEN -> masked mean -> (normalize)."""

    def __init__(self, student_key, n_layers=3, head="linear", mlp_k=192, out_dim=OUT_DIM,
                 max_seq=512):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        repo = REPOS[student_key]
        self.key, self.n_layers, self.head_kind = student_key, n_layers, head
        self.layers = LAYERS[student_key][n_layers]
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.backbone = AutoModel.from_pretrained(repo)
        d_in = self.backbone.config.hidden_size * n_layers
        self.d_in, self.out_dim, self.max_seq = d_in, out_dim, max_seq
        self.head = nn.Linear(d_in, out_dim) if head == "linear" \
            else MLPHead(d_in, mlp_k, out_dim)
        if max(self.layers) > self.backbone.config.num_hidden_layers:
            raise ValueError(f"{student_key}: layer {max(self.layers)} of "
                             f"{self.backbone.config.num_hidden_layers}")

    def features(self, input_ids, attention_mask):
        """-> (b, s, d_in): the concatenated hidden states, before the head."""
        hs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True).hidden_states
        return torch.cat([hs[l] for l in self.layers], dim=-1)

    def token_out(self, input_ids, attention_mask):
        """-> (b, s, out_dim): what the exported ONNX graph emits. Serving starts here."""
        return self.head(self.features(input_ids, attention_mask))

    def forward(self, input_ids, attention_mask, normalize=True):
        t = self.token_out(input_ids, attention_mask)
        m = attention_mask.unsqueeze(-1).to(t.dtype)
        v = (t * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return F.normalize(v, dim=-1, eps=1e-12) if normalize else v

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def under_cap(self):
        return self.n_params() <= CAP

    @torch.inference_mode()
    def encode_queries(self, texts, batch_size=256, prefix=""):
        """-> (n, out_dim) fp32 unit-norm, in input order. The serving path."""
        self.eval()
        dev = next(self.parameters()).device
        out = np.empty((len(texts), self.out_dim), dtype=np.float32)
        order = np.argsort([len(t) for t in texts], kind="stable")
        for i in range(0, len(order), batch_size):
            sel = order[i:i + batch_size]
            b = self.tok([prefix + texts[j] for j in sel], padding=True, truncation=True,
                         max_length=self.max_seq, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                v = self(b["input_ids"], b["attention_mask"])
            out[sel] = v.float().cpu().numpy()
        return out


@torch.inference_mode()
def pooled_features(model, texts, prefix="", batch_size=128):
    """-> (n, d_in) the MEAN-POOLED concatenated features, the warm start's design matrix.

    Pooling before a LINEAR head is the same map as pooling after it, so the ridge solve on these
    is exactly the per-token solve — at a fraction of the memory. `test_nano10` proves the
    equality rather than asserting it.
    """
    model.eval()
    dev = next(model.parameters()).device
    out = np.empty((len(texts), model.d_in), dtype=np.float32)
    order = np.argsort([len(t) for t in texts], kind="stable")
    for i in range(0, len(order), batch_size):
        sel = order[i:i + batch_size]
        b = model.tok([prefix + texts[j] for j in sel], padding=True, truncation=True,
                      max_length=model.max_seq, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            f = model.features(b["input_ids"], b["attention_mask"])
        m = b["attention_mask"].unsqueeze(-1).to(f.dtype)
        out[sel] = ((f * m).sum(1) / m.sum(1).clamp(min=1e-9)).float().cpu().numpy()
    return out


def ridge_head(X, Y, lam):
    """The M9 closed form, unchanged: solve on [X | 1], then row-normalize the prediction."""
    Xc = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])
    G = Xc.T @ Xc
    scale = float(np.trace(G) / G.shape[0])
    A = np.linalg.solve(G + lam * scale * np.eye(G.shape[0], dtype=np.float32), Xc.T @ Y)
    P = Xc @ A
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    return A, float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def warm_start_linear(model, X, Y, lam):
    """Install the ridge solution into a LINEAR per-token head. Identical treatment every arm."""
    if model.head_kind != "linear":
        raise ValueError("G-MLP has its own three-solve warm start (screen_registry warm_start)")
    A, resid = ridge_head(X, np.asarray(Y, dtype=np.float32), lam)
    dev = next(model.parameters()).device
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(np.ascontiguousarray(A[:-1].T)).to(dev))
        model.head.bias.copy_(torch.from_numpy(np.ascontiguousarray(A[-1])).to(dev))
    return {"n_fit": int(X.shape[0]), "lambda": lam, "train_objective": round(resid, 5),
            "d_in": int(X.shape[1])}


# ---- objectives (family D) ------------------------------------------------------------------

def loss_sq_l2(pred, target):
    """The anchor: squared L2 between unit-norm student and unit-norm teacher."""
    return ((pred - target) ** 2).sum(-1).mean()


def loss_norm_e2(pred, target):
    """D-NORM — LEAF's own loss is the NORM of the error, not its square (amendment A1).

    A one-line arm we had silently diverged from; the square is what M9 trained.
    """
    return (pred - target).norm(dim=-1).mean()


def loss_cov_weighted(pred, target, sigma):
    """D-COV — `L = (s - t)^T Σ (s - t)`, the registered form (§Recipe).

    Σ is the FULL covariance of the pool's frozen stella DOCUMENT vectors, unit-trace normalised
    and shrunk as `(1-α)Σ̂ + αI/1024` with α = 0.1 fixed. Not a per-dimension weight: a diagonal
    would keep the coordinate basis, and the whole claim is that error should be charged in the
    directions documents actually differ along, which are not axis-aligned. Plain L2 is
    reconstruction and spends a rank-limited output equally on every direction of the teacher's
    query space; this spends it where nDCG is decided.

    Frozen before the arm runs — it is data about the corpus, never a learned parameter.
    """
    d = pred - target
    return ((d @ sigma) * d).sum(-1).mean()


def cov_matrix(doc_vecs, alpha=0.10):
    """Σ for D-COV: unit-trace normalised, shrunk toward the identity. Computed ONCE from
    document vectors that already exist. α is fixed at 0.10 and is never tuned on a surface."""
    X = np.asarray(doc_vecs, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    S = (X.T @ X) / max(len(X) - 1, 1)
    S = S / np.trace(S)                              # unit trace, as registered
    d = S.shape[0]
    # `(1-α)Σ̂ + αI/d`: the shrunk matrix also has unit trace, so α is a pure mixing weight and
    # cannot rescale the loss. That invariance is what makes D-COV comparable to the anchor at
    # the same learning rate, and it is asserted in `test_nano10`.
    return (1 - alpha) * S + alpha * np.eye(d) / d


LOSSES = {"squared_l2": loss_sq_l2, "leaf_norm_e2": loss_norm_e2,
          "document_covariance_weighted": loss_cov_weighted}


# ---- the kill and plateau rules (§Kill) -------------------------------------------------------

KILL_DROP, PLATEAU_MIN_GAIN, PLATEAU_FROM_CYCLE = 0.0056, 0.003, 3


def kill_fires(evals, kind_of, drop=KILL_DROP):
    """§Kill: two CONSECUTIVE scheduled evaluations more than `drop` below the best evaluation OF
    THEIR OWN KIND — midpoints against midpoints, cycle ends against cycle ends — so the rule can
    fire INSIDE a build and not only at its end (Opus M5).

    **A registered rule with two readings, and the plain one is implemented (T2-9).**
    "Two consecutive scheduled evaluations" can mean consecutive in the SCHEDULE (a midpoint and
    the cycle end after it, each below its own kind's best) or consecutive WITHIN a kind (two
    successive midpoints). The schedule reading is the plain sense of the words and is the safer
    of the two: it demands both kinds be failing at once, where the per-kind reading kills an arm
    on two bad midpoints alone. A false kill costs a whole arm, and the rule's stated purpose —
    "so the rule can fire inside the build and not only at its end" — is served either way,
    because it is satisfied by midpoints entering the comparison at all.

    `evals` is the ordered list of scheduled evaluation values; `kind_of(i)` returns that
    evaluation's kind. A non-finite value fires immediately: that is the other half of the rule.
    -> (fired, reason)
    """
    best = {}
    bad = 0
    for i, m in enumerate(evals):
        if m is None or not np.isfinite(m):
            return True, f"non-finite evaluation at index {i}"
        k = kind_of(i)
        b = best.get(k)
        if b is not None and m < b - drop:
            bad += 1
            if bad >= 2:
                return True, (f"two consecutive evaluations more than {drop} below the best of "
                              f"their own kind (index {i}, kind {k!r}, {m:.4f} vs best {b:.4f})")
        else:
            bad = 0
        best[k] = m if b is None else max(b, m)
    return False, None


def plateau_fires(cycle_end_evals, min_gain=PLATEAU_MIN_GAIN, from_cycle=PLATEAU_FROM_CYCLE):
    """§Kill: read BEST-TO-BEST on annealed checkpoints only. Fires at the first cycle end
    k >= `from_cycle` (1-based) where `m_k - max(m_1..m_{k-1}) >= min_gain` FAILS. Independent of
    the cycle cap (Opus M6, Codex 2026-09-04). -> (fired, cycle_index_1based or None)
    """
    for k in range(from_cycle, len(cycle_end_evals) + 1):
        gain = cycle_end_evals[k - 1] - max(cycle_end_evals[:k - 1])
        if gain < min_gain:
            return True, k
    return False, None


# ---- the 4-step mix window (family B) --------------------------------------------------------

WINDOWS = {"100/0": ("Q", "Q", "Q", "Q"), "75/25": ("Q", "Q", "Q", "D"),
           "50/50": ("Q", "Q", "D", "D")}


def mix_window(pattern, step):
    """-> 'Q' or 'D' for a given step. A fixed 4-step window, not a per-step coin flip, so an
    arm's query/document ratio is exact at every checkpoint rather than only in expectation."""
    w = WINDOWS[pattern]
    return w[step % len(w)]


def window_shares(pattern, steps):
    q = sum(1 for s in range(steps) if mix_window(pattern, s) == "Q")
    return {"Q": q, "D": steps - q, "q_share": q / max(steps, 1)}


# ---- the cyclic schedule (§Recipe) -----------------------------------------------------------

def lr_at(step, total_steps, cycles=3, peak=1e-4, final=1e-5):
    """LEAF's small-batch cyclic schedule: `cycles` cycles of equal length, each a linear decay
    from `peak` to `final`, restarting at `peak`. The last step of every cycle is a cycle END,
    which is where COV is read and where the sign-stability clause looks."""
    per = max(total_steps // cycles, 1)
    within = step % per
    return peak + (final - peak) * (within / max(per - 1, 1))


def cycle_ends(total_steps, cycles=3):
    per = max(total_steps // cycles, 1)
    return [min((c + 1) * per, total_steps) - 1 for c in range(cycles)]
