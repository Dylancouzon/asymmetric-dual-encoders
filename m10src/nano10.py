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

# Which hidden states the head reads, top-down (§Recipe): the last layer, then two-thirds, then
# one-third of depth; family G's fourth arm adds one more. Hard-coded rather than derived, because
# these exact indices are what the parity artifacts were measured on.
#
# `1` is family G's **G-384** arm -- "feature = last layer only (384, M9's head)"
# (`instructions-m10.md`:616) -- and the key was MISSING, so `Nano10(..., n_layers=1)` raised
# `KeyError: 1` and a registered arm of the locked design could not be built at all. Found by the
# registered 90-step smoke of every arm shape, which is exactly what that requirement is for.
# Family G runs on family F's winner, so every student needs the key.
LAYERS = {
    "bge-small":  {1: (12,), 3: (12, 8, 4), 4: (12, 8, 4, 2)},
    "MiniLM-L6":  {1: (6,),  3: (6, 4, 2),  4: (6, 4, 2, 1)},
    "MiniLM-L12": {1: (12,), 3: (12, 8, 4), 4: (12, 8, 4, 2)},
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


# ---- export (the serving path) ----------------------------------------------------------------

def export_onnx(model, out_dir, max_len=512, opset=17):
    """Export the TOKEN OUTPUT — `head(x_t)` per token — and the tokenizer we intend to ship.

    fastembed does the masked mean and the normalize itself, so the graph must stop before the
    pool. Two things this function exists to get right, both of which have already cost time:

    - **`torch.compile` never reaches here.** `m10/HEADROOM.md` §T registers that export, parity,
      encoding and evaluation run EAGER and that only the training step compiles; a compiled
      wrapper's `state_dict` is also the wrong shape. `_orig_mod` is unwrapped if present.
    - **The tokenizer's `max_length` is written explicitly.** fastembed serves
      `min(model_max_length, max_length)` from `tokenizer_config.json`, and `all-MiniLM-*-v2`
      ships 128 there — which made a correct export read 0.93 min-cos against a 512-token
      reference (`results/m10_student_parity_box.json`).
    """
    m = getattr(model, "_orig_mod", model)
    m = m.eval().float()
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ex = m.tok(["a short query", "a somewhat longer example sentence for the export trace"],
               padding=True, truncation=True, max_length=64, return_tensors="pt")

    class TokenOut(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner.token_out(input_ids, attention_mask)

    p = d / "model.onnx"
    torch.onnx.export(TokenOut(m), (ex["input_ids"], ex["attention_mask"]), str(p),
                      input_names=["input_ids", "attention_mask"],
                      output_names=["token_embeddings"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"},
                                    "attention_mask": {0: "b", 1: "s"},
                                    "token_embeddings": {0: "b", 1: "s"}},
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    m.tok.backend_tokenizer.enable_truncation(max_length=max_len)
    m.tok.model_max_length = max_len
    m.tok.save_pretrained(d)
    m.backbone.config.save_pretrained(d)
    tc = d / "tokenizer_config.json"
    cfg = json.loads(tc.read_text())
    cfg["max_length"] = cfg["model_max_length"] = max_len
    tc.write_text(json.dumps(cfg, indent=1))

    import onnx
    g = onnx.load(str(p))
    custom = sorted({n.domain for n in g.graph.node} - {"", "ai.onnx", "ai.onnx.ml"})
    return {"path": str(p), "bytes": p.stat().st_size, "opset": opset,
            "custom_domain_ops": custom, "n_nodes": len(g.graph.node),
            "params_total": int(m.n_params()), "under_35M_cap": bool(m.under_cap()),
            "served_max_length": max_len, "layers": list(m.layers),
            "head": m.head_kind}


def export_parity(model, out_dir, texts, max_len=512):
    """-> min-cos between the training form and ORT's token output followed by an external
    masked mean. `m10src/student_parity.py` is the end-to-end version through fastembed; this is
    the one a trained checkpoint runs, and §T requires it on a checkpoint from a COMPILED run."""
    import onnxruntime as ort
    m = getattr(model, "_orig_mod", model).eval().float()
    b = m.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    with torch.inference_mode():
        ref = m(b["input_ids"], b["attention_mask"]).cpu().numpy()
    sess = ort.InferenceSession(str(Path(out_dir) / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    tok = sess.run(None, {"input_ids": b["input_ids"].numpy(),
                          "attention_mask": b["attention_mask"].numpy()})[0]
    mm = b["attention_mask"].numpy()[..., None].astype(np.float32)
    got = (tok * mm).sum(1) / np.maximum(mm.sum(1), 1e-9)
    got = got / np.maximum(np.linalg.norm(got, axis=1, keepdims=True), 1e-12)
    cos = (ref * got).sum(1)
    return {"n_texts": len(texts), "min_cos": float(cos.min()),
            "max_abs": float(np.abs(ref - got).max())}


# ---- lambda selection, and G-MLP's three-solve warm start -----------------------------------
#
# `m9/registry.json` warm_start registers the lambda recipe and §Recipe says it is **reselected on
# the M10 sample**: fit on a random 50,000 of the 60,000-text warm-start sample (seed 21), score
# the remaining 10,000 under the ACTUAL normalized objective, locked grid 1e-6..1, ties to the
# LARGER lambda. Both halves are training text, so there is no dev surface in it. `m9src/warmfit`
# owns the procedure and is reused rather than reimplemented.

N_FIT_REGISTERED = 60_000       # m9/registry.json warm_start.n_fit


def select_lambda(X, Y, n_fit_split=50_000, seed=21):
    """-> (lambda, rows). The registered training-only holdout, on M10's own feature space.

    The registered split is 50,000 fit of a 60,000 sample -- five sixths. On a SMALLER sample the
    split is scaled to the same ratio rather than clamped to `len - 1`: clamping left exactly ONE
    validation row, whose objective is noise, so `warmfit.select`'s "ties go to the larger lambda"
    rule handed back the top of the grid every time. The arm smoke duly selected lambda = 1.0 for
    all eleven linear shapes, which is a near-zero head. Ratio-scaling keeps the holdout meaningful
    at any sample size and is bit-identical at the registered 60,000.

    **The smoke may still select 1.0, and that is now CORRECT rather than an artifact.** At the
    smoke's n_fit of 256 the split is 213 fit / 43 validation, and 213 rows against 1,153 ridge
    parameters is massively underdetermined, so heavy regularisation genuinely generalises best.
    What changed is the mechanism: the validation objectives now differ (1.9966 vs 2.0273 on a
    random probe) instead of tying on a single row. The real arms fit 60,000 rows and are
    overdetermined.
    """
    import warmfit
    X = np.asarray(X, dtype=np.float32)
    Xc = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])
    m = X.shape[0]
    n = n_fit_split if m >= N_FIT_REGISTERED else int(round(m * n_fit_split / N_FIT_REGISTERED))
    n = max(1, min(n, m - 1))
    return warmfit.select(Xc, np.asarray(Y, dtype=np.float32), n, seed=seed)


@torch.inference_mode()
def token_moments(model, texts, prefix="", batch_size=64, verbose=False):
    """-> (mu, Gram, n_tokens) over PER-TOKEN features, streamed.

    §Recipe: "the per-token PCA is a streamed 1152x1152 Gram matrix". Per-token features for
    60,000 texts are ~10^7 rows of 1152 floats, so they are never materialized: one pass
    accumulates the token count, the token sum and the token second-moment matrix in float64.
    """
    model.eval()
    dev = next(model.parameters()).device
    d = model.d_in
    s = np.zeros(d, dtype=np.float64)
    G = np.zeros((d, d), dtype=np.float64)
    n = 0
    for i in range(0, len(texts), batch_size):
        b = model.tok([prefix + t for t in texts[i:i + batch_size]], padding=True,
                      truncation=True, max_length=model.max_seq, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            f = model.features(b["input_ids"], b["attention_mask"])
        m = b["attention_mask"].bool().reshape(-1)
        X = f.reshape(-1, d).float()[m].double()            # real tokens only, padding excluded
        s += X.sum(0).cpu().numpy()
        G += (X.T @ X).cpu().numpy()
        n += int(X.shape[0])
        if verbose and (i // batch_size) % 100 == 0 and i:
            print(f"    token moments {i:,}/{len(texts):,} ({n:,} tokens)", flush=True)
    return s / max(n, 1), G / max(n, 1), n


def top_directions(mu, G2, k=192):
    """-> (W1, b1): the top-k principal directions of the CENTRED per-token covariance.

    §Recipe: "the top-192 principal directions of the frozen backbone's per-token 1152-d states on
    the fit set, centred (b1 = -W1 mu; sign of each direction fixed so its largest-magnitude
    component is positive)". The sign fix makes the init deterministic -- an eigenvector is only
    defined up to sign, and LAPACK's choice is not stable across machines or library versions.
    """
    C = G2 - np.outer(mu, mu)
    C = (C + C.T) / 2
    w, V = np.linalg.eigh(C)                      # ascending
    idx = np.argsort(w)[::-1][:k]
    W1 = V[:, idx].T                              # (k, d)
    flip = np.sign(W1[np.arange(k), np.argmax(np.abs(W1), axis=1)])
    flip[flip == 0] = 1.0
    W1 = W1 * flip[:, None]
    return W1.astype(np.float32), (-W1 @ mu).astype(np.float32)


@torch.inference_mode()
def gelu_features(model, texts, W1, b1, prefix="", batch_size=64):
    """-> (n, k) the pooled nonlinear feature `mean_t GELU(W1 x_t + b1)`, in input order.

    This is the design matrix for the third solve, and it is what makes the warm start EXACT for
    the training form: `up` is linear, so `mean_t up(GELU(...))` = `up(mean_t GELU(...))`.
    """
    model.eval()
    dev = next(model.parameters()).device
    Wt = torch.from_numpy(np.ascontiguousarray(W1.T)).to(dev)
    bt = torch.from_numpy(np.ascontiguousarray(b1)).to(dev)
    out = np.empty((len(texts), W1.shape[0]), dtype=np.float32)
    order = np.argsort([len(t) for t in texts], kind="stable")
    for i in range(0, len(order), batch_size):
        sel = order[i:i + batch_size]
        b = model.tok([prefix + texts[j] for j in sel], padding=True, truncation=True,
                      max_length=model.max_seq, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            f = model.features(b["input_ids"], b["attention_mask"])
        g = F.gelu(f.float() @ Wt + bt)
        m = b["attention_mask"].unsqueeze(-1).to(g.dtype)
        out[sel] = ((g * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy()
    return out


def warm_start_mlp(model, texts, Y, lam=None, prefix="", verbose=False):
    """G-MLP's registered THREE-SOLVE warm start (`instructions-m10.md`:616, screen_registry).

    1. `W_lin` = the anchor's ridge head, on pooled features -> the same map every other arm gets.
    2. `W1, b1` = the centred top-192 per-token principal directions, signs fixed.
    3. `W2, b2` = ridge from the pooled `mean_t GELU(W1 x_t + b1)` to the RESIDUAL of solve 1.

    All three share one fit sample. Exact for the training form, so G-MLP starts *at* the anchor's
    fitted head plus a fitted correction rather than at a random head -- which is the point: a
    fresh MLP head would put G3 (`G-MLP - G-1152`) at a handicap in the direction that rejects the
    non-default.
    """
    if model.head_kind != "mlp":
        raise ValueError("warm_start_mlp is for the G-MLP head; linear heads use warm_start_linear")
    Y = np.asarray(Y, dtype=np.float32)
    k = model.head.down.out_features
    dev = next(model.parameters()).device

    Xbar = pooled_features(model, texts, prefix=prefix)
    rows = None
    if lam is None:
        lam, rows = select_lambda(Xbar, Y)
    A, resid_lin = ridge_head(Xbar, Y, lam)

    mu, G2, n_tok = token_moments(model, texts, prefix=prefix, verbose=verbose)
    W1, b1 = top_directions(mu, G2, k=k)

    Gf = gelu_features(model, texts, W1, b1, prefix=prefix)
    Xc = np.hstack([Xbar, np.ones((Xbar.shape[0], 1), dtype=np.float32)])
    R = Y - Xc @ A                                  # the residual the correction has to explain
    A2, _ = ridge_head(Gf, R, lam)

    with torch.no_grad():
        model.head.lin.weight.copy_(torch.from_numpy(np.ascontiguousarray(A[:-1].T)).to(dev))
        model.head.lin.bias.copy_(torch.from_numpy(np.ascontiguousarray(A[-1])).to(dev))
        model.head.down.weight.copy_(torch.from_numpy(np.ascontiguousarray(W1)).to(dev))
        model.head.down.bias.copy_(torch.from_numpy(np.ascontiguousarray(b1)).to(dev))
        model.head.up.weight.copy_(torch.from_numpy(np.ascontiguousarray(A2[:-1].T)).to(dev))
        model.head.up.bias.copy_(torch.from_numpy(np.ascontiguousarray(A2[-1])).to(dev))

    P = Xc @ A + np.hstack([Gf, np.ones((Gf.shape[0], 1), dtype=np.float32)]) @ A2
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    obj = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
    return {"n_fit": int(Xbar.shape[0]), "lambda": lam, "lambda_rows": rows, "k": int(k),
            "d_in": int(model.d_in), "n_tokens": int(n_tok),
            "train_objective": round(obj, 5),
            "train_objective_linear_only": round(resid_lin, 5),
            "solves": "W_lin ridge on pooled | W1,b1 centred per-token PCA, signs fixed | "
                      "W2,b2 ridge from pooled GELU features to solve 1's residual"}


def warm_start_from_m9(model, ckpt_path, verbose=False):
    """C-M9init's registered head init: the M9 candidate's 384-d head is KEPT and the extra
    layers' columns are ZERO-initialised (`instructions-m10.md`:510, screen_registry arms.C-M9init).

    M9's student read one layer (384-d); M10's anchor reads three (1152-d). The registered init
    puts M9's fitted 384x1024 block in the columns for the layer it was fitted on -- the LAST
    layer, which `LAYERS[...][1]` names and which is `layers[0]` here -- and zeroes the rest, so
    the arm starts exactly at M9's function and the new capacity starts inert.

    `screen_registry` already discloses that C1 therefore confounds backbone init with head init.
    """
    if model.head_kind != "linear":
        raise ValueError("C-M9init is a linear-head arm")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    bb = {k.split("backbone.", 1)[1]: v for k, v in sd.items() if "backbone." in k}
    hw = next((v for k, v in sd.items() if k.endswith("head.weight")), None)
    hb = next((v for k, v in sd.items() if k.endswith("head.bias")), None)
    if hw is None:
        raise SystemExit(f"{ckpt_path}: no head.weight in the checkpoint")
    d1 = model.backbone.config.hidden_size
    if hw.shape != (model.out_dim, d1):
        raise SystemExit(f"{ckpt_path}: head is {tuple(hw.shape)}, expected "
                         f"{(model.out_dim, d1)} -- M9's head is one layer wide")
    expected = set(model.backbone.state_dict())
    got = set(bb)
    if not got:
        raise SystemExit(f"{ckpt_path}: no `backbone.*` keys -- this would silently install M9's "
                         f"head on a FRESH pretrained backbone and report it as an M9 warm start")
    missing_bb = expected - got
    if missing_bb:
        raise SystemExit(f"{ckpt_path}: {len(missing_bb)} backbone keys absent "
                         f"(e.g. {sorted(missing_bb)[:3]}) -- refusing a partial M9 init")
    missing = model.backbone.load_state_dict(bb, strict=False)
    dev = next(model.parameters()).device
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.weight[:, :d1].copy_(hw.to(dev))       # the LAST layer is features[:, :d1]
        model.head.bias.copy_(hb.to(dev)) if hb is not None else model.head.bias.zero_()
    return {"ckpt": str(ckpt_path), "backbone_keys_loaded": len(bb),
            "backbone_missing": len(getattr(missing, "missing_keys", [])),
            "head_block": [int(model.out_dim), int(d1)],
            "zeroed_columns": int(model.d_in - d1),
            "note": "M9's 384-d head kept in the last layer's columns; the two extra layers' "
                    "columns are zero, so the arm starts at M9's function"}
