"""E14-HEAD: renormalized doc-side heads, trained jointly with the query table.

Dylan's E14 ruling was "measure it small first". This is the small measurement. The document
tower was never trained to be reachable by a bag of token vectors; E14 asks what that costs. The
expensive answer is `E14-LORA` (fine-tune the tower), which is registered and refused pending a
fresh ruling. The cheap answer is here: re-shape the document space with a small head over the teacher's
**cached** document vectors and see whether the space is re-shapeable at all. The transformer is
never re-run, so this costs a training run rather than millions of forward passes.

BOTH HEADS ARE RENORMALIZED, AND THAT IS WHAT MAKES EVEN THE LINEAR ONE REAL CAPACITY. This
file's first version asserted, from the registry row, that a linear doc-side map is absorbable
into the table and therefore a no-op. That is FALSE, and `m8/LEDGER.md` section 6's D1 entry
already said so: retrieval L2-normalizes documents, so the score is `q.(Md)/|Md|` and the
per-document factor `1/|Md|` cannot move into a shared query row. `results/m7_absorb_check.json`
measures rank agreement with the absorbed form at 1.000 WITHOUT renormalization and 0.000 WITH
it. So `LIN` is the primary probe -- cheaper (1.05M vs 4.2M parameters) and better conditioned --
and `MLP` is its control, the arm that says whether nonlinearity bought anything the linear map
did not.

ZERO-INIT MAKES THE HEAD `normalize(d)`, NOT `d`, AND THE DIFFERENCE IS NOT NOTHING. The cached
document vectors are only approximately unit-norm: over 100,000 sampled pool rows only 0.36% have
float32 norm exactly 1, with max |norm-1| = 4.8e-05. R0 scores those raw vectors, so a
head-bearing arm is NOT R0-plus-capacity at step 0, and renormalization shifts Phase-A logits and
therefore the whole training trajectory. The comparator is `R0N` -- this same path with the head
frozen at identity -- and not the existing R0 arms. `R0N` against R0 is reported separately as an
end-to-end null on the patch stack.

THE SCOPE LIMIT IS THE POINT OF THE STAGING, AND IT IS ASYMMETRIC. An MLP on the final document
vector cannot recover information the tower already discarded. So this tests *is the document
space re-shapeable*, NOT *can the tower learn to be bag-reachable* -- which is E14's actual
question. A null here is therefore WEAK evidence about the LoRA and may never be written as
closing E14; a positive here is STRONG evidence for buying it.

WHY THE LOSS IS COPIED RATHER THAN WRAPPED. `m7src/train.py::infonce` computes the false-negative
mask from the SAME `neg_v` tensor it scores with:

    t_neg = teacher_q @ neg_v.T
    mask  = t_neg > (t_pos - fn_margin)
    s_neg = s_neg.masked_fill(mask, -inf)

Wrapping `infonce` and handing it head-transformed vectors would put that mask under the control
of the trainable head -- and masking more negatives makes InfoNCE trivially smaller. That is a
direct reward-hacking channel into the objective: the head could cut its own loss by inflating the
mask rather than by improving any geometry, and `fn_margin` is 0.02 and live in R0. The mask is a
statement about what THE TEACHER considers relevant, a property of the data and not of the head,
so it is computed in the raw teacher space. `fn_space` exposes the choice rather than burying it,
and `self_test()` proves the copy is faithful before any arm runs.

m7src is frozen (G3), so the head reaches the training loop by rebinding `train.infonce` and
`torch.optim.Adam` in a per-arm subprocess. There is no other route short of vendoring 800 lines
of `train.py`, which would rot silently against the original.
"""
import argparse
import json
import sys

import m8base

REPO = m8base.REPO
WORK = REPO / "work"

# The architectures, fixed here so `b6_pre.py` can export the SAME shapes. A PASS on a different
# shape does not cover what is trained.
HIDDEN_MULT = 2                 # MLP hidden width = HIDDEN_MULT * dim
HEADS = ("lin", "mlp")          # `lin` is the PRIMARY; `mlp` is its nonlinearity control
NORM_EPS = 1e-12


def build_head(dim, kind="lin", device="cuda", dtype=None):
    """`normalize(d + f(d))` with `f` ZERO-initialised.

      * kind="lin": f(d) = W d, W (dim, dim) zero-init. ~1.05M params at dim 1024. PRIMARY.
      * kind="mlp": f(d) = W2 GELU(W1 d + b1), hidden HIDDEN_MULT*dim, W2 zero-init. ~4.2M params.

    Zero-init on the output projection means the head starts as `normalize(d)`, so the arm begins
    at its comparator rather than at a randomly scrambled document space -- otherwise part of the
    2500 Phase-A steps would be spent recovering from the initialization and a null would be
    confounded with "not enough budget to recover".

    NOTE WHAT ZERO-INIT DOES *NOT* GIVE, because the first version of this file claimed it did:
    the head starts at `normalize(d)`, NOT at `d`. The cached document vectors are only
    approximately unit-norm (0.36% exactly 1 over 100,000 sampled pool rows, max |norm-1|
    4.8e-05), and R0 scores them raw. So this is not R0-plus-capacity at step 0, and the
    comparator is `R0N` -- this same path with the head frozen at identity.

    For the MLP, zero `W2` also zeroes the gradient to `W1` at step 0 but not to `W2` itself, so
    `W2` leaves zero at step 1 and `W1` starts moving at step 2. This is LoRA's initialization.
    """
    import torch
    import torch.nn as nn

    if kind not in HEADS:
        raise ValueError(f"unknown head {kind!r}; expected one of {HEADS}")

    class DocHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            if kind == "lin":
                self.fc = nn.Linear(dim, dim, bias=False)
                nn.init.zeros_(self.fc.weight)
            else:
                h = HIDDEN_MULT * dim
                self.fc1 = nn.Linear(dim, h)
                self.fc2 = nn.Linear(h, dim, bias=False)
                nn.init.zeros_(self.fc2.weight)

        def forward(self, v):
            if self.kind == "lin":
                y = v + self.fc(v)
            else:
                y = v + self.fc2(torch.nn.functional.gelu(self.fc1(v)))
            # `infonce` takes plain dot products, so an un-normalized head would rescale every
            # logit -- i.e. change the effective temperature, a different treatment wearing this
            # one's name. The renormalization is also exactly what makes the LINEAR head genuine
            # capacity rather than an absorbable no-op (see the module docstring).
            return y / y.norm(dim=-1, keepdim=True).clamp_min(NORM_EPS)

    m = DocHead().to(device)
    return m.to(dtype) if dtype is not None else m


def infonce_head(qv, pos_v, neg_v, temp, teacher_q=None, teacher_pos=None, fn_margin=0.0,
                 neg_pool_idx=None, pos_pool_idx=None, all_pos_idx=None, stats=None,
                 head=None, fn_space="raw"):
    """`m7src/train.py::infonce`, verbatim, with the SCORED document vectors passed through `head`.

    `fn_space` controls the space the false-negative mask is computed in:
      * "raw"  -- the mask uses the untransformed teacher vectors. The registered choice: the mask
                  is a property of the data, and this closes the reward-hacking channel above.
      * "head" -- the mask follows the head. Available so the choice is measurable rather than
                  merely asserted, NOT because it is a defensible default.

    With `head=None` this must be bit-identical to the original; `self_test()` is what proves it.
    """
    import torch
    import torch.nn.functional as F

    def H(x):
        return x if head is None else head(x)

    s_pos = (qv * H(pos_v)).sum(1, keepdim=True) / temp
    s_neg = (qv @ H(neg_v).T) / temp
    if neg_pool_idx is not None and pos_pool_idx is not None:
        same = neg_pool_idx.unsqueeze(0) == pos_pool_idx.unsqueeze(1)
        if all_pos_idx is not None:
            same = same | (all_pos_idx.unsqueeze(2) == neg_pool_idx.view(1, 1, -1)).any(1)
        s_neg = s_neg.masked_fill(same, float("-inf"))
    if fn_margin > 0 and teacher_q is not None:
        with torch.no_grad():
            fn_neg, fn_pos = (neg_v, teacher_pos) if fn_space == "raw" else (H(neg_v),
                                                                            H(teacher_pos))
            t_neg = teacher_q @ fn_neg.T
            t_pos = (teacher_q * fn_pos).sum(1, keepdim=True)
            mask = t_neg > (t_pos - fn_margin)
        if stats is not None:
            stats["fn_masked_frac"] = round(float(mask.float().mean()), 4)
        s_neg = s_neg.masked_fill(mask, float("-inf"))
    logits = torch.cat([s_pos, s_neg], 1)
    return F.cross_entropy(logits, torch.zeros(len(qv), dtype=torch.long, device=qv.device))


def self_test(device="cpu"):
    """Prove the copy is faithful, and prove the head is the identity at init.

    Runs BEFORE any arm. A copied loss that has drifted from its original would make every number
    this probe produces a comparison between two different objectives, and the drift would look
    exactly like a result.
    """
    import numpy as np
    import torch

    sys.path.insert(0, str(REPO / "m7src"))
    sys.path.insert(0, str(REPO / "bench"))
    import train as m7train

    torch.manual_seed(0)
    B, N, D = 8, 64, 32
    checks, failures = [], []

    # Every combination that changes a BRANCH in the function, not just a few plausible ones: a
    # copy is only proved faithful on the paths it was exercised on.
    for fn_margin in (0.0, 0.02):
        for use_idx in (False, True):
            for use_allpos in (False, True):
                if use_allpos and not use_idx:
                    continue                    # all_pos_idx is only read inside the idx branch
                g = torch.Generator().manual_seed(hash((fn_margin, use_idx, use_allpos)) % 2**31)

                def rn(*s):
                    x = torch.randn(*s, generator=g, device=device)
                    return x / x.norm(dim=-1, keepdim=True)

                qv, pos_v, neg_v = rn(B, D), rn(B, D), rn(N, D)
                tq, tp = rn(B, D), rn(B, D)
                pos_idx = torch.randint(0, 1000, (B,), generator=g)
                neg_idx = torch.randint(0, 1000, (N,), generator=g)
                allpos = torch.randint(0, 1000, (B, 3), generator=g)
                kw = dict(teacher_q=tq, teacher_pos=tp, fn_margin=fn_margin)
                if use_idx:
                    kw.update(neg_pool_idx=neg_idx, pos_pool_idx=pos_idx)
                if use_allpos:
                    kw["all_pos_idx"] = allpos
                s_ref, s_new = {}, {}
                ref = m7train.infonce(qv, pos_v, neg_v, 0.02, stats=s_ref, **kw)
                new = infonce_head(qv, pos_v, neg_v, 0.02, stats=s_new, head=None, **kw)
                name = f"fn_margin={fn_margin} idx={use_idx} allpos={use_allpos}"
                same = bool(torch.equal(ref, new)) and s_ref == s_new
                checks.append({"case": name, "ref": float(ref), "copy": float(new),
                               "stats_match": s_ref == s_new, "bit_identical": same})
                if not same:
                    failures.append(name)

    # WHAT ZERO-INIT ACTUALLY GIVES, tested on REAL cached document vectors rather than on the
    # pre-normalized random ones the first version of this test used. That version passed by
    # assuming its own conclusion: it fed unit-norm inputs, for which normalize(d) == d trivially,
    # and so could never have detected that the head is `normalize(d)` and not `d`
    # (m8/CODEMAP.md pitfall 17's class -- a check that cannot fail is not a check).
    import pool as poolmod
    pv = poolmod.build()
    pv = pv[1] if isinstance(pv, tuple) else pv
    pv = getattr(pv, "vecs", pv)
    rows = np.asarray(pv[np.sort(np.random.default_rng(0).choice(pv.shape[0], 4096,
                                                                 replace=False))],
                      dtype=np.float32)
    real = torch.from_numpy(rows).to(device)
    norms = real.norm(dim=-1)
    heads = {k: build_head(real.shape[1], kind=k, device=device) for k in HEADS}
    init = {}
    for k, h in heads.items():
        with torch.no_grad():
            out_v = h(real)
        init[k] = {
            "max_abs_dev_from_raw_d": float((out_v - real).abs().max()),
            "max_abs_dev_from_normalize_d": float(
                (out_v - real / real.norm(dim=-1, keepdim=True)).abs().max()),
        }
    cached_norm = {"frac_exactly_unit_fp32": float((norms == 1.0).float().mean()),
                   "max_abs_norm_minus_1": float((norms - 1).abs().max()),
                   "mean_abs_norm_minus_1": float((norms - 1).abs().mean())}

    # The claim that survives: at init the head IS normalize(d), to machine precision.
    for k, v in init.items():
        if v["max_abs_dev_from_normalize_d"] > 1e-6:
            failures.append(f"{k}: not normalize(d) at init "
                            f"(max abs dev {v['max_abs_dev_from_normalize_d']:.3g})")
    # The claim that does NOT survive, asserted here so it can never be quietly re-adopted: the
    # head is measurably NOT the raw cached vector, which is why the comparator is R0N.
    identity_on_raw = all(v["max_abs_dev_from_raw_d"] <= 1e-6 for v in init.values())
    if identity_on_raw:
        failures.append("the head reproduced the RAW cached vectors, which would mean the pool is "
                        "exactly unit-norm -- the R0N comparator's whole justification. Re-measure "
                        "before trusting either.")

    # Nonlinearity, per head: perturb the output projection off zero and check additivity fails
    # for the MLP and HOLDS for the linear one. Both are genuine capacity because of the
    # renormalization; only the MLP is nonlinear IN f, and the arms exist to separate the two.
    a, b = real[:1], real[1:2]
    additivity = {}
    for k, h in heads.items():
        with torch.no_grad():
            w = h.fc.weight if k == "lin" else h.fc2.weight
            w.normal_(0, 0.05)
            f = (lambda v: h.fc(v)) if k == "lin" else (
                lambda v: h.fc2(torch.nn.functional.gelu(h.fc1(v))))
            additivity[k] = float((f(a + b) - (f(a) + f(b))).abs().max())
    if additivity["lin"] > 1e-4:
        failures.append("the linear head's f is not additive")
    if additivity["mlp"] <= 1e-3:
        failures.append("the mlp head's f is additive -- it is not a nonlinearity control")

    out = {"_what": "E14-HEAD self-test: the copied loss against m7src's, what zero-init "
                    "actually gives on REAL cached vectors, and each head's linearity",
           "loss_equivalence": checks,
           "cached_document_vector_norms": cached_norm,
           "head_at_init": init,
           "head_at_init_note": ("`max_abs_dev_from_normalize_d` near zero is the claim that "
                                 "holds. `max_abs_dev_from_raw_d` is NOT zero, and that is why "
                                 "the comparator is R0N and not the existing R0 arms."),
           "f_additivity_max_abs": additivity,
           "architecture": {"heads": list(HEADS), "hidden_mult": HIDDEN_MULT,
                            "norm_eps": NORM_EPS,
                            "form": "normalize(d + f(d)), output projection zero-init"},
           "pass": not failures, "failures": failures}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["self-test"])
    a = ap.parse_args()
    if a.step == "self-test":
        out = self_test()
        print(json.dumps(out, indent=2))
        return 0 if out["pass"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
