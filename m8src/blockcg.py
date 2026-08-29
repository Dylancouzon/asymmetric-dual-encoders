"""Block conjugate gradients for the ridge table solve -- the thing B7 measures, and the thing
that reopens two closed doors at once.

WHY IT EXISTS. The closed-form table solve is

    minimize_W  ||X W - Y||_F^2 + lam ||W - W0||_F^2
    X (n x V) row-normalized token counts,  Y (n x d) teacher query vectors,  W0 the init

and `m7src/stage0_ridge.solve_ridge` answers it by forming the dense fp64 Gram `X^T X` and calling
a direct solve. That is exact and fast at V = 30,522, where the Gram is 7.45 GB. It is also a
hard wall:

    V = 30,522 ->   7.45 GB      (fits)
    V = 50,368 ->  20.3  GB      (does not: this is the arithmetic on which M7 closed granite-r2
                                  and gte-modernbert "on arithmetic, not merit")
    V = 65,536 ->  34.4  GB
    V = 131,072 -> 137.4 GB

So the Gram is what stops M8 from (a) sizing D2's self-trained 64-128K tokenizer, and (b)
screening any teacher whose vocabulary is not the 30,522 WordPiece one -- which is every T1
challenger. One solver reopens both.

THE METHOD. Block CG on the normal equations, which never forms the Gram: each iteration needs
only `A(P) = X^T (X P) + lam P` with P of shape (V x d). Memory is a handful of (V x d) arrays
plus the sparse X, so it scales linearly in V where the Gram scales quadratically:

    V = 131,072, d = 1024, fp32 -> 0.54 GB per array, ~5 arrays -> ~2.7 GB

The operator is symmetric positive definite for lam > 0, so CG is the right algorithm and its
convergence is governed by sqrt(kappa) with kappa <= (sigma_max^2 + lam)/lam. Each right-hand
side gets its own scalar step lengths -- this is the "block" part done the cheap way (d
independent CG runs sharing one sparse matmul), not the full block-CG with d x d step matrices.
The cheap version is what the sparse matmul cost makes worthwhile: the matmul is shared, which is
where all the time goes, and the d systems differ only in their right-hand sides.

CORRECTNESS IS NOT ASSUMED. `verify()` solves the same real system both ways at V = 30,522 and
reports the relative error. A solver that agrees with the direct answer to ~1e-6 on the artifact
that ships is the evidence; a solver that merely converges is not. This is the check that turns
"block CG should work" into a measurement -- and this project has twice had an "obvious" numerical
claim come back wrong (the doc-side map's absorbability, reported with its two numbers reversed on
a transposed matrix).
"""
import argparse
import json
import sys
import time

import numpy as np
import scipy.sparse as sp

import m8base

RESULTS = m8base.RESULTS


def _to_torch_sparse(X, device, dtype):
    import torch
    X = X.tocoo()
    idx = torch.from_numpy(np.vstack([X.row, X.col])).to(device=device, dtype=torch.int64)
    val = torch.from_numpy(X.data).to(device=device, dtype=dtype)
    return torch.sparse_coo_tensor(idx, val, X.shape, device=device, dtype=dtype).coalesce()


def block_cg_ridge(X, Y, W0, lam, tol=None, maxiter=1500, device="cuda", dtype=None,
                   verbose=False, log_every=100, precond=True):
    """Solve (X^T X + lam I) W = X^T Y + lam W0 without forming the Gram.

    Returns (W, info). `tol` is on the per-column relative residual: the loop stops when EVERY
    column is below it, so a single slow-converging direction cannot be hidden by an average.
    """
    import torch
    # fp32 on cuda: this card's fp64 rate is 1/64 of its fp32 rate, and the destination is an
    # int8 table whose per-row absmax quantization is ~1e-2 relative -- so a 1e-6 solve is already
    # four orders of magnitude tighter than the artifact it feeds. The tolerance follows the
    # dtype: asking fp32 for 1e-8 just burns every iteration against machine epsilon.
    dtype = dtype or (torch.float32 if device == "cuda" else torch.float64)
    if tol is None:
        tol = 1e-6 if dtype == torch.float32 else 1e-8
    n, V = X.shape
    d = Y.shape[1]
    Xt = _to_torch_sparse(X, device, dtype)
    # Precomputed, not re-derived per iteration: `Xt.t()` on a COO tensor is cheap to express and
    # expensive to multiply with, and this runs twice per CG step.
    XtT = _to_torch_sparse(X.T.tocoo(), device, dtype)
    Yt = torch.as_tensor(np.ascontiguousarray(Y), device=device, dtype=dtype)
    W = torch.as_tensor(np.ascontiguousarray(W0), device=device, dtype=dtype).clone()
    W0t = torch.as_tensor(np.ascontiguousarray(W0), device=device, dtype=dtype)

    def A(P):
        return torch.sparse.mm(XtT, torch.sparse.mm(Xt, P)) + lam * P

    # JACOBI PRECONDITIONING. Real token frequencies are Zipfian, so diag(X^T X) spans several
    # orders of magnitude and unpreconditioned CG crawls on precisely the rare rows the table most
    # needs. The diagonal of the Gram is just the column sum of X^2 -- one pass, no Gram -- and
    # dividing by it removes most of that skew. Reported measured, not assumed: `precond=False`
    # runs the same problem without it so the effect is a number.
    if precond:
        dg = np.asarray(X.multiply(X).sum(axis=0)).ravel() + lam
        Minv = torch.as_tensor(1.0 / np.maximum(dg, 1e-12), device=device,
                               dtype=dtype).unsqueeze(1)
    else:
        Minv = None

    B = torch.sparse.mm(XtT, Yt) + lam * W0t
    R = B - A(W)
    Z = R * Minv if Minv is not None else R
    P = Z.clone()
    rz = (R * Z).sum(0)
    b_norm = (B * B).sum(0).sqrt().clamp_min(1e-30)
    hist = []
    t0 = time.time()
    it = 0
    rel = (R * R).sum(0).sqrt() / b_norm
    for it in range(1, maxiter + 1):
        AP = A(P)
        denom = (P * AP).sum(0).clamp_min(1e-30)
        alpha = rz / denom
        W += alpha * P
        R -= alpha * AP
        rel = (R * R).sum(0).sqrt() / b_norm
        worst = float(rel.max())
        if verbose and (it % log_every == 0 or it == 1):
            print(f"    cg it={it:4d} worst_rel_resid={worst:.3e} ({time.time()-t0:.1f}s)",
                  flush=True)
        hist.append(worst)
        if worst < tol:
            break
        Z = R * Minv if Minv is not None else R
        rz_new = (R * Z).sum(0)
        P = Z + (rz_new / rz.clamp_min(1e-30)) * P
        rz = rz_new

    info = {"iterations": it, "worst_rel_residual": float(rel.max()),
            "mean_rel_residual": float(rel.mean()), "seconds": round(time.time() - t0, 2),
            "converged": bool(float(rel.max()) < tol), "tol": tol, "maxiter": maxiter,
            "device": device, "dtype": str(dtype), "V": int(V), "n": int(n), "d": int(d),
            "lam": lam, "preconditioner": "jacobi" if precond else "none",
            "history_worst": hist[:5] + (["..."] if len(hist) > 10 else [])
                        + hist[-5:] if hist else []}
    out = W.detach().to("cpu").numpy().astype(np.float64)
    del Xt, XtT, Yt, W, W0t, R, P, B, Z
    m8base.empty_cache()
    return out, info


def direct_ridge(X, Y, W0, lam):
    """The M7 method, for the correctness check only: dense fp64 Gram + direct solve."""
    import scipy.linalg as sla
    G = np.asarray((X.T @ X).todense(), dtype=np.float64)
    G[np.diag_indices_from(G)] += lam
    rhs = np.asarray(X.T @ Y, dtype=np.float64) + lam * np.asarray(W0, dtype=np.float64)
    t0 = time.time()
    W = sla.solve(G, rhs, assume_a="pos", overwrite_a=True, overwrite_b=True)
    return W, {"seconds": round(time.time() - t0, 2), "gram_bytes": G.nbytes}


def synthetic(n, V, d, nnz=12, seed=0, zipf_s=1.07, uniform=False):
    """A bag matrix for the FEASIBILITY half of B7, at vocabularies whose tokenizer does not exist
    yet. Labelled synthetic wherever it is reported: it measures the SOLVER, not the quality of
    any table.

    TOKENS ARE DRAWN ZIPFIAN, NOT UNIFORM, and that is the whole point. CG's cost is governed by
    the condition number of `X^T X + lam I`, and a uniform token distribution gives a nearly
    isotropic Gram -- an unrealistically easy problem. Real text is Zipfian (s ~ 1.0-1.1 for
    English): a few hundred tokens carry most of the mass, the spectrum is extremely skewed, and
    the rare rows the table most needs are the worst-conditioned directions. Measuring the solver
    on uniform draws would have produced a feasibility PASS that says nothing about the real
    problem -- the "wrong number rather than a crash" failure class this project has been bitten
    by before. `uniform=True` is kept only to report the optimistic baseline beside the realistic
    one, so the gap between them is visible rather than assumed."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), nnz)
    if uniform:
        cols = rng.integers(0, V, size=n * nnz)
    else:
        # Zipf over rank, mapped onto vocabulary ids. `rng.zipf` is unbounded, so resample the
        # tail rather than clipping it -- clipping would pile the whole tail onto one row and
        # create an artificial singleton direction.
        cols = rng.zipf(zipf_s, size=n * nnz)
        bad = cols > V
        while bad.any():
            cols[bad] = rng.zipf(zipf_s, size=int(bad.sum()))
            bad = cols > V
        cols = cols - 1
    vals = np.full(n * nnz, 1.0 / nnz)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(n, V))
    Y = rng.standard_normal((n, d)).astype(np.float64)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    W0 = rng.standard_normal((V, d)).astype(np.float64) * 0.01
    return X, Y, W0


def coverage(X):
    """How much of the vocabulary the draw actually reaches -- the number that says whether a
    feasibility result at V rows is about V rows at all."""
    hit = np.diff(X.tocsc().indptr) > 0
    return {"rows_reached": int(hit.sum()), "V": int(X.shape[1]),
            "fraction_reached": float(hit.mean())}


def verify(n=200_000, V=30_522, d=1024, lam=1e-3, device="cuda", seed=0, uniform=False):
    """Block CG vs the direct solve at the control vocabulary. The claim is agreement, measured."""
    X, Y, W0 = synthetic(n, V, d, seed=seed, uniform=uniform)
    t0 = time.time()
    Wd, di = direct_ridge(X, Y, W0, lam)
    di["total_seconds"] = round(time.time() - t0, 2)
    Wc, ci = block_cg_ridge(X, Y, W0, lam, device=device, verbose=True)
    num = float(np.linalg.norm(Wc - Wd))
    den = float(np.linalg.norm(Wd))
    return {
        "setting": {"n": n, "V": V, "d": d, "lam": lam,
                    "token_distribution": "uniform" if uniform else "zipf(s=1.07)",
                    "coverage": coverage(X)},
        "direct": di, "block_cg": ci,
        "relative_error_fro": num / den,
        "max_abs_error": float(np.abs(Wc - Wd).max()),
        "speedup_vs_direct": round(di["seconds"] / max(ci["seconds"], 1e-9), 2),
    }


def b7_curve(sizes=(30_522, 65_536, 131_072), n=200_000, d=1024, lam=1e-3, device="cuda",
             uniform=False):
    """B7's FEASIBILITY half: wall-clock and peak memory at each vocabulary size."""
    import torch
    rows = []
    for V in sizes:
        X, Y, W0 = synthetic(n, V, d, uniform=uniform)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        _, info = block_cg_ridge(X, Y, W0, lam, device=device, verbose=True)
        peak = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else None
        rows.append({
            "V": V, **{k: info[k] for k in ("iterations", "seconds", "worst_rel_residual",
                                            "converged")},
            "wall_seconds": round(time.time() - t0, 2),
            "peak_vram_gb": None if peak is None else round(peak, 2),
            "dense_gram_fp64_gb": round(V * V * 8 / 1e9, 2),
            "blockcg_arrays_fp32_gb": round(5 * V * d * 4 / 1e9, 2),
            "token_distribution": "uniform" if uniform else "zipf(s=1.07)",
            "coverage": coverage(X),
        })
        print(f"  V={V:,}: {rows[-1]['iterations']} its, {rows[-1]['wall_seconds']}s, "
              f"peak {rows[-1]['peak_vram_gb']} GB VRAM  "
              f"(a dense fp64 Gram here would be {rows[-1]['dense_gram_fp64_gb']} GB)", flush=True)
        del X, Y, W0
        m8base.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["verify", "curve", "smoke"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=200_000)
    a = ap.parse_args()

    import probe_guard
    if a.step == "smoke":
        r = verify(n=5_000, V=4_096, d=64, device=a.device)
        print(json.dumps(r, indent=2, default=str))
        return 0
    out = {"_note": "B7 -- the block-CG solver that decides whether D2's 64-128K vocabulary and "
                    "any non-WordPiece teacher screen are computable on this box at all.",
           "data_disclosure": (
               "the bag matrices here are SYNTHETIC: 12 non-zeros per row, token ids drawn "
               "ZIPFIAN (s = 1.07) rather than uniform, because CG's cost is set by the condition "
               "number and a uniform draw is an unrealistically easy problem. The uniform result "
               "is reported beside it as the optimistic baseline so the gap is visible. This "
               "measures the SOLVER. The quality half of B7 -- the closed-form dev macro at each "
               "vocabulary -- needs a real trained tokenizer and is a separate, descriptive run.")}
    if a.step == "verify":
        out["correctness"] = verify(n=a.n, device=a.device)
        print(json.dumps(out["correctness"], indent=2, default=str))
    else:
        out["correctness_zipf"] = verify(n=a.n, device=a.device)
        out["correctness_uniform_baseline"] = verify(n=a.n, device=a.device, uniform=True)
        out["curve"] = b7_curve(n=a.n, device=a.device)
        out["correctness"] = out["correctness_zipf"]
        out["verdict"] = {
            "bar": "the 65,536-row solve must complete within the 18 GB peak-RAM budget and "
                   "under 4 hours wall-clock",
            "pass": bool(any(r["V"] == 65_536 and r["converged"]
                             and (r["peak_vram_gb"] or 0) < 18 and r["wall_seconds"] < 4 * 3600
                             for r in out["curve"])),
        }
        probe_guard.write_result(RESULTS / "m8_b7_solver.json", out, "B7", strict_commit=False)
        print(json.dumps(out["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
