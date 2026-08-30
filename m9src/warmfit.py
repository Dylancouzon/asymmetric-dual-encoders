"""STAGED — moves to m9src/warmfit.py once the in-flight anchor has written its artifact.

Honest lambda selection for the closed-form head warm start.

Codex pass 3, BLOCKER-B1: `m9/LEDGER.md` claimed lambda was selected on the training residual; the
head probe in fact evaluated every lambda on SCREEN-3 and took the argmax, which is a dev surface.
Selecting on the training residual would not have rescued it either — a ridge residual is monotone
in lambda, so it just picks the bottom of the grid — and the solve minimizes UNNORMALIZED output
error while the training objective normalizes the head output first.

So: split the warm-start pool into a fit half and a validation half, both training text, and score
each lambda with the ACTUAL objective (`||normalize(XA) - Y||^2`, summed over dim, meaned over
examples). No dev surface is touched. Ties go to the LARGER lambda.
"""
import json

import numpy as np

import m9base
from m9base import RESULTS, WORK

import guard9   # noqa: E402

GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
ARTIFACT = RESULTS / "m9_warmfit.json"


def solve(Xc, Y, lam):
    G = Xc.T @ Xc
    scale = float(np.trace(G) / G.shape[0])
    return np.linalg.solve(G + lam * scale * np.eye(G.shape[0], dtype=np.float32), Xc.T @ Y)


def objective(Xc, Y, A):
    """The training loss, not the solver's loss: normalize the head output first."""
    P = Xc @ A
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def select(Xc, Y, n_fit, seed=0):
    """-> (lambda, rows). Fit on a RANDOM `n_fit` of the sample, validate on the rest.

    Both halves are training text, so there is no dev leakage either way -- but taking the first
    n_fit of a sorted index list made the split index-contiguous and therefore length-skewed
    (fit p95 26 words against validation p95 16; Codex pass 4). A permutation removes that.
    """
    perm = np.random.default_rng(seed).permutation(Xc.shape[0])
    fit, val = perm[:n_fit], perm[n_fit:]
    Xf, Yf, Xv, Yv = Xc[fit], Y[fit], Xc[val], Y[val]
    rows = []
    for lam in GRID:
        A = solve(Xf, Yf, lam)
        rows.append({"lambda": lam,
                     "fit_unnormalized_sq_l2": float(np.mean(np.sum((Xf @ A - Yf) ** 2, 1))),
                     "fit_objective": objective(Xf, Yf, A),
                     "val_objective": objective(Xv, Yv, A)})
    # full precision, not the rounded display value (Codex pass 4); ties -> the larger lambda
    best = min(rows, key=lambda r: (r["val_objective"], -r["lambda"]))
    return best["lambda"], rows


def run(student_key="bge-small-en-v1.5"):
    import nano
    import data as m9data
    r = guard9.registry()["warm_start"]
    guard9.begin_run("m9-warmfit")

    texts = json.loads((WORK / "m9_screen_queries.json").read_text())
    rows_idx = np.load(WORK / "m9_screen_rows.npy")
    rng = np.random.default_rng(r["seed"])
    sel = np.sort(rng.choice(len(texts), size=r["n_fit"], replace=False))

    model = nano.Nano(student_key).cuda().eval()
    X = nano._pooled(model, [texts[i] for i in sel], "")
    Y = np.asarray(m9data.stella_query_targets()[rows_idx[sel]], dtype=np.float32)
    Xc = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])

    lam, grid = select(Xc, Y, r["n_fit_split"], seed=r["seed"])
    out = {"student": student_key, "n_fit": r["n_fit"], "n_fit_split": r["n_fit_split"],
           "seed": r["seed"], "grid": list(GRID), "rows": grid, "selected_lambda": lam,
           "registered_lambda": r["lambda"],
           "agrees_with_registry": lam == r["lambda"],
           "_what": "lambda selected on a TRAINING-ONLY holdout under the actual normalized "
                    "objective. No dev surface is read. Ties go to the larger lambda. The split is "
                    "a random permutation of the 60,000-text sample, not a contiguous prefix.",
           "_scope": "calibrated in the ANCHOR student's feature space and reused globally; it is "
                     "an anchor-calibrated global lambda, not a per-arm calibrated one."}
    guard9.write_result(ARTIFACT, out, "m9-warmfit")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for g in grid:
        print(f"  lambda {g['lambda']:g}: fit {g['fit_objective']:.5f}  val {g['val_objective']:.5f}")
    return out


def selected_lambda():
    """The predicate `nano.warm_start_head` consults. Refuses until the selection has run."""
    p = ARTIFACT
    if not p.exists():
        raise SystemExit("the warm-start lambda has not been selected on a training-only holdout "
                         "-- run m9src/warmfit.py (m9/LEDGER.md §3.2a)")
    blob = json.loads(p.read_text())
    if not guard9.eligible(blob):
        raise SystemExit("results/m9_warmfit.json is diagnostic or from a different lock")
    reg = guard9.registry()["warm_start"]["lambda"]
    if blob["selected_lambda"] != reg:
        raise SystemExit(f"the training-only selection chose lambda {blob['selected_lambda']:g} "
                         f"and the registry pins {reg:g} -- re-lock before running an arm")
    return blob["selected_lambda"]


if __name__ == "__main__":
    run()
