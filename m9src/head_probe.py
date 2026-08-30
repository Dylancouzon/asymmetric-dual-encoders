"""Diagnostic: what does the head alone buy, before any backbone training?

M8's most reusable lesson is "screen in closed form before spending training chains". The M9.1
smoke showed a randomly-initialized `Linear(384, 1024)` head destroying the backbone's
representation for the first few thousand steps, which at ~1% of LEAF's dose is a large fraction
of the whole budget. So: fit the head in closed form (ridge from frozen mean-pooled student
outputs to teacher targets) and score THAT, with the backbone untouched.

Two readings, and they point opposite ways, which is why this is worth 3 minutes:
  * a strong closed-form score means a warm-started head is a cheap accelerator and the lock
    should be amended to use one for EVERY arm (identical across arms, so no contrast moves);
  * a weak one means the head is not the bottleneck and the lock stands as written.

This is a DIAGNOSTIC. It reads DEV, it sets no bar, and no screen decision may cite it.
"""
import json
import time

import numpy as np
import torch

import m9base
from m9base import RESULTS, WORK

import data as m9data   # noqa: E402
import eval9            # noqa: E402
import guard9           # noqa: E402
import nano             # noqa: E402

RUN_ID = "m9-head-probe-diag"   # the -diag suffix makes the artifact decision-ineligible

N_FIT = 60_000
SEED = 21
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


@torch.inference_mode()
def pooled(model, texts, batch_size=256, prefix=""):
    """Frozen mean-pooled backbone outputs, no head."""
    dev = next(model.parameters()).device
    h = model.backbone.config.hidden_size
    out = np.empty((len(texts), h), dtype=np.float32)
    order = np.argsort([len(t) for t in texts], kind="stable")
    for i in range(0, len(order), batch_size):
        sel = order[i:i + batch_size]
        b = model.tok([prefix + texts[j] for j in sel], padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            hs = model.backbone(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(hs.dtype)
        out[sel] = ((hs * m).sum(1) / m.sum(1).clamp(min=1e-9)).float().cpu().numpy()
    return out


class Fitted:
    """A Nano whose head is a closed-form ridge solution and whose backbone is untouched."""

    def __init__(self, model, W, b):
        self.model, self.W, self.b = model, W, b

    def encode_queries(self, texts):
        v = pooled(self.model, texts) @ self.W + self.b
        return (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def run(student_key="bge-small-en-v1.5"):
    t0 = time.time()
    texts = json.loads((WORK / "m9_screen_queries.json").read_text())
    rows = np.load(WORK / "m9_screen_rows.npy")
    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(len(texts), size=N_FIT, replace=False))

    model = nano.Nano(student_key).cuda().eval()
    X = pooled(model, [texts[i] for i in sel])
    Y = np.asarray(m9data.stella_query_targets()[rows[sel]], dtype=np.float32)
    Xc = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])          # bias column
    G = Xc.T @ Xc
    R = Xc.T @ Y
    print(f"fit matrices in {time.time()-t0:.0f}s  X{X.shape} Y{Y.shape}", flush=True)

    out = {"student": student_key, "n_fit": N_FIT, "seed": SEED,
           "hidden": int(X.shape[1]), "arms": {}}
    scale = float(np.trace(G) / G.shape[0])
    for lam in LAMBDAS:
        A = np.linalg.solve(G + lam * scale * np.eye(G.shape[0], dtype=np.float32), R)
        W, b = A[:-1], A[-1]
        resid = float(np.mean(np.sum((Xc @ A - Y) ** 2, axis=1)))
        per = eval9.eval_student(Fitted(model, W, b), eval9.INCUMBENT,
                                 comps=eval9.components("SCREEN3"))
        m, means = eval9.macros(per, eval9.INCUMBENT)["SCREEN3"].values()
        out["arms"][f"ridge-{lam:g}"] = {"lambda": lam, "train_sq_l2": round(resid, 5),
                                         "screen3_macro": m, "means": means}
        print(f"  lambda {lam:g}: train L2 {resid:.5f}  SCREEN3 {m:.5f}", flush=True)

    sym = json.loads((RESULTS / "m9_dev_symmetric_stella-400M-v5.json").read_text())
    ceil3 = sym["macros"]["SCREEN3"]["macro"]
    best = max(out["arms"].values(), key=lambda a: a["screen3_macro"])
    out["ceiling_screen3"] = ceil3
    out["best"] = {**best, "retention": round(best["screen3_macro"] / ceil3, 4)}
    out["seconds"] = round(time.time() - t0, 1)
    out["_status"] = ("DIAGNOSTIC. Reads DEV, sets no bar, and no screen decision may cite it. "
                      "lambda is selected on the same surface it reports, so the retention here "
                      "is an OPTIMISTIC upper bound for a closed-form head.")
    guard9.begin_run(RUN_ID)
    guard9.write_result(RESULTS / "m9_head_probe.json", out, RUN_ID)
    print(json.dumps({"best": out["best"], "ceiling": ceil3}, indent=1))
    return out


if __name__ == "__main__":
    run()
