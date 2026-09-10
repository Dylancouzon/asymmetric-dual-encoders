"""M10.0-e — the same-init calibration. Three arms whose PAIRINGS are the measurement.

Registered scope is `m10/LEDGER.md` §M10.0-e and it is narrow on purpose, because a Codex pass
refused the first framing: this brackets contrasts whose two arms share backbone, tokenizer, init,
seed, data order and warm-start head — **B, D, G and C, and nothing else**. Family F compares two
students that share only the teacher target, which in per-query-correlation terms IS the
unrelated-models case, so F and E are read against the 0.008619 already measured.

| arm | differs by | its pairing measures |
|---|---|---|
| P0 | — | — |
| P1 | seed 1 | `SD(P0-P1)` and `\\|macro(P0)-macro(P1)\\|`: the SEED floor |
| P2 | peak LR 8e-5 | `SD(P0-P2)`: a same-init contrast's paired width |

**Two statistics, not one.** The bootstrap distance is query-sampling noise for a FIXED pair; the
seed effect is the point estimate `|macro(P0) - macro(P1)|`, which the bootstrap cannot see because
it resamples queries and not seeds. A screen that "resolves" less than its own seed effect has
resolved noise, so both are reported.

**Warm start:** identical across all three arms, ridge on pooled features at a FIXED lambda = 1e-4
(`m9/registry.json` warm_start.lambda). `m9src/warmfit.selected_lambda()` refuses to serve M9's
value under a different lock, correctly — and a calibration needs the three arms comparable to each
other, not to the screen's exact recipe. Identical treatment cannot bias a paired width. Disclosed.

Peak LR is the lever precisely BECAUSE it is not one of the thirteen registered contrasts, so
nothing here can leak a screen verdict. COV only: no DEV-6, FORMS-12 or CUREv1 read on any P arm.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))
OUT = REPO / "work" / "m10calib"

import numpy as np
import torch

DOSE = 5_000_000
BATCH = 32
N_FIT, FIT_SEED, LAM = 60_000, 21, 1e-4
ARMS = {"P0": {"seed": 0, "peak": 1e-4},
        "P1": {"seed": 1, "peak": 1e-4},
        "P2": {"seed": 0, "peak": 8e-5}}


def build(sub=None, verbose=True):
    """-> (texts, targets, tokenizer-ready ids are built per arm since the tokenizer is shared)."""
    import data10 as D
    texts, rows, meta = D.m9_query_pool()
    if sub:
        texts, rows = texts[:sub], rows[:sub]
    T = D.m9_query_targets(rows)
    if verbose:
        print(f"pool {len(texts):,} queries, targets {T.shape}", flush=True)
    return texts, T, meta


def warm_start(model, texts, T, verbose=True):
    """Identical for every arm: ridge on pooled features at the fixed lambda."""
    import data10 as D
    import nano10 as N
    rng = np.random.default_rng(FIT_SEED)
    sel = np.sort(rng.choice(len(texts), size=min(N_FIT, len(texts)), replace=False))
    t0 = time.time()
    X = N.pooled_features(model, [texts[i] for i in sel])
    rec = N.warm_start_linear(model, X, T[sel], lam=LAM)
    rec["seconds"] = round(time.time() - t0, 1)
    if verbose:
        print(f"  warm start: n_fit {rec['n_fit']:,}, lambda {LAM}, "
              f"train objective {rec['train_objective']:.4f} ({rec['seconds']:.0f}s)", flush=True)
    return rec


def run_arm(name, texts, T, dose=DOSE, verbose=True):
    import data10 as D
    import nano10 as N
    import trainer10 as Tr
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = ARMS[name]
    torch.manual_seed(cfg["seed"])
    m = N.Nano10("bge-small", n_layers=3).cuda()
    ws = warm_start(m, texts, T, verbose=verbose)
    ids = D.pretokenize(m.tok, texts, max_len=512, verbose=verbose, label=name)
    q = D.Stream(ids, T, pad_id=m.tok.pad_token_id, batch_size=BATCH, seed=cfg["seed"])
    steps = dose // BATCH
    if verbose:
        print(f"  {name}: {steps:,} steps x {BATCH} = {steps * BATCH:,} examples, "
              f"peak {cfg['peak']}, seed {cfg['seed']}", flush=True)
    r = Tr.train_arm(m, D.batch_fn(q, q, pattern="100/0"), total_steps=steps, pattern="100/0",
                     peak=cfg["peak"], loss_name="squared_l2", seed=cfg["seed"],
                     device="cuda", log_every=max(steps // 20, 1),
                     ckpt_path=OUT / f"{name}.pt", ckpt_every=max(steps // 10, 1))
    r["warm_start"], r["arm"] = ws, cfg
    r.pop("losses", None)
    Tr.save(OUT / f"{name}.pt", m, torch.optim.AdamW(m.parameters()), steps)
    (OUT / f"{name}.json").write_text(json.dumps(r, indent=2, default=str))
    return m, r


def cov_of(model, verbose=True):
    import cov_eval10 as C
    return C.score_student(lambda t: model.encode_queries(t, batch_size=256), verbose=verbose)


if __name__ == "__main__":
    which = sys.argv[1:] or list(ARMS)
    texts, T, meta = build()
    for name in which:
        t0 = time.time()
        print(f"\n=== {name}", flush=True)
        m, r = run_arm(name, texts, T)
        per = cov_of(m)
        mac, fam, um = __import__("cov_eval10").macro(per)
        json.dump({"per_unit_query": per, "macro": mac, "by_family": fam, "by_unit": um},
                  open(OUT / f"{name}_cov.json", "w"))
        print(f"{name}: COV macro {mac:.4f}  ({time.time() - t0:.0f}s total)", flush=True)
        del m
        torch.cuda.empty_cache()
