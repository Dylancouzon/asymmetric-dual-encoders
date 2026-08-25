"""Sequential experiment driver. Appends every run -- including stopped, failed and OOM ones --
to m7/RESULTS.md, which is the experiment ledger the mandate requires.
"""
import json
import sys
import traceback
from dataclasses import asdict, replace

import torch

from _paths import REPO, WORK
from train import Cfg, run

RESULTS = REPO / "m7" / "RESULTS.md"


def append_row(run_id, cfg, dev, verdict, extra=""):
    with open(RESULTS, "a") as f:
        f.write(f"| {run_id} | `work/runs/{run_id}.json` | "
                f"{'—' if dev is None else f'{dev:.4f}'} | {verdict}{(' — ' + extra) if extra else ''} |\n")


def one(run_id, base=None, **over):
    cfg = replace(base or Cfg(), run_id=run_id, **over)
    (WORK / "runs").mkdir(parents=True, exist_ok=True)
    try:
        dev, model, hist = run(cfg)
        append_row(run_id, cfg, dev, "ok")
        del model
        torch.cuda.empty_cache()
        return dev
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        append_row(run_id, cfg, None, "OOM", str(e)[:120])
        print(f"[{run_id}] OOM: {e}", flush=True)
    except Exception as e:
        append_row(run_id, cfg, None, "FAILED", f"{type(e).__name__}: {str(e)[:120]}")
        print(f"[{run_id}] FAILED\n{traceback.format_exc()}", flush=True)
    return None


def grid(name, base, variants):
    """variants: {suffix: {field: value}}. Runs them in order, returns {run_id: dev macro}."""
    out = {}
    for suffix, over in variants.items():
        rid = f"{name}-{suffix}"
        print(f"\n{'='*80}\n{rid}: {json.dumps(over)}\n{'='*80}", flush=True)
        out[rid] = one(rid, base, **over)
    print(f"\n[{name}] " + "  ".join(f"{k}={'—' if v is None else f'{v:.4f}'}" for k, v in out.items()),
          flush=True)
    return out


if __name__ == "__main__":
    print("import this module and call grid()/one(); see m7/RESULTS.md for the ledger")
