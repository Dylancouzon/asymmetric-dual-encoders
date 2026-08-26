"""Three cost numbers for the released table, in decimal MB, per the M7 mandate:
  query asset   -- what the edge client must hold to encode a query
  doc index     -- bytes per 1M documents at the teacher's dimension
  hydration     -- cold load of the query asset into a usable state

Query-side latency is measured on CPU at batch 1, the edge condition, matching
bench/measure_cost.py's protocol so the M4 rows stay comparable.
"""
import json
import statistics
import time

import numpy as np
import torch

import encoders
from _paths import REPO, WORK
from table import Preproc, get_tokenizer, load_table
from teacher import TEACHER

QUERY = "how does high blood pressure medication affect kidney function"


def timed(fn, n=200, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def table_costs(npz_path, pre: Preproc, dim=None):
    dim = encoders.active().dim if dim is None else dim
    z = np.load(npz_path)
    V = z["rows_fp16"].shape[0]
    has_w = z["token_weights"].size > 0
    tokj = get_tokenizer().backend_tokenizer.to_str().encode()
    sizes = {
        "rows_fp16_mb": round(V * dim * 2 / 1e6, 2),
        "rows_int8_mb": round(V * dim / 1e6, 2),
        "int8_scale_mb": round(V * 4 / 1e6, 4),
        "token_weights_mb": round((V * 4 / 1e6) if has_w else 0.0, 4),
        "tokenizer_mb": round(len(tokj) / 1e6, 2),
    }
    sizes["query_asset_int8_mb"] = round(sizes["rows_int8_mb"] + sizes["int8_scale_mb"]
                                        + sizes["token_weights_mb"] + sizes["tokenizer_mb"], 2)
    sizes["query_asset_fp16_mb"] = round(sizes["rows_fp16_mb"] + sizes["token_weights_mb"]
                                         + sizes["tokenizer_mb"], 2)

    lat, hyd = {}, {}
    for variant in ("fp16", "int8"):
        t0 = time.perf_counter()
        m = load_table(npz_path, variant=variant, device="cpu")
        tok = get_tokenizer()
        hyd[variant] = round(time.perf_counter() - t0, 3)
        lat[variant] = round(timed(lambda: m.encode([QUERY], pre, tok=tok, device="cpu")), 4)
        del m
    return {"sizes_mb": sizes, "cpu_latency_ms_batch1": lat, "hydration_s": hyd,
            "vocab": int(V), "dim": int(dim), "learned_weights": bool(has_w)}


def doc_index_costs(dim=None):
    dim = encoders.active().dim if dim is None else dim
    """Bytes per 1M documents, so the table's index cost sits next to the comparators'."""
    per = lambda d, b: round(1_000_000 * d * b / 1e9, 2)
    return {"dim": dim,
            f"teacher_{dim}d_fp16_gb_per_1m": per(dim, 2),
            f"teacher_{dim}d_fp32_gb_per_1m": per(dim, 4),
            f"teacher_{dim}d_int8_gb_per_1m": per(dim, 1),
            "reference_lightretriever_1536d_fp16_gb_per_1m": per(1536, 2),
            "reference_bge_small_384d_fp16_gb_per_1m": per(384, 2),
            "reference_opensearch_sparse_gb_per_1m": 1.4,
            "note": "M5/M4 rows: LR 3.07, opensearch 1.4, bge-small 0.77, leaf/arctic-m 1.54 GB/1M"}


def main(npz_path, preproc="noprefix"):
    from table import NO_PREFIX, WITH_PREFIX
    pre = {"noprefix": NO_PREFIX, "prefix": WITH_PREFIX}[preproc]
    out = {"teacher": TEACHER, "table": str(npz_path),
           "query_side": table_costs(npz_path, pre), "doc_index": doc_index_costs()}
    (REPO / "results" / "m7_costs.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import sys
    main(WORK / "runs" / f"{sys.argv[1]}.npz", sys.argv[2] if len(sys.argv) > 2 else "noprefix")
