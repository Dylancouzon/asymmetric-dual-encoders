"""Compose the required M18 step-500 serving-parity receipt from existing export gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

import protocol
import system
from common import admit_read, registry, sha_file, write_json


def _gates(bundle):
    path = Path(bundle) / "gates.json"
    return json.loads(admit_read(path).read_text()), sha_file(path)


def build(v0_bundle, step0_bundle, trained_bundle, out):
    reg = registry()
    rows, _ = protocol.load_surface("development")
    texts = [q["text"] for q in rows]
    base_vocab = int(reg["models"]["zero_v1"]["vocab"])
    tok = Tokenizer.from_file(str(admit_read(Path(v0_bundle) / "tokenizer.json")))
    unchanged = [text for text in texts if all(i < base_vocab for i in tok.encode(text).ids)]
    v1 = system._query_encoder(reg["models"]["zero_v1"]["source_path"])
    v0 = system._query_encoder(v0_bundle)
    v1_q, v0_q = v1.encode(unchanged), v0.encode(unchanged)
    v0_v1_error = float(np.abs(v1_q - v0_q).max()) if unchanged else 0.0
    step0, step0_sha = _gates(step0_bundle)
    trained, trained_sha = _gates(trained_bundle)
    tol = float(reg["serving"]["int8_error_max_abs"])
    report = {
        "_schema": "m18-step500-parity-v1",
        "development_queries": len(texts),
        "queries_without_added_tokens": len(unchanged),
        "v0_vs_v1_unchanged_query_max_abs": v0_v1_error,
        "v0_vs_v1_tolerance": tol,
        "step0": {"bundle": str(step0_bundle), "gates_sha256": step0_sha,
                  "training_forward_vs_serving_max_abs": step0["training_forward"]["max_abs"],
                  "loader_parity_max_abs": step0["loader"]["loader_parity_max_abs"],
                  "int8_vs_float_max_abs": step0["loader"]["vs_exported_float_rows_max_abs"]},
        "trained": {"bundle": str(trained_bundle), "gates_sha256": trained_sha,
                    "training_forward_vs_serving_max_abs": trained["training_forward"]["max_abs"],
                    "loader_parity_max_abs": trained["loader"]["loader_parity_max_abs"],
                    "int8_vs_float_max_abs": trained["loader"]["vs_exported_float_rows_max_abs"]},
    }
    parity_tol = float(reg["serving"]["loader_parity_max_abs"])
    report["pass"] = bool(
        v0_v1_error <= tol
        and step0["training_forward"]["max_abs"] <= parity_tol
        and trained["training_forward"]["max_abs"] <= parity_tol
        and step0["loader"]["loader_parity_max_abs"] <= parity_tol
        and trained["loader"]["loader_parity_max_abs"] <= parity_tol
        and step0["loader"]["vs_exported_float_rows_max_abs"] <= tol
        and trained["loader"]["vs_exported_float_rows_max_abs"] <= tol)
    write_json(out, report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v0", required=True)
    ap.add_argument("--step0", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    print(json.dumps(build(args.v0, args.step0, args.trained, args.out), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
