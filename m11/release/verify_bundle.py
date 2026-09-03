"""The published bundle must reproduce the frozen query path bit-for-bit-ish.

Compares m11/release/zero_encoder.py (numpy, no torch -- what ships) against
m7src/table.py `QueryTable.encode` (torch -- what the M7 numbers were measured with),
on both released variants. Tolerance 1e-5 max-abs; the two differ only in float
summation order.

  .venv/bin/python m11/release/verify_bundle.py [bundle_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "m7src"))
sys.path.insert(0, str(REPO / "m11/release"))

BUNDLE = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "work/release/zero-v1"
TOL = 1e-5

FIXTURES = [
    "what is a lookup table",
    "protein folding market impact",
    "argue both sides of a covid tax",
    "zzzqx",
    "",
    "   ",
    "the the the the the the",                       # count saturation must bite here
    "Comment configurer un encodeur asymétrique ?",   # non-ascii
    "COVID-19 vaccine efficacy in immunocompromised patients: a systematic review",
    " ".join(["retrieval augmented generation"] * 400),   # forces 512-token truncation
]


def main():
    from table import Preproc, load_table
    from zero_encoder import ZeroQueryEncoder

    meta = json.loads((BUNDLE / "config.json").read_text())
    pre = Preproc(**meta["preproc"])
    ok = True
    for variant in ("int8", "fp16"):
        ref = load_table(BUNDLE / "model.npz", variant=variant, device="cpu")
        a = ref.encode(FIXTURES, pre)
        b = ZeroQueryEncoder(BUNDLE, variant=variant).encode(FIXTURES)
        dev = float(np.abs(a - b).max())
        cos = float((a * b).sum(1).min())
        status = "PASS" if dev <= TOL else "FAIL"
        ok &= dev <= TOL
        print(f"{status}  {variant:5s}  max-abs {dev:.3e}   min-cosine {cos:.9f}")
        if dev > TOL:
            i = int(np.abs(a - b).max(1).argmax())
            print(f"      worst fixture [{i}]: {FIXTURES[i][:70]!r}")
    print("CONFORMANCE OK" if ok else "CONFORMANCE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
