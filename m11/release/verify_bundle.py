"""The published bundle must reproduce the frozen query path bit-for-bit-ish.

Compares m11/release/zero_encoder.py (numpy, no torch -- what ships) against
m7src/table.py `QueryTable.encode` (torch -- what the M7 numbers were measured with),
on both released variants. Tolerance 1e-5 max-abs; the two differ only in float
summation order.

  .venv/bin/python m11/release/verify_bundle.py [bundle_dir] [--ref-table PATH]

`--ref-table` is the point of the gate (M11a T0): the reference side must be loaded from the
FROZEN SOURCE table, not from the bundle. Without it both sides read BUNDLE/model.npz and the
comparison is self-consistent for any bundle, wrong-but-coherent ones included.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "m7src"))

TOL = 1e-5

def load_encoder_class(bundle, staged):
    """Import ZeroQueryEncoder from the STAGED copy, not from m11/release.

    Codex, 2026-09-03: this file put `m11/release` on sys.path and imported from there, so the
    gate tested the source implementation while a different `zero_encoder.py` shipped. A staged
    file special-casing the card's one query passed every gate.
    """
    if not staged:
        sys.path.insert(0, str(REPO / "m11/release"))
        from zero_encoder import ZeroQueryEncoder
        print("WARNING: testing m11/release/zero_encoder.py, NOT the staged copy "
              "(pass --staged-encoder)")
        return ZeroQueryEncoder
    import importlib.util
    path = Path(bundle) / "zero_encoder.py"
    spec = importlib.util.spec_from_file_location("staged_zero_encoder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ZeroQueryEncoder


# Fable, 2026-09-03: the first ten fixtures let real changes through, because a gate is only as
# wide as what it tokenizes. An encoder dropping [UNK] or [PAD] rows from the bag -- a plausible
# "cleanup" -- was numerically identical on all ten; so were `max_input_chars_per_word: 100 -> 18`
# and `unk_token: "[UNK]" -> "[PAD]"` in the tokenizer. Each fixture below closes one of those.
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
    # --- added 2026-09-03
    "🜛 ᚠᚢᚦ 𐎠𐎢",                       # unmappable scripts -> [UNK]; binds unk_token
    "electroencephalographically " + "a" * 120,   # >100 chars in one word; binds
                                                  # max_input_chars_per_word
    "[PAD]",                                      # literal special tokens must be ordinary rows,
    "[PAD] [PAD] hello",                          #   not dropped and not masked
    "[CLS] [SEP] [UNK] [MASK]",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",   # deep subword splits
    "a",                                          # single content token
    "\u200b\u00a0\t\n",                         # whitespace-only unicode -> likely empty bag
    "Ω≈ç√∫˜µ≤≥÷",                                 # symbol soup
    "0 0 0 0 1 1 1 2",                            # repeated numerics, mixed multiplicity
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", nargs="?", default=str(REPO / "work/release/zero-v1"))
    ap.add_argument("--ref-table", default=None,
                    help="table the reference torch path loads; defaults to BUNDLE/model.npz, "
                         "which makes the comparison self-consistent -- pass the frozen source")
    ap.add_argument("--staged-encoder", action="store_true",
                    help="import zero_encoder.py from BUNDLE (what ships) instead of m11/release")
    args = ap.parse_args()
    BUNDLE = Path(args.bundle)
    ref_table = Path(args.ref_table) if args.ref_table else BUNDLE / "model.npz"

    from table import Preproc, load_table
    ZeroQueryEncoder = load_encoder_class(BUNDLE, args.staged_encoder)

    # The preproc rule comes from FREEZE.json, not from the bundle's own config: taking it from
    # the bundle let a changed rule be compared against itself (Codex, 2026-09-03).
    fz = json.loads((REPO / "m7/FREEZE.json").read_text())
    meta = json.loads((BUNDLE / "config.json").read_text())
    if meta["preproc"] != fz["preproc"] or meta["preproc_fingerprint"] != fz["preproc_fingerprint"]:
        print(f"FAIL  bundle preproc {meta['preproc']} != FREEZE.json {fz['preproc']}")
        return 1
    pre = Preproc(**fz["preproc"])
    self_cmp = ref_table.resolve() == (BUNDLE / "model.npz").resolve()
    print(f"reference table: {ref_table}"
          f"{'   (SELF-COMPARISON — pass --ref-table)' if self_cmp else ''}")
    ok = True
    for variant in ("int8", "fp16"):
        ref = load_table(ref_table, variant=variant, device="cpu")
        a = ref.encode(FIXTURES, pre)
        b = ZeroQueryEncoder(BUNDLE, variant=variant).encode(FIXTURES)
        dev = float(np.abs(a - b).max())
        cos = float((a * b).sum(1).min())
        # b=1 and the `str` overload are the shapes the model card actually uses, and neither is
        # exercised by the batch pass above (Fable, 2026-09-03).
        enc1 = ZeroQueryEncoder(BUNDLE, variant=variant)
        for one in (FIXTURES[0], FIXTURES[10], FIXTURES[12]):
            solo = enc1.encode(one)
            batched = enc1.encode([one])
            i = FIXTURES.index(one)
            d1 = max(float(np.abs(solo[0] - b[i]).max()), float(np.abs(batched[0] - b[i]).max()))
            if d1 > TOL:
                print(f"FAIL  {variant:5s}  b=1/str differs from the batch by {d1:.3e} on {one!r}")
                ok = False
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
