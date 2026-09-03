"""T1 gate: the shipped tokenizer files give a fastembed caller the FROZEN rule.

stella's own tokenizer files declare truncation 8000 (`min(model_max_length, max_length)`) and
fixed-512 padding. `fastembed.common.preprocessor_utils.load_tokenizer` honours both, and our
numpy path honours neither -- so before T1 a fastembed caller got a tokenization that is not the
one the table was distilled under, and not the one the document index was built with.

This checks what `load_tokenizer` ACTUALLY returns for the built bundle, not what the JSON says:

  1. truncation max_length == 512                      (the frozen preproc rule)
  2. padding is dynamic (batch-longest), not Fixed      (no [PAD] rows, no ragged batch)
  3. a >512-token text truncates to exactly 512
  4. a mixed batch [long, short] is rectangular         (the upstream-defect symptom)
  5. ids for short texts, with padded positions stripped by the attention mask, are
     IDENTICAL to the numpy encoder's own tokenizer -- dynamic padding pads to batch-longest,
     so this is the check that the mask (which pooling relies on) recovers the frozen bag

That these checks can fail is proved by `test_gates.py`, which puts the teacher's unsanitised
values back and requires each to be caught.

  .venv/bin/python m11/release/verify_tokenizer.py [bundle_dir]
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
STAGED = "--staged-encoder" in sys.argv
BUNDLE = Path(_args[0]) if _args else REPO / "work/release/zero-v1"

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

LONG = " ".join(["retrieval augmented generation"] * 400)      # ~1200 wordpieces
SHORT = "hello world"
PROBE = ["what is a lookup table", "zzzqx", "Comment configurer un encodeur asymétrique ?"]


def describe(model_dir):
    """What fastembed's own loader makes of a directory."""
    from fastembed.common.preprocessor_utils import load_tokenizer
    tok, _ = load_tokenizer(Path(model_dir))
    batch = tok.encode_batch([LONG, SHORT])
    return {
        "truncation_max_length": tok.truncation and tok.truncation["max_length"],
        "padding": None if tok.padding is None else {
            "length": tok.padding.get("length"), "direction": tok.padding.get("direction")},
        "len_long_alone": len(tok.encode(LONG).ids),
        "mixed_batch_lengths": [len(e.ids) for e in batch],
        "mixed_batch_rectangular": len({len(e.ids) for e in batch}) == 1,
        "short_ids_masked": [[i for i, m in zip(e.ids, e.attention_mask) if m]
                             for e in tok.encode_batch(PROBE)],
    }




def main():
    ZeroQueryEncoder = load_encoder_class(BUNDLE, STAGED)

    ours = describe(BUNDLE)
    enc = ZeroQueryEncoder(BUNDLE, variant="int8")
    ref_ids = [e.ids for e in enc.tokenizer.encode_batch(PROBE)]

    checks = [
        ("1 truncation is the frozen 512", ours["truncation_max_length"] == 512,
         f"{ours['truncation_max_length']}  (teacher ships 8000)"),
        ("2 padding is dynamic, not Fixed",
         ours["padding"] is not None and ours["padding"]["length"] is None,
         f"{ours['padding']}  (teacher ships Fixed 512)"),
        ("3 a 1200-token text truncates to 512", ours["len_long_alone"] == 512,
         f"{ours['len_long_alone']}"),
        ("4 a mixed [long, short] batch is rectangular", ours["mixed_batch_rectangular"],
         f"{ours['mixed_batch_lengths']}"),
        ("5 masked ids match the numpy encoder", ours["short_ids_masked"] == ref_ids,
         f"{len(PROBE)} probes, lengths {[len(x) for x in ours['short_ids_masked']]}"),
    ]
    ok = True
    import fastembed
    print(f"fastembed {fastembed.__version__}   bundle {BUNDLE}")
    for name, passed, detail in checks:
        ok &= bool(passed)
        print(f"{'PASS' if passed else 'FAIL'}  {name:46s}  {detail}")
    print("TOKENIZER OK" if ok else "TOKENIZER FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
