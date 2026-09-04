"""Serve `constella-zero` through FastEmbed the way the card does, and check it against numpy.

The card tells a reader to write `TextEmbedding("DylanCouzon/constella-zero")`. That name resolves
only where the model is registered in FastEmbed's built-in list, so this gate runs against the
branch carrying that registration:

    PYTHONPATH=/home/dylan/fastembed .venv/bin/python m11/release/verify_fastembed.py
    PYTHONPATH=/home/dylan/fastembed .venv/bin/python m11/release/verify_fastembed.py --negative-control

`model.onnx` pools (count-saturated sqrt weights) and L2-normalizes inside the graph and emits
(b, 1024). `OnnxTextEmbedding` passes 2-D output through and re-normalizes -- a no-op on unit
vectors -- so no bespoke class is needed; an earlier version of this work added one and it was
removed as dead weight.

An earlier version of this gate registered the model with `add_custom_model` instead. That is NOT
what ships: it cannot support `parallel>1` (the worker builds `OnnxTextEmbedding`, which cannot
resolve a runtime-registered name), and a card teaching it breaks the moment the model is built in
-- `add_custom_model` then raises `already registered` (Fable, 2026-09-03).
"""
import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
BUNDLE = Path(_args[0]) if _args else REPO / "work/release/zero-v1"
RESULT_DIR = REPO / "results"
NAME = "DylanCouzon/constella-zero"


@contextlib.contextmanager
def silence_fds():
    """Redirect fds 1 and 2 to /dev/null for the block, child processes included."""
    sys.stdout.flush(); sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1); os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for fd in saved + [null]:
            os.close(fd)


def staged_encoder(bundle):
    """Import the STAGED zero_encoder.py, not m11/release's copy (same rule as gate 4)."""
    spec = importlib.util.spec_from_file_location(
        "staged_zero_encoder", Path(bundle) / "zero_encoder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ZeroQueryEncoder(bundle, variant="int8")


def serve(bundle):
    from fastembed import TextEmbedding
    return TextEmbedding(NAME, specific_model_path=str(bundle))


def embed(model, texts, **kw):
    return np.asarray(list(model.embed(texts, **kw)), dtype=np.float32)


def require_registered():
    from fastembed import TextEmbedding
    if NAME not in {d.model for d in TextEmbedding._list_supported_models()}:
        sys.exit(f"REFUSED: {NAME} is not a built-in FastEmbed model here. Run with "
                 "PYTHONPATH pointing at the branch that registers it.")


def negative_control():
    """Prove the parity checks can fail: put the teacher's 8000-token truncation back.

    Gate 7 refuses this bundle at push time, so it reaches serving only by another route -- a
    hand-assembled directory, or files edited on the hub after the push. Note it does NOT crash:
    with `padding: null` shipped, FastEmbed pads dynamically and serves a silently wrong vector.
    """
    import shutil, tempfile
    require_registered()
    td = Path(tempfile.mkdtemp())
    d = td / "bundle"
    shutil.copytree(BUNDLE, d, ignore=shutil.ignore_patterns("__pycache__"))
    tok = json.loads((d / "tokenizer.json").read_text())
    tok["truncation"]["max_length"] = 8000
    (d / "tokenizer.json").write_text(json.dumps(tok, indent=1, ensure_ascii=False) + "\n")
    cfg = json.loads((d / "tokenizer_config.json").read_text())
    cfg["max_length"] = cfg["model_max_length"] = 8000
    (d / "tokenizer_config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")
    try:
        enc = staged_encoder(d)                    # unaffected: it reads config.json:preproc
        fx = [" ".join(["retrieval augmented generation"] * 400)]
        try:
            bad = float(np.abs(embed(serve(d), fx) - enc.encode(fx)).max())
            verdict, caught = f"served, max-abs {bad:.3e}", bad > 1e-5
        except Exception as e:
            verdict, caught = f"{type(e).__name__}: {str(e)[:80]}", True
        print(f"{'PASS' if caught else 'FAIL'}  NEGATIVE CONTROL: truncation 8000 detected"
              f"        {verdict}")
        return 0 if caught else 1
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main():
    import fastembed
    require_registered()
    enc = staged_encoder(BUNDLE)
    model = serve(BUNDLE)
    checks, res = [], {"fastembed_version": fastembed.__version__,
                       "fastembed_path": fastembed.__file__, "bundle": str(BUNDLE)}

    served_by = type(model.model).__name__
    res["served_by"] = served_by
    checks.append(("1 built-in name, served by the stock class",
                   served_by == "OnnxTextEmbedding", served_by))

    qs = json.loads((REPO / "work/dev/heldout-train.json").read_text())["q_texts"][:1024]
    got, want = embed(model, qs), enc.encode(qs)
    ng, nw = np.linalg.norm(got, axis=1), np.linalg.norm(want, axis=1)
    abs_d = float(np.abs(got - want).max())
    cos = float(((got * want).sum(1) / (ng * nw)).min())
    res["dev_queries"] = {"n": len(qs), "max_abs": abs_d, "min_cos": cos}
    checks.append((f"2 parity vs numpy on {len(qs)} real dev queries",
                   abs_d <= 1e-5 and cos >= 1 - 1e-6,
                   f"max-abs {abs_d:.3e}   min-cos {cos:.9f}"))

    # Dev queries are short, so check 2 alone passes under the WRONG tokenizer rule -- the rule
    # only bites past 512 tokens. That is what the negative control exploits.
    long_fx = [" ".join(["retrieval augmented generation"] * n) for n in (150, 200, 400)] + \
              [" ".join(str(i) for i in range(700))]
    long_abs = float(np.abs(embed(model, long_fx) - enc.encode(long_fx)).max())
    res["long_inputs_max_abs"] = long_abs
    checks.append((f"3 parity on {len(long_fx)} inputs past the 512-token rule",
                   long_abs <= 1e-5, f"max-abs {long_abs:.3e}"))

    solo = embed(model, [qs[0]])[0]
    inv = max(float(np.abs(embed(model, qs[:64], batch_size=1)
                           - embed(model, qs[:64], batch_size=64)).max()),
              float(np.abs(solo - embed(model, [qs[0], " ".join(["retrieval"] * 1200)])[0]).max()))
    res["batch_invariance_max_abs"] = inv
    checks.append(("4 batch invariance (bs 1 vs 64, and beside a long query)",
                   inv <= 1e-6, f"{inv:.3e}"))

    checks.append(("5 served vectors are unit-norm",
                   abs(float(ng.max()) - 1) <= 1e-5 and abs(float(ng.min()) - 1) <= 1e-5,
                   f"norms [{ng.min():.7f}, {ng.max():.7f}]"))

    # parallel>1 is the concrete gain from registering natively; add_custom_model cannot do it.
    # The workers spawn and re-import this file, so it must stay under the __main__ guard below.
    with silence_fds():
        try:
            par = embed(model, qs[:600], parallel=2, batch_size=64)
            pd_ = float(np.abs(par - want[:600]).max())
            par_ok, par_detail = pd_ <= 1e-5, f"max-abs {pd_:.3e}"
        except Exception as e:
            par_ok, par_detail = False, f"{type(e).__name__}: {str(e)[:90]}"
    res["parallel_2_max_abs"] = par_detail
    checks.append(("6 parallel=2 agrees with the numpy reference", par_ok, par_detail))

    ok = True
    print(f"\nfastembed {fastembed.__version__} ({fastembed.__file__})\nbundle {BUNDLE}")
    for name, passed, detail in checks:
        ok &= bool(passed)
        print(f"{'PASS' if passed else 'FAIL'}  {name:56s}  {detail}")
    res["pass"] = bool(ok)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "m11_fastembed_serving.json"
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"\n{'FASTEMBED SERVING OK' if ok else 'FASTEMBED SERVING FAILED'}   -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(negative_control() if "--negative-control" in sys.argv else main())
