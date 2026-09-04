"""T4 gate: serve `zero` through fastembed and check it against the numpy encoder.

The release claims a fastembed caller gets the same vectors as `zero_encoder.py`. Nothing
proved that: T3 settled the two *routes* (a `(b,1024)` graph is accepted by `PoolingType.
DISABLED`; the masked mean on the per-token graph recovers the same direction), but neither
measurement ran a query through `TextEmbedding` end to end.

Primary route -- the one both model cards show:

    PoolingType.DISABLED + normalization=False on the pooled `model.onnx`

`model.onnx` already pools (count-saturated sqrt weights) and L2-normalizes, and carries the
degenerate-bag fallback. DISABLED returns the graph output untouched, so the fallbacks survive;
MEAN + normalize would discard them (harmless in practice -- no table row has norm <= EPS,
min 0.196 -- but it is a different rule and the card should not describe one and ship the other).

Checks (1-5 gate; 6-7 are recorded facts, not gates):
  1 registration + serving works at all
  2 parity vs the STAGED numpy encoder on 1024 real dev queries
  3 batch invariance: batch_size 1 vs 64 vs a mixed batch with a >512-token query
  4 output is unit-norm (the graph normalizes; fastembed must not touch it)
  5 the MEAN route on model_tokens.onnx agrees in DIRECTION (its own norm is fastembed's)
  6 parallel>1 raises -- CustomTextEmbedding has no _get_worker_class override (T3, measured)
  7 fastembed version + which tokenizer rule its loader applied

    .venv/bin/python m11/release/verify_fastembed.py [bundle_dir]
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
NAME = "constella/zero-v1-gate"          # registry key only; not the published repo id
DIM = 1024


@contextlib.contextmanager
def silence_fds():
    """Redirect fd 1 and 2 to /dev/null for the block, child processes included."""
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


def register(model_file, name, pooling, normalization):
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType
    TextEmbedding.add_custom_model(
        model=name,
        pooling=getattr(PoolingType, pooling),
        normalization=normalization,
        sources=ModelSource(hf="DylanCouzon/constella-zero"),
        dim=DIM,
        model_file=model_file,
        license="mit",
        size_in_gb=0.031,
    )
    return TextEmbedding(model_name=name, specific_model_path=str(BUNDLE))


def embed(model, texts, **kw):
    return np.asarray(list(model.embed(texts, **kw)), dtype=np.float32)


def negative_control():
    """Prove check 2b can fail: put the teacher's 8000-token truncation back and re-run it.

    A gate that cannot fail is decoration. Gate 7 (`verify_tokenizer.py`) refuses this bundle at
    push time, so this is the accident that reaches serving only by another route -- a
    hand-assembled directory, or files edited on the hub after the push.
    """
    global BUNDLE
    import shutil, tempfile
    td = Path(tempfile.mkdtemp())
    d = td / "zero-v1"
    shutil.copytree(BUNDLE, d, ignore=shutil.ignore_patterns("__pycache__"))
    tok = json.loads((d / "tokenizer.json").read_text())
    tok["truncation"]["max_length"] = 8000
    (d / "tokenizer.json").write_text(json.dumps(tok, indent=1, ensure_ascii=False) + "\n")
    cfg = json.loads((d / "tokenizer_config.json").read_text())
    cfg["max_length"] = cfg["model_max_length"] = 8000
    (d / "tokenizer_config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")

    real, BUNDLE = BUNDLE, d
    try:
        enc = staged_encoder(d)                       # unaffected: it reads config.json:preproc
        model = register("model.onnx", NAME + "-negctl", "DISABLED", False)
        fx = [" ".join(["retrieval augmented generation"] * 400)]
        try:
            got = embed(model, fx)
            bad = float(np.abs(got - enc.encode(fx)).max())
            verdict = f"served, max-abs {bad:.3e}"
            caught = bad > 1e-5
        except Exception as e:                        # ragged batch / shape error is also a catch
            verdict, caught = f"{type(e).__name__}: {str(e)[:80]}", True
        print(f"{'PASS' if caught else 'FAIL'}  NEGATIVE CONTROL: truncation 8000 is detected"
              f"        {verdict}")
        return 0 if caught else 1
    finally:
        BUNDLE = real
        shutil.rmtree(td, ignore_errors=True)


def main():
    import fastembed
    # The fork carries the SAME __version__ as the release it branched from, so the version
    # string cannot tell the two runs apart -- record the import path, and key the result file
    # on it, or a fork run silently overwrites the stock one.
    src = "fork" if not fastembed.__file__.startswith(str(REPO)) else "pypi"
    result = RESULT_DIR / f"m11_fastembed_serving_{src}.json"
    res = {"fastembed_version": fastembed.__version__, "fastembed_path": fastembed.__file__,
           "fastembed_source": src, "bundle": str(BUNDLE)}
    enc = staged_encoder(BUNDLE)
    checks = []

    # 1 -- it serves at all, through the public TextEmbedding entry point
    model = register("model.onnx", NAME, "DISABLED", False)
    probe = embed(model, ["what is a lookup table"])
    checks.append(("1 registers and serves via TextEmbedding",
                   probe.shape == (1, DIM), f"shape {probe.shape}"))

    # 2 -- parity on real dev queries. Same set and thresholds as export_onnx.py check 2, so a
    #      regression here is attributable to the serving path and not to the graph.
    qs = json.loads((REPO / "work/dev/heldout-train.json").read_text())["q_texts"][:1024]
    got, want = embed(model, qs), enc.encode(qs)
    ng, nw = np.linalg.norm(got, axis=1), np.linalg.norm(want, axis=1)
    abs_d = float(np.abs(got - want).max())
    cos = float(((got * want).sum(1) / (ng * nw)).min())
    res["dev_queries"] = {"n": len(qs), "max_abs": abs_d, "min_cos": cos,
                          "fastembed_norms": [float(ng.min()), float(ng.max())]}
    checks.append((f"2 parity vs numpy on {len(qs)} real dev queries",
                   abs_d <= 1e-5 and cos >= 1 - 1e-6,
                   f"max-abs {abs_d:.3e}   min-cos {cos:.9f}"))

    # 2b -- parity on inputs LONGER than the frozen 512, which is where the tokenizer rule
    #       actually bites. Dev queries are short, so check 2 alone passes under the teacher's
    #       unsanitised truncation (8000) -- the one accident this route is exposed to. The
    #       numpy encoder truncates at 512; a served vector must agree past that boundary.
    long_fx = [" ".join(["retrieval augmented generation"] * n) for n in (150, 200, 400)] + \
              [" ".join(str(i) for i in range(700))]
    lgot, lwant = embed(model, long_fx), enc.encode(long_fx)
    long_abs = float(np.abs(lgot - lwant).max())
    res["long_inputs_max_abs"] = long_abs
    checks.append((f"2b parity on {len(long_fx)} inputs past the 512-token rule",
                   long_abs <= 1e-5, f"max-abs {long_abs:.3e}"))

    # 3 -- batch invariance. fastembed picks its own batches, so a query's vector must not
    #      depend on what it was batched with; a >512-token neighbour is the padding stressor.
    long_q = " ".join(["retrieval augmented generation"] * 400)
    solo = embed(model, [qs[0]])[0]
    b1 = embed(model, qs[:64], batch_size=1)
    b64 = embed(model, qs[:64], batch_size=64)
    mixed = embed(model, [qs[0], long_q])[0]
    inv = max(float(np.abs(b1 - b64).max()), float(np.abs(solo - mixed).max()))
    res["batch_invariance_max_abs"] = inv
    checks.append(("3 batch invariance (bs 1 vs 64, and beside a 1200-token query)",
                   inv <= 1e-6, f"{inv:.3e}"))

    # 4 -- normalization=False, so what comes back must already be unit-norm: if it is not,
    #      the graph's own normalize did not run and the card's `normalization=False` is wrong.
    checks.append(("4 served vectors are unit-norm without fastembed normalizing",
                   abs(float(ng.max()) - 1) <= 1e-5 and abs(float(ng.min()) - 1) <= 1e-5,
                   f"norms [{ng.min():.7f}, {ng.max():.7f}]"))

    # 5 -- the alternative route, kept measured so the card can say what it costs. fastembed's
    #      masked mean divides by the real token count -- a positive scalar its normalize
    #      annihilates -- so this agrees in direction, not necessarily to max-abs on the norm.
    tok_model = register("model_tokens.onnx", NAME + "-tokens", "MEAN", True)
    tgot = embed(tok_model, qs[:256])
    tcos = float(((tgot * want[:256]).sum(1)
                  / (np.linalg.norm(tgot, axis=1) * nw[:256])).min())
    res["mean_route_min_cos"] = tcos
    checks.append(("5 MEAN route on model_tokens.onnx agrees in direction",
                   tcos >= 1 - 1e-6, f"min-cos {tcos:.9f}"))

    # 6 -- recorded, NOT a gate: parallel>1 cannot work for add_custom_model repos in 0.8.0.
    #      CustomTextEmbedding does not override _get_worker_class, so each worker constructs
    #      OnnxTextEmbedding, which cannot resolve a runtime-registered name. Every worker
    #      retries and dumps a traceback from its OWN process, so the noise has to be silenced
    #      at the fd level -- a Python-level redirect does not reach a child.
    with silence_fds():
        try:
            embed(model, qs[:300], parallel=2, batch_size=64)
            par = "unexpectedly succeeded"
        except Exception as e:
            par = f"{type(e).__name__}: {str(e)[:120]}"
    res["parallel_2"] = par
    print(f"NOTE  parallel=2 -> {par}")

    # 7 -- what fastembed's loader made of the shipped tokenizer, alongside the vectors it gave
    from fastembed.common.preprocessor_utils import load_tokenizer
    tok, _ = load_tokenizer(BUNDLE)
    res["loader_tokenizer"] = {
        "truncation_max_length": tok.truncation and tok.truncation["max_length"],
        "padding_length": None if tok.padding is None else tok.padding.get("length"),
    }
    print(f"NOTE  loader tokenizer -> {res['loader_tokenizer']}")

    ok = True
    print(f"\nfastembed {fastembed.__version__} ({src}: {fastembed.__file__})"
          f"\nbundle {BUNDLE}")
    for name, passed, detail in checks:
        ok &= bool(passed)
        print(f"{'PASS' if passed else 'FAIL'}  {name:60s}  {detail}")
    res["pass"] = bool(ok)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"\n{'FASTEMBED SERVING OK' if ok else 'FASTEMBED SERVING FAILED'}   -> {result}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(negative_control() if "--negative-control" in sys.argv else main())
