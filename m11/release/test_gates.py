"""Each gate must refuse the bundle it exists to refuse, and for the stated reason.

Passing gates prove nothing on their own -- the previous eight all passed while gate 4 compared
the bundle against itself. Each test below breaks the bundle in a way that could plausibly HAPPEN
(a stale staging dir, a hand-edit, the teacher's tokenizer copied unsanitised, a card edited
without being run) and asserts the gate catches it. `test_positive_control` runs the untouched
bundle through every gate, so "refuses everything" cannot masquerade as success.

  .venv/bin/python m11/release/test_gates.py
"""
import json
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import push

REPO_ID = "DylanCouzon/constella-zero"
TESTS = []


def attempt(fn, expect):
    """None if fn refused with a message matching `expect`, else why the test failed."""
    try:
        fn()
    except SystemExit as e:
        if e.code in (0, None):
            return "exited 0"
        if not re.search(expect, str(e.code), re.I):
            return f"refused for the wrong reason: {str(e.code)[:160]!r} !~ /{expect}/"
        return None
    except Exception as e:
        return f"raised {type(e).__name__}: {e}"          # a crash is not a refusal
    return "did not refuse"


@contextmanager
def staged():
    """A throwaway copy of the built bundle, wired in as push.OUT."""
    td = Path(tempfile.mkdtemp())
    d = td / "zero-v1"
    shutil.copytree(push.OUT, d, ignore=shutil.ignore_patterns("__pycache__"))
    out, push.OUT = push.OUT, d
    try:
        yield d
    finally:
        push.OUT = out
        shutil.rmtree(td, ignore_errors=True)


def test(name, expect=None):
    def deco(f):
        TESTS.append((name, expect, f))
        return f
    return deco


@test("POSITIVE CONTROL: the untouched bundle passes every gate")
def t_positive_control():
    push.gates_only(REPO_ID)


@test("gate 1 catches a stale staging dir", "staged model.npz hashes")
def t_stale_table():
    with staged() as d:
        import numpy as np
        z = dict(np.load(d / "model.npz"))
        z["rows_int8"] = np.zeros_like(z["rows_int8"])
        np.savez(d / "model.npz", **z)
        return attempt(lambda: push.gate_artifact(push.freeze()), "staged model.npz hashes")


@test("gate 4 tests the STAGED encoder, not m11/release's copy",
      "does not reproduce the frozen query path")
def t_staged_encoder():
    """The bug Codex found: both gates imported the source file, so a different shipped
    zero_encoder.py was never executed. Here the staged copy drops [UNK] and [PAD] rows from the
    bag -- a plausible cleanup, and identical to the frozen path on ordinary queries."""
    with staged() as d:
        src = (d / "zero_encoder.py").read_text()
        line = "        uniq, counts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)"
        assert line in src
        (d / "zero_encoder.py").write_text(src.replace(line,
            "        ids = [i for i in ids if i not in (0, 100)]\n"
            "        if not ids:\n            return self._fallback\n" + line))
        return attempt(lambda: push.gate_conformance(push.freeze()),
                       "does not reproduce the frozen query path")


@test("gate 4 catches an encoder that is wrong only at b=1",
      "does not reproduce the frozen query path")
def t_encoder_b1():
    """b=1 is the card's own usage and the batch pass does not exercise it, so an encoder correct
    for len(texts) > 1 and wrong for a single text passed every gate (Codex, 2026-09-03)."""
    with staged() as d:
        src = (d / "zero_encoder.py").read_text()
        line = "        out = np.empty((len(texts), self.dim), dtype=np.float32)"
        assert line in src
        (d / "zero_encoder.py").write_text(src.replace(line, line +
            "\n        if len(texts) == 1:\n"
            "            out[0] = 0.0\n            out[0, 0] = 1.0\n            return out"))
        return attempt(lambda: push.gate_conformance(push.freeze()),
                       "does not reproduce the frozen query path")


@test("gate 4 reads the preproc rule from FREEZE.json, not from the bundle",
      "does not reproduce the frozen query path")
def t_preproc_drift():
    with staged() as d:
        cfg = json.loads((d / "config.json").read_text())
        cfg["preproc"]["max_length"] = 511
        (d / "config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")
        return attempt(lambda: push.gate_conformance(push.freeze()),
                       "does not reproduce the frozen query path")


@test("gate 5 catches a file that should not ship", r"extra=\['notes\.txt'\]")
def t_extra_file():
    with staged() as d:
        (d / "notes.txt").write_text("scratch")
        return attempt(lambda: push.gate_manifest(), r"extra=\['notes\.txt'\]")


@test("gate 5 catches a missing file", r"missing=\['special_tokens_map\.json'\]")
def t_missing_file():
    with staged() as d:
        (d / "special_tokens_map.json").unlink()
        return attempt(lambda: push.gate_manifest(),
                       r"missing=\['special_tokens_map\.json'\]")


@test("gate 7 catches the teacher's 8000-token truncation surviving the build",
      "does not give a fastembed caller the frozen rule")
def t_unsanitised_truncation():
    with staged() as d:
        cfg = json.loads((d / "tokenizer_config.json").read_text())
        cfg["max_length"], cfg["model_max_length"] = 8000, 32768
        (d / "tokenizer_config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")
        return attempt(push.gate_tokenizer,
                       "does not give a fastembed caller the frozen rule")


@test("gate 7 catches the teacher's fixed-512 padding surviving the build",
      "does not give a fastembed caller the frozen rule")
def t_unsanitised_padding():
    with staged() as d:
        tok = json.loads((d / "tokenizer.json").read_text())
        tok["padding"] = {"strategy": {"Fixed": 512}, "direction": "Right",
                          "pad_to_multiple_of": None, "pad_id": 0, "pad_type_id": 0,
                          "pad_token": "[PAD]"}
        (d / "tokenizer.json").write_text(json.dumps(tok, indent=1, ensure_ascii=False) + "\n")
        return attempt(push.gate_tokenizer,
                       "does not give a fastembed caller the frozen rule")


@test("gate 6 catches a card whose python raises", "raises")
def t_card_raises():
    """T5: the shipped card did exactly this -- `enc.encode([q])[0]` on a (1,1024) array."""
    with staged() as d:
        (d / "README.md").write_text(
            '```python\nfrom huggingface_hub import snapshot_download\n'
            f'd = snapshot_download("{REPO_ID}")\nraise ValueError("boom")\n```\n')
        return attempt(lambda: push.gate_readme(REPO_ID), "raises")


@test("gate 6 catches a card pointing at the wrong repo", "never names")
def t_card_wrong_repo():
    with staged() as d:
        (d / "README.md").write_text(
            '```python\nfrom huggingface_hub import snapshot_download\n'
            'd = snapshot_download("someone-else/other-model")\n```\n')
        return attempt(lambda: push.gate_readme(REPO_ID), "never names")


@test("gate 6 runs against the STAGING DIR even if the card renames its variable")
def t_card_renamed_variable():
    """A card edit must not silently redirect the gate to the published bundle: the substitution
    is counted, and any surviving download call is refused (Fable, 2026-09-03)."""
    with staged() as d:
        (d / "README.md").write_text(
            '```python\nfrom huggingface_hub import snapshot_download\nimport sys\n'
            f'bundle = snapshot_download("{REPO_ID}")\nsys.path.insert(0, bundle)\n'
            'from zero_encoder import ZeroQueryEncoder\n'
            'enc = ZeroQueryEncoder(bundle, variant="int8")\n'
            'assert enc.encode(["hello"]).shape == (1, 1024)\n```\n')
        push.gate_readme(REPO_ID)               # must PASS, against the staged bytes


@test("gate 6 catches a card serving from a LITERAL repo id", "literal repo id")
def t_card_literal_repo_id():
    """The substitution rewrites `TextEmbedding(NAME)`; a hard-coded id would bypass it and be
    served from the PUBLISHED bytes (Fable, 2026-09-03)."""
    with staged() as d:
        (d / "README.md").write_text(
            '```python\nfrom fastembed import TextEmbedding\n'
            f'from huggingface_hub import snapshot_download\nd = snapshot_download("{REPO_ID}")\n'
            f'm = TextEmbedding("{REPO_ID}")\n```\n')
        return attempt(lambda: push.gate_readme(REPO_ID), "literal repo id")


@test("gate 6 catches a TextEmbedding call the substitution misses", "not redirected")
def t_card_unredirected_textembedding():
    """`TextEmbedding(model_name=NAME)` is not the exact text the regex rewrites, so without this
    check it would run offline against the CACHED PUBLISHED bytes."""
    with staged() as d:
        (d / "README.md").write_text(
            '```python\nfrom fastembed import TextEmbedding\n'
            f'from huggingface_hub import snapshot_download\nd = snapshot_download("{REPO_ID}")\n'
            f'NAME = "{REPO_ID}"\nm = TextEmbedding(model_name=NAME)\n```\n')
        return attempt(lambda: push.gate_readme(REPO_ID), "not redirected")


@test("gate 8 catches a corrupted ONNX graph",
      "do not reproduce the numpy query path")
def t_onnx_corrupt():
    """Gate 8 re-runs the parity arithmetic, so zeroing the table inside the graph must fail it --
    a recorded `"pass": true` would not have noticed."""
    import onnx
    from onnx import numpy_helper
    with staged() as d:
        m = onnx.load(str(d / "model.onnx"))
        for init in m.graph.initializer:
            if init.name == "TABLE":
                arr = numpy_helper.to_array(init)
                init.CopyFrom(numpy_helper.from_array(arr // 2, "TABLE"))
        onnx.save(m, str(d / "model.onnx"))
        return attempt(push.gate_onnx, "do not reproduce the numpy query path")


@test("push refuses without a build in the same invocation", "requires --build")
def t_no_build():
    built, push._BUILT = push._BUILT, False
    try:
        return attempt(lambda: push.push(REPO_ID, False), "requires --build")
    finally:
        push._BUILT = built


def main():
    print("building the reference bundle and rendering the card…")
    push.build()
    push.render_card(REPO_ID)
    print()
    bad = 0
    for name, expect, fn in TESTS:
        try:
            why = fn()
        except Exception as e:
            why = f"raised {type(e).__name__}: {e}"
        ok = why is None
        bad += not ok
        label = "PASSED " if expect is None else "REFUSED"
        print(f"{label if ok else 'FAILED '}   {name}" + ("" if ok else f"\n            <-- {why}"))
    print()
    print(f"{len(TESTS) - bad}/{len(TESTS)} checks held" if bad else
          f"ALL {len(TESTS)} CHECKS HOLD")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
