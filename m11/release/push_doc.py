"""Build and push the stella-400M ONNX DOCUMENT tower to a NEW public repo. M11a T3.

    .venv/bin/python m11/release/push_doc.py --build --gates          # uploads nothing
    .venv/bin/python m11/release/push_doc.py --build --push           # create, upload, verify, public

`--push` REQUIRES `--build`, for the same reason as `push.py`: the gates bind the bytes `build()`
just wrote, and a push over a stale staging dir would be ungated.

This is a NEW repo, so unlike `push.py` it does NOT pass `exist_ok=True`. `zero`'s repo already
existed and was already public, which spent the private-first ordering guarantee before it could be
used (`m11/STATUS.md`). Here the ordering is available and is used: create PRIVATE with
`exist_ok=False` -> upload -> verify the returned commit -> flip PUBLIC. A pre-existing destination
is refused rather than overwritten, and the repo id is a constant, not a free-form flag, so there
is no typo-to-publication path (Codex, 2026-09-03).

Gates, all against OUT/ after build and before a byte is uploaded:
  1. staged graphs re-verified by `export_doc.py --check` -- checker, opset, IO, no external data,
     and parity RE-DERIVED against the torch module on the frozen real-passage fixtures. Not a
     recorded `"pass": true`, which binds a file to a claim rather than to the arithmetic.
  2. manifest exactness -- OUT/ is every declared file and nothing else
  3. the tokenizer, as `fastembed.load_tokenizer` actually reads it: truncation 512, dynamic
     padding, a mixed [long, short] batch rectangular
  4. fastembed serves the staged graph through `PoolingType.DISABLED` and agrees with ORT
  5. every python block in the generated README executes, offline, against the staging dir

What these catch is ACCIDENTS -- a stale staging dir, the wrong tokenizer copied, a card that
raises, an export that silently lost its head. Not a malicious wrong-but-passing bundle: that
threat model was built once for T0 and rolled back on Dylan's instruction (`m11/STATUS.md`
§Scope note).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = Path(__file__).resolve().parent
OUT = REPO / "work/release/stella-doc-onnx"
ONNX_SRC = REPO / "work/m11onnx/stella-doc"

# Authorised by instructions-m11.md Amendment A ruling 2: a NEW PUBLIC repo on Dylan's account.
# A constant, not a flag -- see the module docstring.
REPO_ID = "DylanCouzon/stella-en-400M-v5-doc-onnx"

import export_doc as ed  # noqa: E402  (same directory; REVISION/CONFIG_KWARGS/MAX_LEN live there)

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.txt",
                   "special_tokens_map.json")
BASE_MANIFEST = ("config.json", "README.md", "model.onnx") + TOKENIZER_FILES

# Same edit as T1 on the `zero` repo, for the same reason and a sharper one. fastembed truncates
# at min(model_max_length, max_length); stella's own files say 32768/8000 while the document index
# was built at 512, so a fastembed caller would send documents of 513-8000 tokens through whole and
# silently fail to reproduce the index.
TOKENIZER_OVERRIDES = {"model_max_length": ed.MAX_LEN, "max_length": ed.MAX_LEN}

_BUILT = False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest():
    """fp32 only. An fp16 graph was built, measured and rejected: it passes CPU parity because ORT
    up-converts it, and disagrees with fp32 on 255 of 259 passages on CUDA, where it actually runs
    in fp16 (`results/m11_doc_fp16_gpu.json`). `export_doc.py --fp16` still builds one, into a
    directory this never reads."""
    return set(BASE_MANIFEST)


# ---------------------------------------------------------------- build

def sanitise_tokenizer(out):
    cfg_p = out / "tokenizer_config.json"
    cfg = json.loads(cfg_p.read_text())
    before = {k: cfg.get(k) for k in TOKENIZER_OVERRIDES}
    cfg.update(TOKENIZER_OVERRIDES)
    cfg_p.write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")

    tok_p = out / "tokenizer.json"
    tok = json.loads(tok_p.read_text())
    pad_before = tok.get("padding")
    tok["padding"] = None
    trunc = tok.get("truncation") or {}
    if trunc.get("max_length") != ed.MAX_LEN:
        sys.exit(f"REFUSED: tokenizer.json truncation is {trunc}, expected {ed.MAX_LEN}")
    tok_p.write_text(json.dumps(tok, indent=1, ensure_ascii=False) + "\n")
    print(f"  tokenizer sanitised  {before} -> {TOKENIZER_OVERRIDES};  padding "
          f"{pad_before and pad_before.get('strategy')} -> null")
    return {"tokenizer_config": {"before": before, "after": TOKENIZER_OVERRIDES},
            "tokenizer_json_padding": {"before": pad_before, "after": None}}


def build():
    global _BUILT
    snap = ed.snapshot()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    src = ONNX_SRC / "model.onnx"
    if not src.exists():
        sys.exit(f"REFUSED: {src} missing; run export_doc.py --export first")
    shutil.copy2(src, OUT / "model.onnx")
    if (ONNX_SRC / "model_fp16.onnx").exists():
        sys.exit(f"REFUSED: {ONNX_SRC / 'model_fp16.onnx'} exists. fp16 is not released "
                 "(results/m11_doc_fp16_gpu.json); move it out of the export dir so it cannot be "
                 "staged by accident.")

    for f in TOKENIZER_FILES:
        shutil.copy2(snap / f, OUT / f)
    tok_deviation = sanitise_tokenizer(OUT)

    # stella's own config.json, with the two xformers flags set to what the graph WAS EXPORTED
    # WITH. Shipping them as upstream has them (both true) would describe a model this repo does
    # not contain. fastembed reads only `pad_token_id` from this file.
    cfg = json.loads((snap / "config.json").read_text())
    for k, v in ed.CONFIG_KWARGS.items():
        if k not in cfg:
            sys.exit(f"REFUSED: stella config.json has no {k}; the export assumption is stale")
        cfg[k] = v
    cfg["export"] = {"format": "onnx", "opset": 17, "source_repo": ed.REPO_ID,
                     "source_revision": ed.REVISION, "head": "masked mean -> 2_Dense_1024 -> L2",
                     "max_length": ed.MAX_LEN, "training": "none -- format conversion only",
                     "tokenizer_deviation_from_source": tok_deviation}
    (OUT / "config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")

    _BUILT = True
    print(f"  built {OUT}  ({len(list(OUT.iterdir()))} files, "
          f"{sum(p.stat().st_size for p in OUT.iterdir())/1e9:.2f} GB)")


# ---------------------------------------------------------------- gates

def _run(script, *args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(SRC / script), *map(str, args)],
                          capture_output=True, text=True, env=env)


def gate_graphs():
    """(1) re-derive every graph check against the STAGED files, on the FULL fixture set.

    No `--n`. A shortened parity run still exits zero, so accepting a subset here would let a
    mistyped push certify the release against a fraction of the fixtures -- and the boundary and
    over-512 strata are the whole point of having them (Codex, 2026-09-03).
    """
    r = _run("export_doc.py", "--check", "--onnx-dir", OUT, "--no-write")
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        sys.exit("REFUSED: the staged graphs do not reproduce the torch document path")
    print("  gate 1 OK    staged graphs re-checked against the torch module")


def gate_manifest():
    want = manifest()
    have = {p.name for p in OUT.iterdir() if p.name != "__pycache__"}
    if have != want:
        sys.exit(f"REFUSED: staging dir is not the manifest.  extra={sorted(have - want)}  "
                 f"missing={sorted(want - have)}")
    print(f"  gate 2 OK    staging dir is exactly the {len(want)}-file manifest")
    return want


def gate_tokenizer():
    """(3) what fastembed's own loader makes of the shipped tokenizer files."""
    from fastembed.common.preprocessor_utils import load_tokenizer
    tk, _ = load_tokenizer(OUT)
    trunc = tk.truncation
    pad = tk.padding
    if not trunc or trunc["max_length"] != ed.MAX_LEN:
        sys.exit(f"REFUSED: fastembed's loader gets truncation {trunc}, not {ed.MAX_LEN}; "
                 "documents of 513-8000 tokens would not reproduce the index")
    if pad is None or pad.get("length") is not None:
        sys.exit(f"REFUSED: fastembed's loader gets padding {pad}; a fixed length makes a mixed "
                 "batch ragged or wastes compute")
    long_text = "retrieval " * 800
    enc = tk.encode_batch([long_text, "a short document"])
    shapes = [len(e.ids) for e in enc]
    if len(set(shapes)) != 1 or shapes[0] != ed.MAX_LEN:
        sys.exit(f"REFUSED: a mixed [long, short] batch encodes to {shapes}, not "
                 f"[{ed.MAX_LEN}, {ed.MAX_LEN}]")
    print(f"  gate 3 OK    fastembed loader: truncation {ed.MAX_LEN}, dynamic padding, "
          f"mixed batch {shapes}")


def gate_fastembed():
    """(4) fastembed actually SERVES the staged graph, and agrees with ORT on real text.

    `PoolingType.DISABLED` + `normalization=False` passes an already-pooled graph through
    untouched, so the pooled `model.onnx` is the fastembed artifact and no second per-token graph
    is needed. Measured here rather than asserted -- the M9 note claiming the opposite is what
    would have cost a third 1.75 GB file.
    """
    import numpy as np
    import onnxruntime as ort
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    name = "qdrant-research/stella-doc-onnx-gate"
    texts, _ = ed.fixtures(12)
    TextEmbedding.add_custom_model(
        model=name, pooling=PoolingType.DISABLED, normalization=False,
        sources=ModelSource(hf=REPO_ID), dim=1024, model_file="model.onnx",
        description="M11a T3 gate", license="mit", size_in_gb=1.8)
    emb = TextEmbedding(model_name=name, specific_model_path=str(OUT))
    got = np.asarray(list(emb.embed(texts, batch_size=4)))
    del emb

    sess = ort.InferenceSession(str(OUT / "model.onnx"), providers=["CPUExecutionProvider"])
    ref = np.concatenate([sess.run(None, {"input_ids": i, "attention_mask": m})[0]
                          for i, m in ed.encode(texts, batch=4)])
    del sess
    cos = ((ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))).min()
    mx = float(np.abs(ref - got).max())
    if not (cos >= 1 - 1e-5 and mx <= 1e-4):
        sys.exit(f"REFUSED: fastembed disagrees with ORT -- min-cos {cos:.8f}, max-abs {mx:.2e}")
    print(f"  gate 4 OK    fastembed serves model.onnx (DISABLED pooling): min-cos {cos:.8f}, "
          f"max-abs {mx:.2e}")


# The cards document FastEmbed usage by BUILT-IN model name, which only resolves on the branch
# carrying the registration (Dylan, 2026-09-03: "the card should assume the model is in Fastembed
# ... point the card to our branch for now"). The card gate therefore runs against that checkout.
FASTEMBED_FORK = Path("/home/dylan/fastembed")

README_SUBSTITUTIONS = [
    (re.compile(r'^(\w+) = snapshot_download\(.*$', re.M), r'\1 = BUNDLE_DIR'),
    (re.compile(r'^from huggingface_hub import snapshot_download$', re.M), ''),
]

# Applied separately, and NOT counted: a card may legitimately have no FastEmbed block (every
# negative fixture in test_gates.py is such a card). TextEmbedding(NAME) would otherwise download
# the PUBLISHED repo. DOC_NAME on the zero card is deliberately left alone -- it is the sibling
# model, not the bytes under test.
FASTEMBED_SUBSTITUTION = (re.compile(r'TextEmbedding\(NAME\)'),
                          'TextEmbedding(NAME, specific_model_path=BUNDLE_DIR)')


def gate_readme():
    """(5) the card's python blocks run, offline, against the STAGED bytes."""
    card = (OUT / "README.md").read_text()
    # The card names the model rather than downloading it by hand, so the id is checked
    # textually: everything the gate then runs is pointed at the staging dir below.
    if REPO_ID not in card:
        sys.exit(f"REFUSED: the card never names {REPO_ID!r}; a wrong repo id would ship unnoticed")
    blocks = re.findall(r"```python\n(.*?)```", card, re.S)
    if not blocks:
        sys.exit("REFUSED: the generated README.md has no python blocks to execute")
    script = "\n".join(blocks)
    for pat, rep in README_SUBSTITUTIONS:
        script, _ = pat.subn(rep, script)
    script = FASTEMBED_SUBSTITUTION[0].sub(FASTEMBED_SUBSTITUTION[1], script)
    if REPO_ID and re.search(rf'TextEmbedding\(\s*["\']{re.escape(REPO_ID)}', script):
        sys.exit("REFUSED: the card builds TextEmbedding from a literal repo id; the gate rewrites "
                 "TextEmbedding(NAME) only, so this would be served from the PUBLISHED bytes")
    # FASTEMBED_SUBSTITUTION rewrites the exact text `TextEmbedding(NAME)`. A card edited to
    # `TextEmbedding(model_name=NAME)` or `TextEmbedding(NAME, threads=1)` would slip past it and,
    # under HF_HUB_OFFLINE, be served from the CACHED PUBLISHED bytes -- the redirection this
    # machinery exists to prevent (Fable, 2026-09-03). So refuse any call that was not redirected.
    # DOC_NAME is the sibling model, not the bytes under test, and is allowed through by name.
    stray = [m.group(0) for m in re.finditer(r"TextEmbedding\([^)]*\)", script)
             if "specific_model_path=BUNDLE_DIR" not in m.group(0)
             and not m.group(0).startswith("TextEmbedding(DOC_NAME")]
    if stray:
        sys.exit(f"REFUSED: {stray[0]!r} is not redirected at the staging dir; the gate rewrites "
                 "the exact text `TextEmbedding(NAME)` only, so this would run against the "
                 "PUBLISHED bytes")
    if "snapshot_download" in script or "hf_hub_download" in script:
        sys.exit("REFUSED: the card still reaches the Hub after substitution")
    script = f"BUNDLE_DIR = {str(OUT)!r}\n" + script
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    r = subprocess.run([sys.executable, path], capture_output=True, text=True, cwd=str(REPO),
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                                PYTHONPATH=str(FASTEMBED_FORK)))
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:])
        sys.stderr.write(r.stderr[-4000:])
        sys.exit(f"REFUSED: a python block in the card raises.  script kept at {path}")
    Path(path).unlink()
    print(f"  gate 5 OK    the card's {len(blocks)} python blocks run against the staged bytes")


def render_card():
    """Substitute the repo id and the MEASURED parity numbers into the card.

    The numbers come from `results/m11_doc_export.json`, so the card cannot quote a figure nobody
    produced, and a missing placeholder is refused rather than shipped as literal `PARITY_...`.
    """
    card = (SRC / "MODEL_CARD_DOC.md").read_text().replace("REPO_ID", REPO_ID)
    res = json.loads(ed.RESULT.read_text())

    if "parity_model_fp16.onnx" in res:
        sys.exit(f"REFUSED: {ed.RESULT} still records an fp16 graph; re-run "
                 "`export_doc.py --check` with fp16 out of the export dir")
    subs = {}
    p32 = res["parity_model.onnx"]
    subs["PARITY_FP32_COS"] = f"{p32['min_cos']:.8f}"
    subs["PARITY_FP32_ABS"] = f"{p32['max_abs']:.2e}"
    subs["PARITY_FP32_NORMS"] = f"{p32['out_norms'][0]:.6f}–{p32['out_norms'][1]:.6f}"
    card = card.replace("259 real", f"{res['n_fixtures']} real")
    for k, v in subs.items():
        card = card.replace(k, v)
    left = re.findall(r"\b(PARITY_[A-Z0-9_]+|BATCH_INV_FP16)\b", card)
    if left:
        sys.exit(f"REFUSED: the card still has unsubstituted placeholders {sorted(set(left))}; "
                 f"{ed.RESULT} has no measurement for them")
    (OUT / "README.md").write_text(card)


def run_gates():
    """-> {filename: sha256} captured AFTER the last gate.

    The snapshot is what the upload is verified against. Comparing the remote commit with the
    staging dir as it exists at verification time (which push.py does) compares changed, ungated
    bytes against themselves (Codex, 2026-09-03).
    """
    render_card()
    gate_graphs()
    want = gate_manifest()
    gate_tokenizer()
    gate_fastembed()
    gate_readme()
    return {name: sha256(OUT / name) for name in sorted(want)}


# ---------------------------------------------------------------- push

def verify_remote(api, snapshot, revision):
    """Compare every uploaded file against the build snapshot at the commit just written.

    The graphs go through LFS, whose object id IS the sha256 of the content, so they are verified
    from `files_metadata` rather than by pulling 2.6 GB back down. The small files are compared by
    re-download, which is cheap and covers the non-LFS path.
    """
    from huggingface_hub import snapshot_download
    want = set(snapshot)
    info = api.repo_info(REPO_ID, revision=revision, files_metadata=True)
    remote = {s.rfilename: s for s in info.siblings if s.rfilename != ".gitattributes"}
    if set(remote) != want:
        sys.exit(f"REFUSED: {revision} file set differs.  extra={sorted(set(remote) - want)}  "
                 f"missing={sorted(want - set(remote))}")

    small = []
    for name, s in sorted(remote.items()):
        oid = getattr(getattr(s, "lfs", None), "sha256", None)
        if oid:
            local = snapshot[name]
            if oid != local:
                sys.exit(f"REFUSED: remote LFS {name} sha256 {oid}, staged {local}")
            print(f"    lfs  {name}  {oid[:12]}…  matches")
        else:
            small.append(name)

    with tempfile.TemporaryDirectory() as td:
        got = snapshot_download(REPO_ID, revision=revision, local_dir=td, force_download=True,
                                allow_patterns=small)
        for name in small:
            a, b = snapshot[name], sha256(Path(got) / name)
            if a != b:
                sys.exit(f"REFUSED: remote {name} hashes {b}, staged {a}")
    print(f"  verified  {len(want)} files at {revision[:10]} match the staging dir "
          f"({len(want) - len(small)} by LFS oid, {len(small)} by re-download)")


def push(update=False):
    if not _BUILT:
        sys.exit("REFUSED: --push requires --build in the same invocation")
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError
    api = HfApi()
    who = api.whoami()["name"]
    if not REPO_ID.startswith(who + "/"):
        sys.exit(f"REFUSED: authenticated as {who}, but REPO_ID is {REPO_ID}")

    snapshot = run_gates()

    # The repo is created once, private, and flipped public after byte verification. A second
    # push (a card fix, say) has to update it in place -- refusing outright meant the rewritten
    # card could never ship through the gated path (Fable, 2026-09-03). Updating requires the
    # explicit --update flag, so it is a decision and not a default.
    try:
        api.repo_info(REPO_ID)
        exists = True
    except RepositoryNotFoundError:
        exists = False
    if exists and not update:
        sys.exit(f"REFUSED: {REPO_ID} already exists; pass --update to publish over it.")
    if not exists:
        api.create_repo(REPO_ID, repo_type="model", private=True, exist_ok=False)
        print(f"  created {REPO_ID} PRIVATE")
    else:
        print(f"  updating existing {REPO_ID}")
    # delete_patterns so an --update commit is exactly the manifest. Without it a renamed file
    # stays live and public, and verify_remote refuses AFTER the commit exists (Fable, 2026-09-03).
    info = api.upload_folder(repo_id=REPO_ID, folder_path=str(OUT), repo_type="model",
                             allow_patterns=sorted(snapshot), delete_patterns=["*"],
                             commit_message=f"stella_en_400M_v5 document path -> ONNX opset 17 "
                                            f"(source revision {ed.REVISION[:12]})")
    print(f"  uploaded commit {info.oid[:10]}")
    verify_remote(api, snapshot, info.oid)

    if not exists:
        api.update_repo_settings(repo_id=REPO_ID, repo_type="model", private=False)
    if api.repo_info(REPO_ID).private:
        sys.exit("REFUSED: asked for PUBLIC but the repo still reports private")
    print(f"\nPUBLIC → https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--update", action="store_true",
                    help="publish over the existing repo instead of creating it")
    a = ap.parse_args()
    if not (a.build or a.gates or a.push):
        ap.error("nothing to do: pass --build, --gates and/or --push")
    if a.push and not a.build:
        ap.error("--push requires --build in the same invocation")
    if a.build:
        build()
    if a.gates and not a.push:
        run_gates()
    if a.push:
        push(update=a.update)
