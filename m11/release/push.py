"""Build and push the `constella-zero` release bundle to the Hugging Face Hub.

  .venv/bin/python m11/release/push.py --build --gates            # build + gate, upload nothing
  .venv/bin/python m11/release/push.py --build --push             # build, gate, upload, verify

The repo is ALREADY PUBLIC, so `--push` replaces public bytes and `--public` is a no-op for it;
the flag survives only for a repo that has to be created private first (Fable, 2026-09-03).

`--push` REQUIRES `--build` in the same invocation: the gates below bind the bytes that `build()`
just wrote, and a push over a stale or hand-edited staging directory would be ungated (M11a T0).

Gates, all of which run against `OUT/` after it is built and before a byte is uploaded:
  1. the frozen source table AND `OUT/model.npz` both hash to m7/FREEZE.json's `table_sha256`
  2. every training-lineage run record hashes to the value FREEZE.json recorded
  3. freeze.assert_releasable(run_id) -- no non-commercial source anywhere in the lineage
  4. verify_bundle.py -- the shipped numpy encoder reproduces the frozen torch path, with the
     reference table loaded from the FROZEN SOURCE PATH so the two sides cannot be the same file
  5. manifest exactness -- OUT/ contains every declared file and nothing else
  6. every python block in the generated README.md executes
  7. verify_tokenizer.py -- fastembed's own loader gets the frozen 512 rule and dynamic padding
  8. export_onnx.py --check -- re-runs every ONNX parity check against the STAGED graphs. It
     re-derives the numbers rather than reading a `"pass": true` JSON, because a recorded verdict
     binds the file to a claim and not to the arithmetic.

The gates exist to catch ACCIDENTS -- a stale staging directory, a hand-edit nobody remembers, the
wrong tokenizer copied, a card that raises. Two adversarial reviews (Codex and Fable, 2026-09-03)
also named malicious wrong-but-passing bundles; that is not the threat model here (one researcher,
one box, a research artifact), so those findings are actioned only where the fix also catches a
plausible mistake. What they caught that WAS real: gates 4 and 8 imported the source
`zero_encoder.py` rather than the staged copy, gate 4's reference table was the bundle's own, and
the sidecar `.meta.json` -- not FREEZE.json -- set the shipped preproc rule.

Then: upload, re-download at the commit just written, and compare every file against the staging
dir. The repo is already PUBLIC (see `m11/STATUS.md` -- it was never private, despite the docs), so
there is no visibility to flip and no ordering left to protect; the re-download is how we know the
bytes landed. `create_repo(exist_ok=True)` silently ignores `private`, so visibility is read back
and reported rather than assumed.
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
SRC = REPO / "m11/release"
OUT = REPO / "work/release/zero-v1"

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.txt",
                   "special_tokens_map.json")
# Exactly what ships. push() refuses to upload anything not on this list, and refuses to
# leave anything on it behind.
ONNX_FILES = ("model.onnx", "model_tokens.onnx")
ONNX_SRC = REPO / "work/m11onnx/zero-v1"
MANIFEST = (("model.npz", "config.json", "zero_encoder.py", "README.md")
            + TOKENIZER_FILES + ONNX_FILES)

# The frozen preprocessing rule is max_length 512 (FREEZE.json:preproc) and the document index
# was built at 512 (FREEZE.json:encoder_spec). stella's own tokenizer files declare truncation
# 8000 / model_max_length 32768 and fixed-512 padding, which fastembed reads and our numpy path
# does not -- so a fastembed caller would neither truncate at 512 nor get a dynamic batch.
# Ruled 2026-09-03 (Dylan): edit the files, state the deviation in the card. M11a T1.
TOKENIZER_OVERRIDES = {"model_max_length": 512, "max_length": 512}

_BUILT = False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def freeze():
    return json.loads((REPO / "m7/FREEZE.json").read_text())


# ---------------------------------------------------------------- build

def sanitise_tokenizer(out):
    """T1: ship the frozen 512 rule and dynamic padding, for the readers that honour these fields.

    The numpy path is unaffected -- zero_encoder.py calls no_padding() and takes truncation from
    config.json:preproc.max_length -- so its output is byte-identical before and after. Only
    fastembed (and any transformers caller) reads what changes here.
    """
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
    if trunc.get("max_length") != 512:
        sys.exit(f"REFUSED: tokenizer.json truncation is {trunc}, expected max_length 512")
    tok_p.write_text(json.dumps(tok, indent=1, ensure_ascii=False) + "\n")
    print(f"  T1 tokenizer sanitised  {before} -> {TOKENIZER_OVERRIDES};  "
          f"padding {pad_before and pad_before['strategy']} -> null")
    return {"tokenizer_config": {"before": before, "after": TOKENIZER_OVERRIDES},
            "tokenizer_json_padding": {"before": pad_before, "after": None}}


def build():
    global _BUILT
    fz = freeze()
    table = REPO / fz["table_relpath"]
    if sha256(table) != fz["table_sha256"]:
        sys.exit(f"REFUSED: source {table} does not hash to FREEZE.json's table_sha256")
    meta_p = table.parent / (table.stem + ".meta.json")
    if sha256(meta_p) != fz["table_meta_sha256"]:
        sys.exit(f"REFUSED: {meta_p} hashes {sha256(meta_p)}, FREEZE.json's table_meta_sha256 is "
                 f"{fz['table_meta_sha256']}; the sidecar sets the shipped preproc rule")

    if OUT.exists():
        shutil.rmtree(OUT)                      # a fresh staging dir every time, no residue
    OUT.mkdir(parents=True)

    shutil.copy2(table, OUT / "model.npz")
    shutil.copy2(SRC / "zero_encoder.py", OUT / "zero_encoder.py")

    snaps = Path.home() / ".cache/huggingface/hub/models--NovaSearch--stella_en_400M_v5/snapshots"
    tokdir = snaps / fz["teacher_revision"]
    if not tokdir.exists():
        sys.exit(f"REFUSED: teacher revision not cached at {tokdir}; the tokenizer must come "
                 "from the pinned revision, not from whatever is on the Hub today")
    for f in TOKENIZER_FILES:
        shutil.copy2(tokdir / f, OUT / f)
    tok_deviation = sanitise_tokenizer(OUT)

    # Every frozen field comes from FREEZE.json; the sidecar is cross-checked against it, never
    # trusted as the source. (Its bytes are pinned above, so disagreement is a real inconsistency.)
    meta = json.loads(meta_p.read_text())
    for k in ("preproc", "preproc_fingerprint"):
        if meta[k] != fz[k]:
            sys.exit(f"REFUSED: sidecar {k}={meta[k]!r} disagrees with FREEZE.json {fz[k]!r}")
    if (meta["vocab"], meta["dim"]) != (fz["encoder_spec"]["vocab"], fz["encoder_spec"]["dim"]):
        sys.exit(f"REFUSED: sidecar vocab/dim {meta['vocab']}/{meta['dim']} disagrees with "
                 f"FREEZE.json encoder_spec")
    (OUT / "config.json").write_text(json.dumps({
        "model_type": "zero-lookup-table", "version": "v1", "run_id": fz["run_id"],
        "vocab": fz["encoder_spec"]["vocab"], "dim": fz["encoder_spec"]["dim"],
        "fallback_token_id": fz["encoder_spec"]["cls_id"],
        "learned_weights": False, "weights_folded": True,
        "preproc": fz["preproc"], "preproc_fingerprint": fz["preproc_fingerprint"],
        "teacher": fz["teacher"], "teacher_revision": fz["teacher_revision"],
        "document_encoder": fz["encoder_spec"], "source_table_sha256": fz["table_sha256"],
        "recommended_variant": "int8", "similarity": "cosine",
        "tokenizer_deviation_from_teacher": tok_deviation,
    }, indent=1, sort_keys=True) + "\n")

    for name in ONNX_FILES:
        src = ONNX_SRC / name
        if not src.exists():
            sys.exit(f"REFUSED: {src} does not exist; run m11/release/export_onnx.py first")
        shutil.copy2(src, OUT / name)
    _BUILT = True
    print(f"  built {OUT}  ({len(list(OUT.iterdir()))} files)")






# ---------------------------------------------------------------- gates

def gate_artifact(fz):
    """(1) and (2): the staged bytes are the frozen bytes; the lineage records are the frozen ones."""
    if sha256(REPO / fz["table_relpath"]) != fz["table_sha256"]:
        sys.exit(f"REFUSED: source table {REPO / fz['table_relpath']} is not the frozen bytes")
    if sha256(OUT / "model.npz") != fz["table_sha256"]:
        sys.exit(f"REFUSED: staged model.npz hashes {sha256(OUT / 'model.npz')}, "
                 f"FREEZE.json says {fz['table_sha256']}; the staging dir is stale")
    for rid, want in fz["training_lineage_record_sha256"].items():
        p = REPO / "work/runs" / f"{rid}.json"
        if not p.exists():
            sys.exit(f"REFUSED: lineage record {p} is missing; releasability is unprovable")
        if sha256(p) != want:
            sys.exit(f"REFUSED: lineage record {rid} changed since the freeze ({sha256(p)})")
    print(f"  gate 1-2 OK  source AND staged table {fz['table_sha256'][:12]}…  "
          f"lineage {list(fz['training_lineage_record_sha256'])}")


def gate_licence(fz):
    sys.path.insert(0, str(REPO / "m7src"))
    import freeze as freeze_mod
    freeze_mod.assert_releasable(fz["run_id"])
    print(f"  gate 3 OK    no non-commercial source in {fz['run_id']}'s lineage")


def _run(script, *args):
    """Run a gate script. PYTHONDONTWRITEBYTECODE keeps __pycache__ out of the staging dir, so
    the digest snapshot stays true across the gates that import from it."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(SRC / script), *map(str, args)],
                          capture_output=True, text=True, env=env)


def gate_conformance(fz):
    """(4) the STAGED encoder vs the frozen torch path, reference loaded from the FROZEN SOURCE.

    Both `--staged-encoder` and `--ref-table` are load-bearing: without the first the gate tests
    `m11/release/zero_encoder.py` and not the file that ships; without the second both sides read
    the bundle's own table."""
    r = _run("verify_bundle.py", OUT, "--ref-table", REPO / fz["table_relpath"],
             "--staged-encoder")
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit("REFUSED: the shipped encoder does not reproduce the frozen query path")
    print("  gate 4 OK    shipped encoder matches the frozen path (ref = frozen source table)")


def gate_manifest():
    """(5) the staging dir is exactly the manifest -- no extra file, none missing."""
    want = set(MANIFEST)
    have = {p.name for p in OUT.iterdir() if p.name != "__pycache__"}
    if have != want:
        sys.exit(f"REFUSED: staging dir is not the manifest.  extra={sorted(have - want)}  "
                 f"missing={sorted(want - have)}")
    print(f"  gate 5 OK    staging dir is exactly the {len(want)}-file manifest")
    return want


def gate_tokenizer():
    """(8) T1: what fastembed's loader makes of the shipped tokenizer files."""
    r = _run("verify_tokenizer.py", OUT, "--staged-encoder")
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit("REFUSED: the shipped tokenizer does not give a fastembed caller the frozen rule")
    print("  gate 7 OK    fastembed's loader gets truncation 512 and dynamic padding")


# The cards document FastEmbed usage by BUILT-IN model name, which only resolves on the branch
# carrying the registration (Dylan, 2026-09-03: "the card should assume the model is in Fastembed
# ... point the card to our branch for now"). The card gate therefore runs against that checkout.
FASTEMBED_FORK = Path("/home/dylan/fastembed")
DEFAULT_REPO_ID = "DylanCouzon/constella-zero"

README_SUBSTITUTIONS = [
    # The card tells a user to snapshot_download the repo; the gate runs the same code against
    # the staging dir, because the bytes being gated are not on the Hub yet. This is the only
    # edit made to the card's code before executing it.
    #
    # Fable, 2026-09-03: the old pattern was anchored to `^d = snapshot_download(...)`, so
    # renaming the variable in the card made it match nothing -- the gate then DOWNLOADED THE LIVE
    # REPO and certified the card against bytes that are not the ones shipping. Hence the
    # download-argument check, the residual-name check, and HF_HUB_OFFLINE below.
    (re.compile(r'^(\w+) = snapshot_download\(.*$', re.M), r'\1 = BUNDLE_DIR'),
    (re.compile(r'^from huggingface_hub import snapshot_download$', re.M), ''),
]

# Applied separately, and NOT counted: a card may legitimately have no FastEmbed block (every
# negative fixture in test_gates.py is such a card). TextEmbedding(NAME) would otherwise download
# the PUBLISHED repo. DOC_NAME on the zero card is deliberately left alone -- it is the sibling
# model, not the bytes under test.
FASTEMBED_SUBSTITUTION = (re.compile(r'TextEmbedding\(NAME\)'),
                          'TextEmbedding(NAME, specific_model_path=BUNDLE_DIR)')


def gate_readme(repo_id=None):
    """(6) execute the card's python blocks AS ONE SEQUENTIAL TUTORIAL, in one namespace.

    That is what it tests -- the blocks deliberately share `q`, `D`, `docs`, so this does not show
    that any block runs standalone. `snapshot_download(...)` is rewritten to the staging dir
    because the bytes being gated are not on the Hub yet, so the gate also does not test the
    download itself; `repo_id` is checked textually instead. Nothing runs the card against the
    PUBLISHED bytes -- a `--post-push-smoke` flag was described here but never existed
    (Fable, 2026-09-03)."""
    card = (OUT / "README.md").read_text()
    # The card names the model rather than downloading it by hand, so the id is checked
    # textually: everything the gate then runs is pointed at the staging dir below.
    if repo_id and repo_id not in card:
        sys.exit(f"REFUSED: the card never names {repo_id!r}; a wrong repo id would ship unnoticed")
    # Every Hub download in the card must name THIS repo. The substitution below rewrites the
    # line regardless of its argument, so a typo'd id would vanish before execution and the
    # "names it somewhere" check above would still pass (Codex, 2026-09-03).
    for arg in re.findall(r"snapshot_download\(\s*[\"']([^\"']+)[\"']", card):
        if repo_id and arg != repo_id:
            sys.exit(f"REFUSED: the card downloads {arg!r}, not {repo_id!r}; the gate rewrites "
                     "that line, so the wrong id would ship unnoticed")
    blocks = re.findall(r"```python\n(.*?)```", card, re.S)
    if not blocks:
        sys.exit("REFUSED: the generated README.md has no python blocks to execute")
    script = "\n".join(blocks)
    for pat, rep in README_SUBSTITUTIONS:
        script, _ = pat.subn(rep, script)
    script = FASTEMBED_SUBSTITUTION[0].sub(FASTEMBED_SUBSTITUTION[1], script)
    if repo_id and re.search(rf'TextEmbedding\(\s*["\']{re.escape(repo_id)}', script):
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
        sys.exit("REFUSED: the card still downloads from the Hub after substitution; the gate "
                 "must exercise the STAGED bytes, not whatever is published")
    script = f"BUNDLE_DIR = {str(OUT)!r}\n" + script
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        path = f.name
    # HF_HUB_OFFLINE closes the escape route: if any block still reaches the Hub, it fails loudly
    # instead of silently validating published bytes. The card's stella pin is cached, so the
    # document-side block runs offline.
    r = subprocess.run([sys.executable, path], capture_output=True, text=True, cwd=str(REPO),
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                                PYTHONPATH=str(FASTEMBED_FORK)))
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:])
        sys.stderr.write(r.stderr[-4000:])
        sys.exit(f"REFUSED: a python block in the card raises.  script kept at {path}")
    Path(path).unlink()
    print(f"  gate 6 OK    the card's {len(blocks)} python blocks run as one sequential tutorial")


def render_card(repo_id):
    card = (SRC / "MODEL_CARD.md").read_text().replace("REPO_ID", repo_id)
    (OUT / "README.md").write_text(card)


def gate_onnx():
    """(8) re-run the ONNX parity checks against the STAGED graphs."""
    r = _run("export_onnx.py", "--check", "--no-write", "--onnx-dir", OUT, "--bundle", OUT)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit("REFUSED: the staged ONNX graphs do not reproduce the numpy query path")
    print("  gate 8 OK    staged ONNX graphs re-checked against the numpy path")


def run_gates(repo_id=None):
    """-> {filename: sha256} captured AFTER the last gate.

    That snapshot is what the upload is verified against. Re-hashing OUT at verification time
    would compare changed, ungated bytes with themselves -- the defect already fixed in
    push_doc.py, which this had not inherited (Codex, 2026-09-03).
    """
    fz = freeze()
    gate_artifact(fz)
    gate_licence(fz)
    gate_conformance(fz)
    gate_manifest()
    gate_readme(repo_id)
    gate_tokenizer()
    gate_onnx()
    return {name: sha256(OUT / name) for name in sorted(MANIFEST)}


# ---------------------------------------------------------------- push

def verify_remote(api, repo_id, snapshot, revision):
    """Re-download `revision` and compare every byte against the post-gate hash SNAPSHOT.

    Pinned to the revision we just wrote, never to the mutable branch head: the head could be a
    different commit than the one uploaded, and could move again afterwards.
    """
    from huggingface_hub import snapshot_download
    want = set(snapshot)
    remote = {s.rfilename for s in api.repo_info(repo_id, revision=revision).siblings}
    remote -= {".gitattributes"}
    if remote != want:
        sys.exit(f"REFUSED: {revision} file set differs.  extra={sorted(remote - want)}  "
                 f"missing={sorted(want - remote)}")
    with tempfile.TemporaryDirectory() as td:
        got = snapshot_download(repo_id, revision=revision, local_dir=td, force_download=True,
                                allow_patterns=sorted(want))
        for name in sorted(want):
            a, b = snapshot[name], sha256(Path(got) / name)
            if a != b:
                sys.exit(f"REFUSED: remote {name} hashes {b}, staged {a}")
    print(f"  verified  {len(want)} files at {revision[:10]} match the gated snapshot")


def gates_only(repo_id):
    """--gates: render the card and run every gate, uploading nothing."""
    render_card(repo_id)
    run_gates(repo_id=repo_id)


def push(repo_id, public):
    if not _BUILT:
        sys.exit("REFUSED: --push requires --build in the same invocation; the gates bind the "
                 "bytes build() writes, and a push over a stale staging dir would be ungated")
    from huggingface_hub import HfApi
    api = HfApi()
    repo_id = repo_id or DEFAULT_REPO_ID

    render_card(repo_id)            # BEFORE the gates: gate 6 executes what actually ships
    snapshot = run_gates(repo_id=repo_id)
    want = set(snapshot)

    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    # create_repo(exist_ok=True) ignores `private` on an existing repo, so say it outright. If the
    # repo is already PUBLIC this cannot un-publish what was already served -- report, don't assume.
    was_public = not api.repo_info(repo_id).private
    if was_public:
        print("  NOTE the repo is ALREADY PUBLIC; this upload replaces public bytes, and every "
              "earlier revision stays publicly reachable")
    else:
        api.update_repo_settings(repo_id=repo_id, repo_type="model", private=True)

    # delete_patterns so the commit is exactly the manifest, not the manifest plus whatever an
    # earlier commit left in the repo.
    info = api.upload_folder(repo_id=repo_id, folder_path=str(OUT), repo_type="model",
                             allow_patterns=sorted(want), delete_patterns=["*"],
                             commit_message=f"constella-zero — FastEmbed-first card, run {freeze()['run_id']}")
    print(f"  uploaded commit {info.oid[:10]} → https://huggingface.co/{repo_id}")
    verify_remote(api, repo_id, snapshot, info.oid)

    if public and not was_public:
        api.update_repo_settings(repo_id=repo_id, repo_type="model", private=False)
        if api.repo_info(repo_id).private:
            sys.exit("REFUSED: asked for PUBLIC but the repo still reports private")
    final = "PUBLIC" if (public or was_public) else "PRIVATE"
    print(f"\n{final} → https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--gates", action="store_true", help="run the gates without uploading")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--public", action="store_true", help="flip to PUBLIC after remote verification")
    a = ap.parse_args()
    if not (a.build or a.push or a.gates):
        ap.error("nothing to do: pass --build, --gates and/or --push")
    if a.push and not a.build:
        ap.error("--push requires --build in the same invocation (see the module docstring)")
    if a.build:
        build()
    if a.gates and not a.push:
        # The card now names a FastEmbed model, so the gate needs the real id: a placeholder
        # resolves to no registered model and gate 6 fails on a card that is in fact correct.
        rid = a.repo_id or DEFAULT_REPO_ID
        gates_only(rid)
    if a.push:
        push(a.repo_id, public=a.public)
