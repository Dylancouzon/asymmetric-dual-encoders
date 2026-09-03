"""Build and push the `zero` release bundle to the Hugging Face Hub.

  .venv/bin/python m11/release/push.py --build              # (re)build work/release/zero-v1
  .venv/bin/python m11/release/push.py --build --push       # build, verify, upload (PRIVATE)
  .venv/bin/python m11/release/push.py --push --public      # explicit opt-in for a public repo

Gates, all of which must pass before a single byte is uploaded:
  1. the table on disk hashes to m7/FREEZE.json's `table_sha256`
  2. every training-lineage run record hashes to the value FREEZE.json recorded
  3. freeze.assert_releasable(run_id) -- no non-commercial source anywhere in the lineage
  4. m11/release/verify_bundle.py -- the shipped numpy encoder reproduces the frozen torch path
The repo is created PRIVATE unless --public is passed explicitly.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "m11/release"
OUT = REPO / "work/release/zero-v1"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.txt",
                   "special_tokens_map.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def freeze():
    return json.loads((REPO / "m7/FREEZE.json").read_text())


def gate_artifact(fz):
    """(1) and (2): the bytes are the frozen bytes, and the lineage records are the frozen ones."""
    table = REPO / fz["table_relpath"]
    got = sha256(table)
    if got != fz["table_sha256"]:
        sys.exit(f"REFUSED: {table} hashes {got}, FREEZE.json says {fz['table_sha256']}")
    for rid, want in fz["training_lineage_record_sha256"].items():
        p = REPO / "work/runs" / f"{rid}.json"
        if not p.exists():
            sys.exit(f"REFUSED: lineage record {p} is missing; releasability is unprovable")
        if sha256(p) != want:
            sys.exit(f"REFUSED: lineage record {rid} changed since the freeze ({sha256(p)})")
    print(f"  gate 1-2 OK  table {got[:12]}…  lineage {list(fz['training_lineage_record_sha256'])}")
    return table


def gate_licence(fz):
    sys.path.insert(0, str(REPO / "m7src"))
    import freeze as freeze_mod
    freeze_mod.assert_releasable(fz["run_id"])
    print(f"  gate 3 OK    no non-commercial source in {fz['run_id']}'s lineage")


def build():
    fz = freeze()
    table = gate_artifact(fz)
    gate_licence(fz)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(table, OUT / "model.npz")
    shutil.copy2(SRC / "zero_encoder.py", OUT / "zero_encoder.py")

    snaps = Path.home() / ".cache/huggingface/hub/models--NovaSearch--stella_en_400M_v5/snapshots"
    tokdir = snaps / fz["teacher_revision"]
    if not tokdir.exists():
        sys.exit(f"REFUSED: teacher revision not cached at {tokdir}; the tokenizer must come "
                 "from the pinned revision, not from whatever is on the Hub today")
    for f in TOKENIZER_FILES:
        shutil.copy2(tokdir / f, OUT / f)

    meta = json.loads((table.parent / (table.stem + ".meta.json")).read_text())
    (OUT / "config.json").write_text(json.dumps({
        "model_type": "zero-lookup-table", "version": "v1", "run_id": fz["run_id"],
        "vocab": meta["vocab"], "dim": meta["dim"], "fallback_token_id": 101,
        "learned_weights": False, "weights_folded": True,
        "preproc": meta["preproc"], "preproc_fingerprint": meta["preproc_fingerprint"],
        "teacher": fz["teacher"], "teacher_revision": fz["teacher_revision"],
        "document_encoder": fz["encoder_spec"], "source_table_sha256": fz["table_sha256"],
        "recommended_variant": "int8", "similarity": "cosine",
    }, indent=1, sort_keys=True) + "\n")
    print(f"  built {OUT}")


def gate_conformance():
    r = subprocess.run([sys.executable, str(SRC / "verify_bundle.py"), str(OUT)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit("REFUSED: the shipped encoder does not reproduce the frozen query path")
    print("  gate 4 OK    shipped encoder matches the frozen path")


def push(repo_id, private):
    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()["name"]
    repo_id = repo_id or f"{who}/zero-query-encoder-v1"
    gate_artifact(freeze())
    gate_licence(freeze())
    gate_conformance()

    card = (SRC / "MODEL_CARD.md").read_text().replace("REPO_ID", repo_id)
    (OUT / "README.md").write_text(card)

    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(OUT), repo_type="model",
                      ignore_patterns=["__pycache__/*", "*.pyc"],
                      commit_message=f"zero v1 — M7 lookup table, run {freeze()['run_id']}")
    print(f"\n{'PRIVATE' if private else 'PUBLIC'} → https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--public", action="store_true", help="opt in to a PUBLIC repo")
    a = ap.parse_args()
    if not (a.build or a.push):
        ap.error("nothing to do: pass --build and/or --push")
    if a.build:
        build()
    if a.push:
        push(a.repo_id, private=not a.public)
