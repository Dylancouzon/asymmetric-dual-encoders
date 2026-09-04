"""Verify what the two published repos actually SERVE, against local truth.

Every other gate here checks bytes on the way OUT. This one checks what the Hub gives back to a
stranger: that both repos are public, that the shipped table still hashes to `m7/FREEZE.json`,
that the document graph's LFS sha256 is the byte string the gates signed off, that neither repo
carries a file its manifest does not name, and that both cards say what we believe they say.

    .venv/bin/python m11/release/verify_published.py
"""
import json, os, pathlib, sys, hashlib
os.environ.pop("HF_TOKEN", None)
# popping HF_TOKEN is not enough: hf_hub_download and HfApi still pick up the token
# cached in ~/.cache/huggingface (Fable, 2026-09-03).
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
import numpy as np, requests

ZERO, DOC = "DylanCouzon/constella-zero", "DylanCouzon/stella-en-400M-v5-doc-onnx"
ok = True
def chk(n, p, d=""):
    global ok; ok &= bool(p); print(f"{'PASS' if p else 'FAIL'}  {n:52s}  {d}", flush=True)

# The commits M11 closed on. Pinned, not printed: a verifier that prints whatever the Hub
# currently serves cannot tell you the Hub still serves what you shipped (Codex, 2026-09-03).
EXPECTED = {ZERO: "d9d575a4a9", DOC: "e0430a63b6"}

for rid in (ZERO, DOC):
    r = requests.get(f"https://huggingface.co/api/models/{rid}", timeout=30).json()
    chk(f"{rid.split('/')[1]}: public, anonymous", r.get("private") is False,
        f"{len(r.get('siblings', []))} files")
    chk(f"{rid.split('/')[1]}: head is the commit M11 closed on",
        r.get("sha", "").startswith(EXPECTED[rid]),
        f"{r.get('sha','')[:10]} (expected {EXPECTED[rid]})")

# the published zero table still hashes to FREEZE.json
from huggingface_hub import hf_hub_download
fz = json.load(open("m7/FREEZE.json"))
p = hf_hub_download(ZERO, "model.npz")
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
chk("published model.npz == FREEZE.json table_sha256", h == fz["table_sha256"], h[:16])

# the published doc graph is the byte string the gates signed off
from huggingface_hub import HfApi
sib = HfApi().repo_info(DOC, files_metadata=True).siblings
oid = next((getattr(getattr(s, "lfs", None), "sha256", None)
            for s in sib if s.rfilename == "model.onnx"), None)
chk("published doc model.onnx LFS sha256 unchanged",
    bool(oid) and oid.startswith("fe31555e"), str(oid)[:20])

# The file NAMES, not a count: a count lets an expected file be swapped for an unexpected one.
MANIFESTS = {
    ZERO: {"README.md", "config.json", "model.npz", "model.onnx", "model_tokens.onnx",
           "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
           "zero_encoder.py"},
    DOC: {"README.md", "config.json", "model.onnx", "special_tokens_map.json", "tokenizer.json",
          "tokenizer_config.json", "vocab.txt"},
}
for rid, want in MANIFESTS.items():
    names = {s.rfilename for s in HfApi().repo_info(rid).siblings} - {".gitattributes"}
    chk(f"{rid.split('/')[1]}: file set is exactly the manifest", names == want,
        f"extra={sorted(names - want)} missing={sorted(want - names)}" if names != want else "")

# both cards say what we think they say
for rid, must, mustnot in [
        (ZERO, ["library_name: fastembed", "constellation + stella", "Distance.COSINE",
                "TextEmbedding(NAME)", "0.4339"],
               ["release bar", "LightRetriever", "OpenSearch", "sentence-similarity", "89 lines"]),
        (DOC,  ["library_name: fastembed", "TextEmbedding(NAME)", "SentenceTransformer",
                "s2p_query", "0.82"],
               ["release bar", "sentence-similarity", "add_custom_model"])]:
    card = open(hf_hub_download(rid, "README.md")).read()
    miss = [m for m in must if m not in card]
    bad = [m for m in mustnot if m in card]
    chk(f"{rid.split('/')[1]}: card content", not miss and not bad,
        f"missing={miss} forbidden={bad}" if (miss or bad) else f"{card.count(chr(10))} lines")

# Every published file against the staged bytes, not just the two headline artifacts. The big
# graph is compared by LFS sha256 (which IS the content hash) so nothing multi-GB is pulled.
for rid, staged_dir in ((ZERO, "work/release/zero-v1"), (DOC, "work/release/stella-doc-onnx")):
    sd = pathlib.Path(staged_dir)
    if not sd.exists():
        print(f"SKIP  {rid.split('/')[1]}: no staging dir to compare against")
        continue
    sib = HfApi().repo_info(rid, files_metadata=True).siblings
    bad = []
    for f in sib:
        if f.rfilename == ".gitattributes":
            continue
        local = sd / f.rfilename
        if not local.exists():
            bad.append(f"{f.rfilename} (not staged)"); continue
        want_hash = hashlib.sha256(local.read_bytes()).hexdigest()
        oid = getattr(getattr(f, "lfs", None), "sha256", None)
        got = oid or hashlib.sha256(
            open(hf_hub_download(rid, f.rfilename), "rb").read()).hexdigest()
        if got != want_hash:
            bad.append(f.rfilename)
    chk(f"{rid.split('/')[1]}: every published file matches the staged bytes", not bad,
        f"differ: {bad}" if bad else f"{len(sib) - 1} files")

# Published vs staged card. The keyword checks above pass happily while the repo and the Hub
# have drifted -- which is exactly what happened when a card was edited after its last push.
for rid, staged in ((ZERO, "work/release/zero-v1/README.md"),
                    (DOC, "work/release/stella-doc-onnx/README.md")):
    sp = pathlib.Path(staged)
    if not sp.exists():
        print(f"SKIP  {rid.split('/')[1]}: no staging dir to compare against")
        continue
    live = open(hf_hub_download(rid, "README.md")).read()
    chk(f"{rid.split('/')[1]}: published card == staged card",
        live == sp.read_text(), "identical" if live == sp.read_text() else "DRIFTED — re-push")

print("\nLIVE VERIFICATION OK" if ok else "\nLIVE VERIFICATION FAILED")
sys.exit(0 if ok else 1)
