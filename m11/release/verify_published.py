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

ZERO = "DylanCouzon/constella-zero"
DOC = "DylanCouzon/stella-en-400M-v5-doc-onnx"
NANO = "DylanCouzon/constella-nano"
ok = True
def chk(n, p, d=""):
    global ok; ok &= bool(p); print(f"{'PASS' if p else 'FAIL'}  {n:52s}  {d}", flush=True)

# Pinned published revisions. A verifier that accepts whatever the Hub currently serves cannot
# detect an unexpected change to a model repository.
EXPECTED = {ZERO: "c231ef5a77", DOC: "977aca1b9c", NANO: "8742597e87"}

for rid in (ZERO, DOC, NANO):
    r = requests.get(f"https://huggingface.co/api/models/{rid}", timeout=30).json()
    chk(f"{rid.split('/')[1]}: public, anonymous", r.get("private") is False,
        f"{len(r.get('siblings', []))} files")
    chk(f"{rid.split('/')[1]}: head matches pinned published revision",
        r.get("sha", "").startswith(EXPECTED[rid]),
        f"{r.get('sha','')[:10]} (expected {EXPECTED[rid]})")

# the published zero table still hashes to FREEZE.json
from huggingface_hub import hf_hub_download
fz = json.load(open("m7/FREEZE.json"))
p = hf_hub_download(ZERO, "model.npz")
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
chk("published model.npz == FREEZE.json table_sha256", h == fz["table_sha256"], h[:16])

# the published document and Nano graphs are the byte strings the gates signed off
from huggingface_hub import HfApi
for rid, expected_hash in ((DOC, "fe31555e"), (NANO, "9ba0acf5")):
    sib = HfApi().repo_info(rid, files_metadata=True).siblings
    oid = next((getattr(getattr(s, "lfs", None), "sha256", None)
                for s in sib if s.rfilename == "model.onnx"), None)
    chk(f"published {rid.split('/')[1]} model.onnx LFS sha256 unchanged",
        bool(oid) and oid.startswith(expected_hash), str(oid)[:20])

# The file NAMES, not a count: a count lets an expected file be swapped for an unexpected one.
MANIFESTS = {
    ZERO: {"README.md", "config.json", "model.npz", "model.onnx", "model_tokens.onnx",
           "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
           "zero_encoder.py"},
    DOC: {"README.md", "config.json", "model.onnx", "special_tokens_map.json", "tokenizer.json",
          "tokenizer_config.json", "vocab.txt"},
    NANO: {"README.md", "config.json", "model.onnx", "special_tokens_map.json", "tokenizer.json",
           "tokenizer_config.json", "vocab.txt"},
}
for rid, want in MANIFESTS.items():
    names = {s.rfilename for s in HfApi().repo_info(rid).siblings} - {".gitattributes"}
    chk(f"{rid.split('/')[1]}: file set is exactly the manifest", names == want,
        f"extra={sorted(names - want)} missing={sorted(want - names)}" if names != want else "")

# both cards say what we think they say
for rid, must, mustnot in [
        (ZERO, ["library_name: fastembed", "Distance.COSINE", "TextEmbedding(NAME)",
                "DBSF", "prefetch 100", "338,076"],
               ["convex", "registered", "reserved-four", "np.stack"]),
        (DOC,  ["library_name: fastembed", "TextEmbedding(NAME)", "s2p_query",
                "259", "Natural Questions passages", "0.99999988"],
               ["registered", "reserved-four", "np.stack", "add_custom_model"]),
        (NANO, ["library_name: fastembed", "TextEmbedding(NAME)", "34,540,672",
                "199,999,721", "DylanCouzon/constella-zero"],
               ["registered", "reserved-four", "np.stack", "add_custom_model"])]:
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

# Published vs rendered source card. This comparison does not depend on retained staging dirs.
zero_card = pathlib.Path("m11/release/MODEL_CARD.md").read_text().replace("REPO_ID", ZERO)
doc_card = pathlib.Path("m11/release/MODEL_CARD_DOC.md").read_text().replace("REPO_ID", DOC)
doc_result = json.load(open("results/m11_doc_export.json"))
p32 = doc_result["parity_model.onnx"]
for key, value in {
        "PARITY_FIXTURE_COUNT": str(doc_result["n_fixtures"]),
        "PARITY_FP32_COS": f"{p32['min_cos']:.8f}",
        "PARITY_FP32_ABS": f"{p32['max_abs']:.2e}",
        "PARITY_FP32_NORMS": f"{p32['out_norms'][0]:.6f} to {p32['out_norms'][1]:.6f}",
}.items():
    doc_card = doc_card.replace(key, value)
nano_card = pathlib.Path("m14/MODEL_CARD.md").read_text()
for rid, expected_card in ((ZERO, zero_card), (DOC, doc_card), (NANO, nano_card)):
    live = open(hf_hub_download(rid, "README.md")).read()
    chk(f"{rid.split('/')[1]}: published card == rendered source card",
        live == expected_card, "identical" if live == expected_card else "DRIFTED")

print("\nLIVE VERIFICATION OK" if ok else "\nLIVE VERIFICATION FAILED")
sys.exit(0 if ok else 1)
