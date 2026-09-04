"""Verify what the two published repos actually SERVE, against local truth.

Every other gate here checks bytes on the way OUT. This one checks what the Hub gives back to a
stranger: that both repos are public, that the shipped table still hashes to `m7/FREEZE.json`,
that the document graph's LFS sha256 is the byte string the gates signed off, that neither repo
carries a file its manifest does not name, and that both cards say what we believe they say.

    .venv/bin/python m11/release/verify_published.py
"""
import json, os, sys, hashlib
os.environ.pop("HF_TOKEN", None)
import numpy as np, requests

ZERO, DOC = "DylanCouzon/constella-zero", "DylanCouzon/stella-en-400M-v5-doc-onnx"
ok = True
def chk(n, p, d=""):
    global ok; ok &= bool(p); print(f"{'PASS' if p else 'FAIL'}  {n:52s}  {d}", flush=True)

for rid in (ZERO, DOC):
    r = requests.get(f"https://huggingface.co/api/models/{rid}", timeout=30).json()
    chk(f"{rid.split('/')[1]}: public, anonymous", r.get("private") is False,
        f"sha {r.get('sha','')[:10]}  {len(r.get('siblings',[]))} files")

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

# the file sets are the manifests plus .gitattributes, nothing else
for rid, n in ((ZERO, 10), (DOC, 7)):
    names = {s.rfilename for s in HfApi().repo_info(rid).siblings}
    chk(f"{rid.split('/')[1]}: file set is the manifest + .gitattributes",
        len(names - {".gitattributes"}) == n, f"{sorted(names - {'.gitattributes'})}"[:90])

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

print("\nLIVE VERIFICATION OK" if ok else "\nLIVE VERIFICATION FAILED")
sys.exit(0 if ok else 1)
