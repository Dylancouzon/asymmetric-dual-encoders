"""Package the Stella document vectors behind the workload studio for the Hugging Face Hub (exploratory).

Writes work/m15/hub_vectors/<dataset>/{doc_vecs.npy, doc_ids.json} plus manifest.json and the dataset card.
Vectors only: no passage text. Row i of doc_vecs.npy is the document doc_ids[i], in the order the Edge build uses.

  .venv/bin/python m15src/talk_package_vectors.py
  .venv/bin/python m15src/talk_package_vectors.py upload <repo_id>     # public dataset repo
"""
import json
import sys
from pathlib import Path

import numpy as np

from common import REPO, load_public, sha_file
import roster_ids as I

OUT = REPO / "work/m15/hub_vectors"
CARD = REPO / "m15/talk/demo/studio/HUB_CARD.md"


def fiqa():
    data = load_public("fiqa")
    src = REPO / "artifacts/demo-stella/fiqa"
    ids = json.loads((src / "doc_ids.json").read_text())
    vecs = np.load(src / "doc_vecs.npy")
    pos = {d: i for i, d in enumerate(ids)}
    return data["doc_ids"], vecs[[pos[d] for d in data["doc_ids"]]], "local cache; passed the M15 reproduction gate (E2)"


def nq():
    d = REPO / "work/m15/vecs/nq_stella"
    ids = json.loads((d / "doc_ids.json").read_text())
    out = np.lib.format.open_memmap(OUT / "nq/doc_vecs.npy", mode="w+", dtype=np.float16, shape=(len(ids), 1024))
    off = 0
    for p in sorted(d.glob("shard_*.npy")):
        a = np.load(p, mmap_mode="r")
        out[off:off + len(a)] = a
        off += len(a)
    if off != len(ids):
        raise RuntimeError(f"NQ has {off} vectors for {len(ids)} docs")
    out.flush()
    return ids, None, "NVIDIA RTX 3080 (CUDA), fp32 compute; parity vs Mac MPS: min cosine 0.99994 on 256 passages"


def package():
    manifest = {"model": I.STELLA_MODEL, "revision": I.STELLA_REVISION, "dim": 1024, "dtype": "float16",
                "max_length": 512, "prompt": None, "encoder_code": "m7src/teacher.py (via m15src/vectors15.encode_docs)",
                "datasets": {}}
    for name, fn in (("fiqa", fiqa), ("nq", nq)):
        (OUT / name).mkdir(parents=True, exist_ok=True)
        ids, vecs, device = fn()
        if vecs is not None:
            np.save(OUT / name / "doc_vecs.npy", vecs.astype(np.float16))
        (OUT / name / "doc_ids.json").write_text(json.dumps(ids))
        arr = np.load(OUT / name / "doc_vecs.npy", mmap_mode="r")
        if arr.shape != (len(ids), 1024) or arr.dtype != np.float16:
            raise RuntimeError(f"{name}: bad shape {arr.shape} {arr.dtype}")
        manifest["datasets"][name] = {
            "n_docs": len(ids), "encoded_on": device,
            "doc_vecs.npy": sha_file(OUT / name / "doc_vecs.npy"), "doc_ids.json": sha_file(OUT / name / "doc_ids.json")}
        print(name, manifest["datasets"][name], flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (OUT / "README.md").write_text(CARD.read_text())


def upload(repo_id):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
    info = api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(OUT),
                             commit_message="Stella document vectors for BEIR FiQA and NQ")
    print(info, flush=True)


if __name__ == "__main__":
    upload(sys.argv[2]) if sys.argv[1:2] == ["upload"] else package()
