#!/usr/bin/env python3
"""Read-only dependency preflight for fresh registered cloud E arms; never score or train.

Reads only registered training metadata and admitted COV data/caches. Requires local HF
caches and verifies every COV document chunk plus its stitched array. No teacher cache
writer/encoder, DEV-6, reserved, six-set or LoTTE evaluator is called. The sole intentional
write is a new results/m13_e_preflight.json receipt after all checks pass; never overwrite.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
for name in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / name))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_identity(teacher, transformers_version, name, texts):
    """Mirror teacher9's serialized identity, including its source hash (never encode)."""
    spec = teacher.TEACHERS["stella-400M-v5"]
    return {"name": "m10cov-" + name + "-d", "repo": spec["repo"],
            "revision": spec["revision"], "role": "doc", "prompt": spec["doc_prompt"],
            "max_length": 512, "dim": spec["dim"], "store_dtype": "fp16",
            "compute_dtype": str(teacher.COMPUTE_DTYPE),
            "config_kwargs": spec.get("config_kwargs") or {},
            "transformers": transformers_version,
            "encoder_code_sha256": sha(teacher.__file__)[:16],
            "corpus_sha256": teacher._sha_texts(texts), "path": "sentence-transformers"}


def verify_cache(root, blob, count):
    import numpy as np
    key = hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12]
    directory = root / (blob["name"] + "-stella-400M-v5-" + key)
    require(directory.is_dir(), f"Missing exact COV cache: {directory}")
    require(json.loads((directory / "meta.json").read_text()) == blob,
            f"COV metadata identity mismatch: {directory}")
    manifest = json.loads((directory / "chunks.json").read_text())
    chunk_size, dim = 50_000, blob["dim"]
    names = [f"chunk_{i:05d}.npy" for i in range((count + chunk_size - 1) // chunk_size)]
    require(set(manifest) == set(names), f"Wrong COV chunk inventory: {directory}")
    combined = directory / "combined.f16"
    require(combined.is_file() and combined.stat().st_size == count * dim * 2,
            f"Missing or truncated stitched COV cache: {combined}")
    mapped = np.memmap(combined, dtype=np.float16, mode="r", shape=(count, dim))
    for i, name in enumerate(names):
        path = directory / name
        require(sha(path) == manifest[name]["sha256"], f"COV chunk hash mismatch: {path}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        lo, hi = i * chunk_size, min(count, (i + 1) * chunk_size)
        require(values.shape == (hi - lo, dim) and values.dtype == np.float16
                and manifest[name]["rows"] == hi - lo, f"Wrong COV chunk shape: {path}")
        require(np.array_equal(values, mapped[lo:hi]), f"Stitched COV cache differs: {path}")
        require(np.isfinite(values).all(), f"Non-finite COV vectors: {path}")
    return {"path": str(directory), "documents": count, "chunks": len(names),
            "identity": key, "verified": True}


def main():
    output = REPO / "results/m13_e_preflight.json"
    require(not output.exists(), f"Preflight receipt already exists: {output}")
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    import datasets
    import huggingface_hub.constants
    import transformers
    import run_arm as runner
    import cov_probe
    import cov_ledger  # legacy imports clear env; set both env and library flags afterward
    import cov_macro
    import teacher9
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    datasets.config.HF_DATASETS_OFFLINE = True
    huggingface_hub.constants.HF_HUB_OFFLINE = True

    reg = runner.SL.cfg()
    errors = runner.SL.validate(reg)
    require(not errors, f"Screen registry validation failed: {errors}")
    verdict = runner.f_verdict(reg)
    plans = {}
    for arm in ("E-bs32", "E-bs128"):
        out, record, published = runner.record_paths(arm, False)
        require(not out.exists() and not record.exists() and not published.exists(),
                f"Registered output already exists for {arm}; fresh launch refused")
        plans[arm] = runner.arm_plan(arm, reg, verdict=verdict)
    warm = reg.get("warm_start", {}).get("G-MLP", {})
    require(warm.get("n_fit") == 60_000 and warm.get("seed") == 21,
            "Registered 60k warm-start configuration is absent or changed")

    units = cov_probe.units()
    require(len({u[0] for u in units}) == len(units), "Duplicate COV units")
    cov_macro.assert_surface({u[0]: u[1] for u in units})
    # BRIGHT has one shared cache in the registry's slice order, exactly as cov_eval10.
    texts = {}
    for uid, _family, _qs, _qids, docs, _dids, _qrels in units:
        if not uid.startswith("BRIGHT/"):
            texts[uid] = docs
    by_id = {u[0]: u for u in units}
    texts["BRIGHT"] = [text for sl in cov_probe.BRIGHT_SLICES
                       for text in by_id["BRIGHT/" + sl][4]]
    caches = []
    for name, docs in texts.items():
        blob = cache_identity(teacher9, transformers.__version__, name, docs)
        caches.append(verify_cache(teacher9.ENC9, blob, len(docs)))
    receipt = {"status": "PASSED", "scope": "dependency checks only; no scoring or training",
               "git_head": runner.git_head(), "registry_sha256": sha(runner.REGISTRY),
               "warm_start": {"n_fit": warm["n_fit"], "seed": warm["seed"]},
               "plans": plans, "cov_units": len(units),
               "cov_queries": sum(len(u[2]) for u in units), "cov_caches": caches}
    payload = json.dumps(receipt, indent=2) + "\n"
    with output.open("x") as stream:
        stream.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
