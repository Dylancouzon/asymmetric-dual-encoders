#!/usr/bin/env python3
"""CPU dependency checks for the registered build; never train, score, or encode.

The cloud controller verifies transferred file hashes before this helper. This
helper exercises full corpus assembly and existing COV/DEV-6 caches. Assembly may
populate CPU token caches; DEV-6 cache hits may refresh metadata, preserving all
legacy provenance. No protected six, reserved, or LoTTE payload is opened.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
DEV6 = ("nq-250k", "hotpotqa", "cqadup-programmers", "cqadup-physics",
        "heldout-train", "heldout-longq")
TEXT_CACHES = {"dev-" + name + "-docs" for name in DEV6[:4]}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def refuse_encode(*args, **kwargs):
    raise RuntimeError("Build preflight refuses teacher encoding or model loading")


def verify_dev_cache(teacher, name, texts, kwargs):
    """Require a complete reusable cache before its normal verify=False reader runs."""
    import numpy as np
    require(name in TEXT_CACHES, "Unexpected DEV-6 text cache: " + name)
    key, _ = teacher.cache_key(name, kwargs.get("prefix", ""), kwargs.get("max_length", 512),
                               teacher.TEACHER, teacher.TEACHER_REV,
                               teacher.sha_texts(texts), kwargs["dtype"])
    directory = teacher.ENC / key
    manifest = json.loads((directory / "shards.json").read_text())
    count = (len(texts) + teacher.SHARD - 1) // teacher.SHARD
    shard_ids = [f"{i:05d}" for i in range(count)]
    require(set(manifest["shards"]) == set(shard_ids), "DEV-6 shard inventory mismatch")
    sources, legacy = [], 0
    for i, sid in enumerate(shard_ids):
        part = directory / ("shard_" + sid + ".npy")
        row = manifest["shards"][sid]
        require(sha(part) == row["sha256"], "DEV-6 shard checksum mismatch: " + str(part))
        values = np.load(part, mmap_mode="r", allow_pickle=False)
        rows = min(teacher.SHARD, len(texts) - i * teacher.SHARD)
        require(values.shape == (rows, 1024) and values.dtype == np.float16,
                "DEV-6 shard shape or dtype mismatch: " + str(part))
        require(row.get("shard_size") in (None, teacher.SHARD), "DEV-6 shard layout changed")
        sources.append(row["sha256"])
        legacy += bool(row.get("trusted_on_first_use"))
        del values
    combined_legacy = False
    if count > 1:
        combined = directory / "combined.f16"
        row = manifest.get("combined", {})
        require(combined.is_file() and combined.stat().st_size == len(texts) * 1024 * 2
                and row.get("n_rows") == len(texts) and row.get("from_shard_sha256") == sources,
                "DEV-6 stitched cache missing or stale")
        require(sha(combined) == row.get("sha256"), "DEV-6 stitched cache checksum mismatch")
        combined_legacy = bool(row.get("trusted_on_first_use"))
    return {"name": name, "key": key, "rows": len(texts), "verified_current_bytes": True,
            "legacy_shards": legacy, "legacy_combined": combined_legacy}


def check_dev6(teacher, evaluator):
    """Exercise the actual six development document-vector readers, never query scoring."""
    original_cached, original_encode, original_load = teacher.encode_cached, teacher.encode, teacher.load_teacher
    checks = []

    def cached(name, texts, *args, **kwargs):
        require(not args, "Unexpected positional DEV-6 cache options")
        check = verify_dev_cache(teacher, name, texts, kwargs)
        # This is the existing descriptive DEV-6 policy, not a provenance upgrade.
        kwargs["verify"] = False
        values = original_cached(name, texts, **kwargs)
        require(teacher.PROVENANCE[name]["shards_written_now"] == 0,
                "DEV-6 preflight unexpectedly wrote vector shards")
        checks.append(check)
        return values

    teacher.encode, teacher.load_teacher, teacher.encode_cached = refuse_encode, refuse_encode, cached
    try:
        components = tuple(evaluator.components("DEV6"))
        require(components == DEV6, "Unexpected DEV-6 component surface")
        rows = {}
        for component in components:
            ids, qids, queries, _rels, vectors = evaluator.doc_vecs(component, evaluator.INCUMBENT)
            require(vectors.shape == (len(ids), 1024) and len(qids) == len(queries),
                    "DEV-6 vector/query alignment mismatch: " + component)
            rows[component] = {"documents": len(ids), "queries": len(qids), "dtype": str(vectors.dtype)}
            print("DEV-6 cache ready:", component, flush=True)
            del ids, qids, queries, _rels, vectors
            gc.collect()
        require({check["name"] for check in checks} == TEXT_CACHES,
                "DEV-6 did not exercise all four text caches")
        return {"components": rows, "cache_checks": checks,
                "legacy_provenance": "Current bytes match recorded hashes; historical trust-on-first-use labels remain unchanged."}
    finally:
        teacher.encode_cached, teacher.encode, teacher.load_teacher = original_cached, original_encode, original_load


def check_cov():
    import transformers
    import cov_probe
    import cov_macro
    import teacher9
    import m13_e_preflight as existing
    units = cov_probe.units()
    require(len({unit[0] for unit in units}) == len(units), "Duplicate COV units")
    cov_macro.assert_surface({unit[0]: unit[1] for unit in units})
    texts = {uid: docs for uid, _family, _qs, _qids, docs, _dids, _rels in units
             if not uid.startswith("BRIGHT/")}
    by_id = {unit[0]: unit for unit in units}
    texts["BRIGHT"] = [text for sl in cov_probe.BRIGHT_SLICES for text in by_id["BRIGHT/" + sl][4]]
    caches = []
    for name, docs in texts.items():
        identity = existing.cache_identity(teacher9, transformers.__version__, name, docs)
        caches.append(existing.verify_cache(teacher9.ENC9, identity, len(docs)))
        print("COV cache ready:", name, flush=True)
    return {"units": len(units), "queries": sum(len(unit[2]) for unit in units), "caches": caches}


def check_build():
    import build13 as build
    from transformers import AutoTokenizer
    cfg, cfg_path = build.BL.load()
    reg, batch, source, gate, report = build._preflight(
        {}, cfg, cfg_path, smoke=False, device="cuda", batch=None, lotte_gate=None,
        compile_step=False, real_eval=False, max_len=None, ckpt_every=None, n_fit=None)
    require(not any(path.exists() for path in build.run_paths(cfg["arm"], False) if path is not None),
            "Registered build output already exists")
    build.BL.check_dose(cfg, batch)
    merged = build.BL.merged_registry(cfg, reg, batch=batch)
    plan = build.build_plan(cfg, merged, batch)
    tokenizer = AutoTokenizer.from_pretrained(build.N.REPOS[plan["student"]], local_files_only=True)
    batch_fn, manifest = build.CL.assemble_arm(
        cfg["arm"], tokenizer, plan["student"], batch_size=batch, seed=plan["seed"],
        max_len=build.MAX_LEN, registry=merged, verbose=True)
    require(plan["dose_examples"] == 200_000_000 and not plan["cut_corpus"]
            and manifest.get("rescreen10_report_validated") is True
            and manifest.get("document_policy") is not None,
            "Full registered build assembly contract was not satisfied")
    stable_manifest = build.R.fingerprint_manifest(manifest)
    result = {"plan": plan, "batch_source": source, "gate": gate,
              "assembly_manifest_sha256": hashlib.sha256(json.dumps(stable_manifest, sort_keys=True, default=str).encode()).hexdigest(),
              "validation": report}
    del batch_fn, manifest, tokenizer
    gc.collect()
    build.CL.release_arena()
    return result


def main():
    output = REPO / "results/m13_build_preflight.json"
    require(not output.exists(), "Build preflight receipt already exists")
    os.environ.update(CUDA_VISIBLE_DEVICES="", M7_DEVICE="cpu", M7_ENCODER="stella-400M-v5",
                      HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    for directory in ("scripts", "m7src", "m9src", "m10src", "m13src"):
        sys.path.insert(0, str(REPO / directory))
    import m9base  # installs the existing protected-path guard
    import cov_ledger  # legacy imports clear offline settings; restore library flags below
    import datasets
    import huggingface_hub.constants
    import teacher
    import eval9
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    datasets.config.HF_DATASETS_OFFLINE = True
    huggingface_hub.constants.HF_HUB_OFFLINE = True
    metadata = ("m13/build_transfer_manifest.json", "m13/build_config.json", "m10/screen_registry.json",
                "m13/LOTTE_GATE.json", "m13/LOTTE_GATE_MANIFEST.json", "results/m10_screen_verdicts.json",
                "results/m10_arm_E-bs32.json", "results/m10_arm_E-bs128.json")
    code = ("scripts/m13_build_preflight.py", "scripts/m13_e_preflight.py", "m13src/build13.py",
            "m13src/build_lock.py", "m10src/run_arm.py", "m10src/corpus_loader.py", "m10src/data10.py",
            "m10src/rescreen10.py", "m10src/nano10.py", "m10src/cov_probe.py", "m9src/teacher9.py",
            "m9src/eval9.py", "m7src/teacher.py", "m7src/dev_eval.py", "m7src/heldout.py", "m7src/devsuite.py")
    bound = {path: sha(REPO / path) for path in metadata + code}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    transfer = json.loads((REPO / metadata[0]).read_text())
    require(transfer.get("status") == "HASH_PINNED_NOT_LAUNCH_APPROVAL"
            and transfer["files"] == len(transfer["entries"]) == 394, "Unexpected build transfer inventory")
    for item in transfer["entries"]:
        path = Path(item["destination"])
        require(path.is_file() and path.stat().st_size == item["bytes"],
                "Missing or truncated staged build input: " + str(path))
    print("Checking full registered build assembly", flush=True)
    assembly = check_build()
    print("Checking admitted COV caches", flush=True)
    cov = check_cov()
    gc.collect()
    print("Checking descriptive DEV-6 cache reuse", flush=True)
    development = check_dev6(teacher, eval9)
    require(all(sha(REPO / path) == digest for path, digest in bound.items()),
            "Build preflight source or registration changed during checks")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() == head,
            "Build preflight HEAD changed during checks")
    receipt = {"status": "PASSED", "git_head": head,
               "scope": "CPU dependency checks only; no scoring, training or teacher encoding",
               "full_uncut_assembly_verified": True, "dev6_cache_reuse_verified": True,
               "cov_cache_integrity_verified": True, "artifact_sha256": bound,
               "transfer_files_present": transfer["files"], "transfer_hash_verification": "owned by cloud controller before this helper",
               "assembly": assembly, "cov": cov, "dev6": development}
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print("Build dependency preflight PASSED", flush=True)


if __name__ == "__main__":
    main()
