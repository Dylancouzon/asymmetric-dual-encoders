"""M17 P0: hand-authored tokenization fixtures and disposable resource measurements.

No dataset, retrieval evaluation, network request or real-weight training. See m17/LEDGER.md.
Run from repo root: .venv/bin/python m17src/planning_probe.py [--gpu]
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import tokenizers
from tokenizers import AddedToken, Tokenizer

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "work/release/zero-v1"
TERMS = ["s3", "k8s", "kubernetes", "kubectl", "eks", "gke", "aks", "iam", "ec2",
         "terraform", "helm", "istio", "prometheus", "grafana", "postgresql", "redis",
         "nginx", "grpc", "oauth2", "oidc", "jwt", "vpc", "cidr", "dns", "coredns",
         "etcd", "containerd", "argocd", "crd", "ingress", "c++", "c#", ".net",
         "s3://", "CrashLoopBackOff", "aws_s3_bucket"]
BOUNDARIES = ["S3 bucket policy", "s3-compatible storage", "s3://my-bucket/path",
              "s3client", "xs3", "s3_bucket", "k8s deployment", "K8S networking",
              "k8s-client", "k8sclient", "k8s_cluster", "C++ compiler", "C# code",
              ".NET runtime", "oauth2 vs oauth20", "S3 heart sound", "S3 galaxy phone"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_probe():
    spec = importlib.util.spec_from_file_location("m17_zero_reference", ROOT / "m11/release/zero_encoder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = time.perf_counter()
    enc = module.ZeroQueryEncoder(BUNDLE)
    load_seconds = time.perf_counter() - started
    old = enc.tokenizer
    new = Tokenizer.from_str(old.to_str())
    extension = [t.lower() for t in TERMS if old.token_to_id(t.lower()) is None]
    added = new.add_tokens([AddedToken(t, single_word=True, normalized=True) for t in extension])
    segmentation = [{"text": t, "old": old.encode(t, add_special_tokens=False).tokens,
                     "extended": new.encode(t, add_special_tokens=False).tokens}
                    for t in TERMS + BOUNDARIES]
    samples = [f"how to configure {t} access and troubleshoot errors" for t in TERMS]
    timing = []
    for batch in (32, 256):
        texts = [samples[i % len(samples)] for i in range(batch)]
        enc.encode(texts)
        observations = []
        for _ in range(7):
            start = time.perf_counter()
            enc.encode(texts)
            observations.append(time.perf_counter() - start)
        timing.append({"batch": batch, "repeats": 7, "batch_seconds": observations,
                       "median_ms_per_query": float(np.median(observations) * 1000 / batch)})
    rows = np.concatenate([enc.rows, np.zeros((added, enc.dim), np.float32)])
    for term in extension:
        ids = old.encode(term, add_special_tokens=False).ids
        # Composition by sum is a reference initialization, not a quality claim.
        rows[new.token_to_id(term)] = enc.rows[ids].sum(0)

    def pooled(ids):
        uniq, counts = np.unique(ids, return_counts=True)
        v = (rows[uniq] * np.sqrt(counts)[:, None]).sum(0)
        return v / max(np.linalg.norm(v), 1e-6)

    composition = []
    for text in ("configure s3 bucket", "s3 s3 bucket", "s3 k8s bucket", "s3 3 bucket",
                 "k8s k8s networking", "k8s kubernetes networking"):
        a = pooled(old.encode(text).ids)
        b = pooled(new.encode(text).ids)
        composition.append({"text": text, "max_abs": float(np.max(np.abs(a-b))),
                            "cosine": float(a @ b), "old_tokens": old.encode(text).tokens,
                            "new_tokens": new.encode(text).tokens})
    return {"original_vocab": old.get_vocab_size(), "fixture_added_rows": added,
            "segmentation": segmentation, "composition": composition,
            "load_seconds": load_seconds, "cpu_timing": timing,
            "resident_rows_bytes": enc.rows.nbytes,
            "size_arithmetic": [{"rows": v, "table_parameters": v * 1024,
                "with_learned_scalar_parameters": v * 1025,
                "int8_codes_and_fp32_scales_bytes": v * 1028,
                "fp32_resident_rows_bytes": v * 4096}
                for v in (30522, 33594, 34179, 65536)],
            "interpretation": "Illustrative fixtures only; no coverage or quality estimate."}


def gpu_probe():
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        return {"available": False}
    results = []
    for batch in (128, 256):
        torch.manual_seed(17)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rows = torch.nn.Parameter(torch.randn(33594, 1024, device="cuda") * 0.02)
        opt = torch.optim.Adam([rows], lr=1e-4)
        ids = torch.randint(0, len(rows), (batch * 24,), device="cuda")
        offsets = torch.arange(0, batch * 24, 24, device="cuda")
        docs = F.normalize(torch.randn(batch, 64, 1024, device="cuda", dtype=torch.float16), dim=-1)
        targets = F.normalize(torch.randn(batch, 1024, device="cuda"), dim=-1)
        teacher_prob = F.softmax(torch.randn(batch, 64, device="cuda"), dim=-1)

        def step():
            opt.zero_grad(set_to_none=True)
            q = F.normalize(F.embedding_bag(ids, rows, offsets, mode="mean"), dim=-1)
            scores = torch.bmm(docs, q.half().unsqueeze(-1)).squeeze(-1).float() / 0.05
            loss = (1 - (q * targets).sum(-1)).mean() + F.kl_div(
                F.log_softmax(scores, dim=-1), teacher_prob, reduction="batchmean")
            loss.backward()
            opt.step()
            return loss

        for _ in range(5):
            step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            loss = step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        results.append({"batch": batch, "tokens_per_query": 24, "candidates": 64,
                        "steps": 20, "seconds": elapsed, "ms_per_step": elapsed * 50,
                        "examples_per_second": batch * 20 / elapsed,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "allocator_retries": torch.cuda.memory_stats()["num_alloc_retries"],
                        "loss_finite": bool(torch.isfinite(loss).item())})
        del step, opt, rows, ids, offsets, docs, targets, teacher_prob, loss
    torch.cuda.empty_cache()
    return {"available": True, "device": torch.cuda.get_device_name(), "torch": torch.__version__,
            "shapes": results,
            "limitations": "Synthetic resident tensors, uniform token IDs and mean pooling; excludes real sqrt counts, I/O, teacher encoding, candidate mining, regularization, evaluation and checkpointing. Feasibility only, not an end-to-end training forecast."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    inputs = [BUNDLE / "model.npz", BUNDLE / "tokenizer.json", BUNDLE / "config.json",
              ROOT / "m11/release/zero_encoder.py", ROOT / "m17/LEDGER.md", Path(__file__)]
    result = {"diagnostic": "M17-P0", "date": "2026-09-11", "python": platform.python_version(),
              "numpy": np.__version__, "tokenizers": tokenizers.__version__,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
              "cpu": cpu_probe()}
    if args.gpu:
        result["gpu"] = gpu_probe()
    for p in inputs:
        assert digest(p) == result["input_sha256"][str(p.relative_to(ROOT))], p
    output = ROOT / "results/m17_planning_probe.json"
    if output.exists():
        raise FileExistsError(f"Preserve the existing observation: {output}")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "cpu_timing": result["cpu"]["cpu_timing"],
                      "gpu": result.get("gpu")}, indent=2))


if __name__ == "__main__":
    main()
