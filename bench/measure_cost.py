"""Edge-relevant costs per query-encoder class, measured on CPU (batch 1, idle machine).

Outputs results/costs.json:
  latency_ms  - median single-query encode time over 50 runs (10 warmup)
  load_s      - model/table load time inside a warm process (imports excluded, page cache warm)
  disk_mb     - query-side artifact size on disk
"""
import json
import statistics
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "costs.json"
QUERY = "how does high blood pressure medication affect kidney function"

ST_MODELS = {
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "gte-small": "thenlper/gte-small",
    "arctic-embed-s": "Snowflake/snowflake-arctic-embed-s",
    "granite-small-r2": "ibm-granite/granite-embedding-small-english-r2",
    "mdbr-leaf-ir": "MongoDB/mdbr-leaf-ir",
    "potion-base-8M": "minishlab/potion-base-8M",
    "potion-retrieval-32M": "minishlab/potion-retrieval-32M",
    "static-retrieval-mrl-en-v1": "sentence-transformers/static-retrieval-mrl-en-v1",
}


def timed(fn, n=50, warmup=10):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main():
    results = {}
    import torch
    from sentence_transformers import SentenceTransformer

    for slug, hf_id in ST_MODELS.items():
        t = time.perf_counter()
        model = SentenceTransformer(hf_id, device="cpu", model_kwargs={"dtype": torch.float32})
        load_s = time.perf_counter() - t
        lat = timed(lambda: model.encode(QUERY, normalize_embeddings=True))
        # parameter bytes, not repo size: HF repos bundle onnx/openvino/tf duplicates
        numel = sum(p.numel() for p in model.parameters())
        results[slug] = {
            "latency_ms": round(lat, 2),
            "load_s": round(load_s, 2),
            "params_m": round(numel / 1e6, 1),
            "weights_fp32_mb": round(numel * 4 / 1e6, 1),
            "weights_fp16_mb": round(numel * 2 / 1e6, 1),
        }
        print(slug, results[slug], flush=True)
        del model

    # LightRetriever lookup table (fp16 npy + tokenizer)
    from transformers import AutoTokenizer

    table_path = REPO / "artifacts/lightretriever-qwen2.5-1.5b/table_websearch.npy"
    if table_path.exists():
        t = time.perf_counter()
        tok = AutoTokenizer.from_pretrained("lightretriever/lightretriever-qwen2.5-1.5b")
        table = np.load(table_path).astype(np.float32)
        load_s = time.perf_counter() - t

        def lr_encode():
            ids = tok(QUERY, add_special_tokens=False)["input_ids"]
            v = table[ids].mean(0)
            return v / (np.linalg.norm(v) + 1e-12)

        results["lightretriever-lookup"] = {
            "latency_ms": round(timed(lr_encode), 3),
            "load_s": round(load_s, 2),
            "disk_mb": round(table_path.stat().st_size / 1e6, 1),
        }
        print("lightretriever-lookup", results["lightretriever-lookup"], flush=True)

        # int8 table size (quality delta for int8 is recorded by run_lightretriever.py eval)
        results["lightretriever-lookup-int8"] = {"disk_mb": round((table.shape[0] * table.shape[1] + table.shape[0] * 4) / 1e6, 1)}
        del table

    # OpenSearch inference-free query side (tokenizer + idf dict)
    from huggingface_hub import hf_hub_download

    from run_opensearch import MODEL_ID as os_id, load_idf

    t = time.perf_counter()
    os_tok = AutoTokenizer.from_pretrained(os_id)
    idf = load_idf(os_tok)
    idf_path = hf_hub_download(os_id, "idf.json")
    load_s = time.perf_counter() - t

    def os_encode():
        ids = set(os_tok(QUERY, add_special_tokens=False)["input_ids"])
        return {i: idf.get(i, 1.0) for i in ids}

    results["opensearch-query-side"] = {
        "latency_ms": round(timed(os_encode), 3),
        "load_s": round(load_s, 2),
        "disk_mb": round(Path(idf_path).stat().st_size / 1e6, 1),
    }
    print("opensearch-query-side", results["opensearch-query-side"], flush=True)

    OUT.write_text(json.dumps(results, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
