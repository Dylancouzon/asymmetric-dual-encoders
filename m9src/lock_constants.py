"""Materialize every quantity the M9.0 lock has to state as a number.

Run once, before the lock is committed. Nothing here trains, evaluates retrieval, or opens a
protected path: it tokenizes locked text lists and hashes locked id lists. Output:
`results/m9_lock_constants.json`, which `m9/registry.json` and `m9/LEDGER.md` then quote.

Codex 2026-08-30 forced most of this: pinned qid manifests (B5), a token budget computed at lock
time rather than "recorded at run time" (B2 + post-number-freedom table), the eligible-row union
count and mask hash (MINOR-4), and framework pins (B4).
"""
import hashlib
import json
import platform

import numpy as np

import m9base
from m9base import RESULTS, WORK, DEV_FULL

import data as m9data   # noqa: E402
import nano             # noqa: E402

DOC_CANDIDATES = 400_000
DOC_CANDIDATE_SEED = 9


def _sha_list(items):
    h = hashlib.sha256()
    for x in items:
        h.update(str(x).encode())
        h.update(b"\x00")
    return h.hexdigest()


def qid_manifest():
    import dev_eval
    out = {}
    for comp in DEV_FULL:
        _doc_ids, _dt, q_ids, _qt, qrels, dv = dev_eval.doc_vecs(comp)
        q = [str(x) for x in q_ids]
        out[comp] = {"n_queries": len(q), "qids_sha256": _sha_list(sorted(q)),
                     "qids_ordered_sha256": _sha_list(q),
                     "n_qrels": len(qrels), "n_docs": int(dv.shape[0]), "dim": int(dv.shape[1])}
    return out


def token_budget():
    """-> per-student non-pad token totals for the baseline schedule, and the doc candidate list.
    Deterministic: 16 epochs x every query text once, so the total is 16 * sum(len(ids))."""
    from m9base import M9
    reg = json.loads((M9 / "registry.json").read_text())
    epochs = reg["dose"]["epochs_query_only"]
    texts = json.loads((WORK / "m9_screen_queries.json").read_text())

    rows, dmeta = m9data.doc_pool_rows(DOC_CANDIDATES, DOC_CANDIDATE_SEED)
    dtexts = m9data.row_texts(rows)

    out = {"doc_candidates": {"n": DOC_CANDIDATES, "seed": DOC_CANDIDATE_SEED,
                              "rows_sha256": dmeta["rows_sha256"],
                              "n_eligible": dmeta["n_eligible"],
                              "n_banned": dmeta["n_banned"]},
           "students": {}}
    for key, spec in nano.STUDENTS.items():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
        q_ids = nano.pretokenize(tok, texts, reg["dose"]["max_seq"], verbose=False)
        qlen = np.array([len(x) for x in q_ids], dtype=np.int64)
        d_ids = nano.pretokenize(tok, [reg["templates"]["doc_student"] + t for t in dtexts],
                                 reg["dose"]["max_seq"], verbose=False)
        dlen = np.array([len(x) for x in d_ids], dtype=np.int64)
        t_base = int(qlen.sum()) * epochs
        # arm 6 consumes the doc candidate list IN ORDER until 0.30 * T_base tokens are reached
        want = int(round(0.30 * t_base))
        cum = np.cumsum(dlen)
        n_docs = int(np.searchsorted(cum, want) + 1)
        out["students"][key] = {
            "query_tokens_per_epoch": int(qlen.sum()),
            "T_base_nonpad_tokens": t_base,
            "query_len": {"mean": round(float(qlen.mean()), 2), "p50": int(np.percentile(qlen, 50)),
                          "p95": int(np.percentile(qlen, 95)), "max": int(qlen.max())},
            "doc_len": {"mean": round(float(dlen.mean()), 2), "p50": int(np.percentile(dlen, 50)),
                        "p95": int(np.percentile(dlen, 95)), "max": int(dlen.max())},
            "mix_doc_token_target": want,
            "mix_n_docs_single_epoch": n_docs,
            "mix_doc_tokens_realized": int(cum[n_docs - 1]),
            "mix_query_token_target": t_base - want,
            "per_step_token_budget": round(t_base / reg["dose"]["steps"], 1),
        }
    return out


def artifact_sizes():
    import onnx  # noqa: F401
    out = {}
    for key in nano.STUDENTS:
        m = nano.Nano(key)
        n = m.n_params()
        out[key] = {"params": n, "weights_fp16_bytes": n * 2,
                    "weights_fp16_MB_decimal": round(n * 2 / 1e6, 3)}
        del m
    return out


def framework():
    import numpy
    import onnxruntime
    import torch
    import transformers
    mods = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "numpy": numpy.__version__,
            "onnxruntime": onnxruntime.__version__}
    try:
        import sentence_transformers
        mods["sentence_transformers"] = sentence_transformers.__version__
    except Exception:
        pass
    try:
        import fastembed
        mods["fastembed"] = fastembed.__version__
    except Exception:
        mods["fastembed"] = None
    mods["cuda"] = torch.version.cuda
    mods["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return mods


def code_hashes():
    out = {}
    for p in sorted((m9base.REPO / "m9src").glob("*.py")):
        out[f"m9src/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    for rel in ("m7src/evalkit.py", "m7src/teacher.py", "m7src/devsuite.py", "m7src/dev_eval.py",
                "m7src/pool.py", "m7src/mix.py", "m7src/train.py", "m7src/heldout.py",
                "m8src/paths_guard.py"):
        p = m9base.REPO / rel
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def main():
    blob = {"_what": "every numeric quantity the M9.0 lock states, materialized once, before the "
                     "lock is committed. No training, no retrieval evaluation, no protected path.",
            "framework": framework(),
            "artifact_sizes": artifact_sizes(),
            "token_budget": token_budget(),
            "qid_manifest": qid_manifest(),
            "code_hashes": code_hashes()}
    (RESULTS / "m9_lock_constants.json").write_text(json.dumps(blob, indent=2))
    print(json.dumps({k: v for k, v in blob.items() if k != "code_hashes"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
