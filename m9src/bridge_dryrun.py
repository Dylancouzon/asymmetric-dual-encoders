"""M9.1 pilot: the bridge-tolerance dry run (m9/LEDGER.md §6).

The six-set bridge is phase 1 of the sole six-set transaction, and a failure there consumes the
access. So the machinery is exercised on DEV instead: freeze a per-query reference for the
bge-small anchor, then re-derive it end to end in a FRESH PROCESS and require zero qid drift and
max |delta nDCG@10| <= 3e-4. What this proves is that the scorer is deterministic across processes
-- GPU top-k merge order, tie-breaking, fp16 memmap reads, tokenizer state -- which is exactly the
class of drift the real bridge exists to catch.

  python m9src/bridge_dryrun.py freeze     # pass 1, writes the reference
  python m9src/bridge_dryrun.py verify     # pass 2, fresh process, applies the tolerance
"""
import json
import sys
import time

import numpy as np
import torch

import m9base
from m9base import RESULTS, WORK

import eval9      # noqa: E402
import guard9     # noqa: E402

REF = WORK / "m9_bridge_ref.json"


class Anchor:
    """bge-small-en-v1.5 served symmetrically -- its own prompt, its own 384-d space. This is the
    release-bar comparator, and the point is reproducibility, not quality."""

    PREFIX = "Represent this sentence for searching relevant passages: "
    REPO = "BAAI/bge-small-en-v1.5"
    REV = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.REPO, revision=self.REV)
        self.m = AutoModel.from_pretrained(self.REPO, revision=self.REV).cuda().eval()

    @torch.inference_mode()
    def encode(self, texts, batch_size=256):
        out = np.empty((len(texts), 384), dtype=np.float32)
        for i in range(0, len(texts), batch_size):
            b = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=512,
                         return_tensors="pt").to("cuda")
            h = self.m(**b).last_hidden_state[:, 0]
            out[i:i + batch_size] = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()
        return out


def score(components):
    """-> {comp: {qid: ndcg}}, anchor queries against anchor-encoded documents."""
    import devsuite
    import evalkit

    a = Anchor()
    out = {}
    for comp in components:
        doc_ids, doc_texts, q_ids, q_texts, qrels = devsuite.load(comp)
        dv = a.encode(list(doc_texts))
        qv = a.encode([Anchor.PREFIX + t for t in q_texts])
        run = evalkit.topk_ids_scores(qv, dv, doc_ids, k=100, chunk=200_000, qids=q_ids)
        out[comp] = {str(k): float(v) for k, v in evalkit.per_query_ndcg(run, qrels).items()}
    return out


# The two small CQADupStack components: 70,492 documents through a 33M encoder is ~1 minute per
# pass, and determinism does not need a bigger corpus to show itself. Fixed here, at lock time.
COMPONENTS = ("cqadup-programmers", "cqadup-physics")


def freeze():
    t0 = time.time()
    per = score(COMPONENTS)
    REF.write_text(json.dumps({"components": list(COMPONENTS), "per_query": per,
                               "seconds": round(time.time() - t0, 1)}))
    print(f"froze {sum(len(v) for v in per.values())} query scores in {time.time()-t0:.0f}s")


def verify():
    tol = guard9.registry()["validation_samples"]["bridge_dryrun"]["max_abs_delta_ndcg"]
    ref = json.loads(REF.read_text())["per_query"]
    t0 = time.time()
    got = score(ref.keys())

    rows, worst = {}, 0.0
    for c in ref:
        miss = sorted(set(ref[c]) - set(got[c]))
        extra = sorted(set(got[c]) - set(ref[c]))
        order = list(ref[c]) == list(got[c])
        d = [abs(ref[c][q] - got[c][q]) for q in ref[c] if q in got[c]]
        m = max(d) if d else float("inf")
        worst = max(worst, m)
        rows[c] = {"n": len(ref[c]), "missing": len(miss), "extra": len(extra),
                   "order_preserved": order, "max_abs_delta": m,
                   "mean_abs_delta": float(np.mean(d)) if d else None}

    out = {"tolerance": tol, "components": rows, "max_abs_delta_overall": worst,
           "qid_drift": sum(r["missing"] + r["extra"] for r in rows.values()),
           "seconds": round(time.time() - t0, 1)}
    out["pass"] = bool(out["qid_drift"] == 0 and worst <= tol
                       and all(r["order_preserved"] for r in rows.values()))
    guard9.begin_run("m9-bridge-dryrun")
    guard9.write_result(RESULTS / "m9_bridge_dryrun.json", out, "m9-bridge-dryrun")
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    {"freeze": freeze, "verify": verify}[sys.argv[1]]()
