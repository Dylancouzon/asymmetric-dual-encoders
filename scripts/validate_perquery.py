"""Executable check: results/perquery.json must agree with results/quality.json per cell.

Exits nonzero on any disagreement outside the two allowlisted potion cells (provenance
note in results/FINAL_MATRIX.md: encode-time fp32 scores vs fp16 vectors at rest).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
q = json.load(open(REPO / "results" / "quality.json"))
p = json.load(open(REPO / "results" / "perquery.json"))

# (system, dataset) -> known encode-time-fp32 vs fp16-at-rest deltas, documented in FINAL_MATRIX.md
ALLOW = {("potion-retrieval-32M", "arguana"), ("potion-retrieval-32M", "trec-covid"),
         ("mdbr-leaf-ir", "arguana"), ("arctic-embed-m-v1.5", "scidocs")}
QUALITY_SLUG = {"lr-dense-pertask": "lightretriever-qwen2.5-1.5b-dense",
                "lr-dense-websearch": "lightretriever-qwen2.5-1.5b-dense-websearch"}

failures = []
for ds, blob in p["datasets"].items():
    for name, vec in blob["systems"].items():
        mean = sum(vec) / len(blob["qids"])
        rec = q[QUALITY_SLUG.get(name, name)][ds]["ndcg@10"]
        if abs(mean - rec) > 5e-5 and (name, ds) not in ALLOW:
            failures.append(f"{name}/{ds}: perquery {mean:.6f} vs quality.json {rec:.6f}")
        elif (name, ds) in ALLOW and abs(mean - rec) > 5e-4:
            failures.append(f"{name}/{ds}: allowlisted cell drifted beyond 5e-4 ({mean:.6f} vs {rec:.6f})")

if failures:
    print("PERQUERY VALIDATION FAILED:\n" + "\n".join(failures))
    sys.exit(1)
n = sum(len(b["systems"]) for b in p["datasets"].values())
print(f"OK: {n} cells validated ({len(ALLOW)} allowlisted, documented in FINAL_MATRIX.md)")
