"""Do the dev components this machine built match the pinned ones? Refuse the probe if not.

The teacher-learnability probe scores every candidate on cqadup-programmers and cqadup-physics and
is compared against a stella row produced on another machine. If HF serves a different corpus,
query set or qrels here, the two rows are not comparable and nothing in the probe path would say
so -- `devsuite.load` caches whatever it downloads and no hash is checked on the way in.

    PYTHONPATH=m7src python scripts/verify_dev_hashes.py cqadup-programmers cqadup-physics
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m7src"))

import devsuite
from _paths import REPO

FIELDS = ("n_docs", "n_queries", "corpus_ids_sha256", "corpus_text_sha256", "qids_sha256",
          "qrels_sha256")

pinned = json.loads((REPO / "results" / "m7_dev_manifest.json").read_text())
bad = []
for c in sys.argv[1:] or devsuite.COMPONENTS:
    got, want = devsuite.manifest_entry(c), pinned[c]
    diff = {f: (want[f], got[f]) for f in FIELDS if want[f] != got[f]}
    print(f"{c:20s} {got['n_docs']:>8,} docs {got['n_queries']:>6,} queries  "
          f"{'MATCHES the pin' if not diff else 'MISMATCH ' + str(diff)}", flush=True)
    if diff:
        bad.append(c)
if bad:
    raise SystemExit(f"dev components differ from results/m7_dev_manifest.json: {bad}. A probe "
                     "scored on different content is not comparable to the committed rows.")
print("dev components match the pin")
