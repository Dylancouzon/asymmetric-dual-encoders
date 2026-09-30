import sys
sys.path.insert(0, "m15src")
from common import load_public
import vectors15 as V
from pathlib import Path
for ds in sys.argv[1:]:
    data = load_public(ds)
    out = Path("work/m15/vecs") / ds
    V.encode_docs(data["doc_texts"], out, device="cuda", shard=20_000, batch_tokens=32768)
    print(ds, "encoded", len(data["doc_ids"]), flush=True)
