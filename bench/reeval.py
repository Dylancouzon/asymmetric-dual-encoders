"""Re-score every cached (model, dataset) vector pair, including composed asym configs."""
import run_asym
from core import ARTIFACTS, DATASETS, evaluate, load_beir, load_vecs, record

qrels_cache = {ds: load_beir(ds)[4] for ds in DATASETS}
for model_dir in sorted(ARTIFACTS.iterdir()):
    slug = model_dir.name
    for ds in DATASETS:
        doc_ids, doc_vecs = load_vecs(slug, ds, "doc")
        q_ids, q_vecs = load_vecs(slug, ds, "query")
        if doc_vecs is None or q_vecs is None or doc_vecs.shape[1] != q_vecs.shape[1]:
            continue
        record(slug, ds, evaluate(doc_ids, doc_vecs, q_ids, q_vecs, qrels_cache[ds]))
run_asym.main()
