"""Build the M7 training mix from the approved sources only (research/m7-data-licensing.md).

Layout under work/train/:
  sources/<name>.json   {"pairs": [{"qid","query","pos":[docid..],"hardneg":[docid..]}],
                         "docstore": <store name>}
  stores/<name>.json    {"ids": [...], "texts": [...]}   (a doc store; hotpotqa reuses the dev corpus)
  querytext/<name>.json ["query text", ...]              (objective-B-only sources)

Held-out dev slice: the mod-50 rule is applied at QUERY granularity
(sha256(f"{source}:{qid}") % 50 == 0) so a held-out query never appears in training —
without that, holding out per pair would leak the query into TRAIN. Logged as an
interpretation in m7/LEDGER.md.

MIRACL was dropped: recovering its 2,863 English train queries' positives requires
downloading the 32.9M-passage miracl-corpus (no parquet mirror), for Wikipedia coverage
that HotpotQA/FEVER/SQuAD/NQ-open already supply. See m7/EXPLORED.md.
"""
import hashlib
import json
import sys
from collections import defaultdict

from datasets import load_dataset

from _paths import WORK
from core import doc_text

TRAIN = WORK / "train"
for sub in ("sources", "stores", "querytext"):
    (TRAIN / sub).mkdir(parents=True, exist_ok=True)

HELDOUT_MOD = 50


def heldout(source, qid):
    return int(hashlib.sha256(f"{source}:{qid}".encode()).hexdigest(), 16) % HELDOUT_MOD == 0


def _save_source(name, pairs, store):
    (TRAIN / "sources" / f"{name}.json").write_text(json.dumps({"pairs": pairs, "docstore": store}))
    nq = len(pairs)
    npos = sum(len(p["pos"]) for p in pairs)
    nhn = sum(len(p.get("hardneg", [])) for p in pairs)
    nho = sum(1 for p in pairs if heldout(name, p["qid"]))
    print(f"  {name}: {nq:,} queries / {npos:,} positives / {nhn:,} hard negs / {nho:,} held-out queries", flush=True)


def _save_store(name, ids, texts):
    (TRAIN / "stores" / f"{name}.json").write_text(json.dumps({"ids": ids, "texts": texts}))
    print(f"  store {name}: {len(ids):,} docs", flush=True)


def beir_train_pairs(ds, name, store_name, store_positives_only):
    """BEIR corpus + train qrels -> pairs. store_positives_only keeps the store to the positives."""
    qrels = defaultdict(dict)
    for r in load_dataset(f"BeIR/{ds}-qrels", split="train"):
        if int(r["score"]) > 0:
            qrels[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
    _q = load_dataset(f"BeIR/{ds}", "queries")["queries"]
    qtext = {str(i): t for i, t in zip(list(_q["_id"]), list(_q["text"]))}
    need = {d for v in qrels.values() for d in v}
    corpus = load_dataset(f"BeIR/{ds}", "corpus")["corpus"]
    ids, texts = [], []
    if store_positives_only:
        for r in corpus:
            if str(r["_id"]) in need:
                ids.append(str(r["_id"]))
                texts.append(doc_text(r))
        _save_store(store_name, ids, texts)
    have = set(ids) if store_positives_only else None
    pairs = [{"qid": q, "query": qtext[q], "pos": [d for d in v if have is None or d in have]}
             for q, v in qrels.items() if q in qtext]
    pairs = [p for p in pairs if p["pos"]]
    _save_source(name, pairs, store_name)


def build_hotpotqa():
    # store = the full BEIR HotpotQA corpus, already encoded for the dev component (reused as the
    # Wikipedia negative bank); positives are ids into it.
    import devsuite
    doc_ids, doc_texts, *_ = devsuite.load("hotpotqa")
    _save_store("hotpotqa-corpus", doc_ids, doc_texts)
    beir_train_pairs("hotpotqa", "hotpotqa-train", "hotpotqa-corpus", store_positives_only=False)


def build_fever():
    beir_train_pairs("fever", "fever-train", "fever-pos", store_positives_only=True)


def build_squad():
    d = load_dataset("rajpurkar/squad", split="train")
    ctx = {}
    pairs = []
    for r in d:
        c = r["context"]
        cid = ctx.setdefault(c, f"squad-{len(ctx)}")
        pairs.append({"qid": r["id"], "query": r["question"], "pos": [cid]})
    _save_store("squad-ctx", list(ctx.values()), list(ctx.keys()))
    _save_source("squad-train", pairs, "squad-ctx")


def build_esci():
    d = load_dataset("tasksource/esci", split="train")
    d = d.filter(lambda b: [x == "us" for x in b["product_locale"]], batched=True, num_proc=8)
    pos, neg, prod = defaultdict(list), defaultdict(list), {}
    qtext = {}
    for r in d.select_columns(["query", "query_id", "product_id", "esci_label", "product_text"]):
        lab = r["esci_label"]
        if lab not in ("Exact", "Irrelevant"):
            continue
        pid, qid = r["product_id"], str(r["query_id"])
        prod.setdefault(pid, r["product_text"])
        qtext[qid] = r["query"].strip()
        (pos if lab == "Exact" else neg)[qid].append(pid)
    _save_store("esci-prod", list(prod.keys()), list(prod.values()))
    pairs = [{"qid": q, "query": qtext[q], "pos": pos[q], "hardneg": neg.get(q, [])}
             for q in pos if qtext[q]]
    _save_source("esci-us", pairs, "esci-prod")


def build_mrtydi():
    url = "hf://datasets/castorini/mr-tydi@refs%2Fconvert%2Fparquet/english/train/0000.parquet"
    d = load_dataset("parquet", data_files={"train": url}, split="train")
    store, pairs = {}, []
    for r in d:
        def add(ps):
            out = []
            for p in ps:
                store.setdefault(p["docid"], f"{p['title']} {p['text']}".strip())
                out.append(p["docid"])
            return out
        p = add(r["positive_passages"])
        if p:
            pairs.append({"qid": str(r["query_id"]), "query": r["query"],
                          "pos": p, "hardneg": add(r["negative_passages"])})
    _save_store("mrtydi-docs", list(store.keys()), list(store.values()))
    _save_source("mrtydi-en", pairs, "mrtydi-docs")


def build_querytext():
    """Objective-B-only sources: query text with no usable positive documents."""
    nq = list(load_dataset("google-research-datasets/nq_open", split="train")["question"])
    (TRAIN / "querytext" / "nqopen.json").write_text(json.dumps(nq))
    print(f"  querytext nqopen: {len(nq):,}", flush=True)
    tq = list(load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")["question"])
    (TRAIN / "querytext" / "triviaqa.json").write_text(json.dumps(tq))
    print(f"  querytext triviaqa: {len(tq):,}", flush=True)


BUILDERS = {"hotpotqa": build_hotpotqa, "fever": build_fever, "squad": build_squad,
            "esci": build_esci, "mrtydi": build_mrtydi, "querytext": build_querytext}

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(BUILDERS)):
        print(f"[{name}]", flush=True)
        BUILDERS[name]()
