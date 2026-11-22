"""One real NQ search for the talk's slide 2 (exploratory illustration).

Encodes one NQ test query with Stella's query encoder (with its prompt), runs exact search over the committed
Stella NQ document vectors (work/m15/vecs/nq_stella), and records the first dimensions of the query vector and of
three passage vectors (the judged answer and two other passages shown on the slide), plus the answer's rank.

Run: .venv/bin/python m15src/talk_slide2_example.py   -> results/m15_talk_slide2_example.json
"""
import json

import numpy as np

import encoders15
from common import REPO, load_public, receipt, utc_now, write_result

QUERY = "who played joey's agent estelle on friends"
SHOWN = ("Catch-22", "Mitral valve")          # two other passages drawn on the slide, by title prefix
VECS = REPO / "work/m15/vecs/nq_stella"
OUT = REPO / "results/m15_talk_slide2_example.json"
DIMS = 6


def main():
    started = utc_now()
    data = load_public("nq")
    qid = data["q_ids"][data["q_texts"].index(QUERY)]
    answer = next(iter(data["qrels"][qid]))
    ids = json.loads((VECS / "doc_ids.json").read_text())
    q = encoders15.make("stella-query").encode([QUERY])[0].astype(np.float32)
    scores, off = np.empty(len(ids), np.float32), 0
    for p in sorted(VECS.glob("shard_*.npy")):
        a = np.load(p, mmap_mode="r")
        scores[off:off + len(a)] = np.asarray(a, np.float32) @ q
        off += len(a)
    assert off == len(ids)
    top = np.argsort(-scores)[:3]
    pos = {d: i for i, d in enumerate(ids)}
    text = {d: t for d, t in zip(data["doc_ids"], data["doc_texts"])}
    shown = [answer] + [next(d for d in data["doc_ids"] if text[d].startswith(s + " ")) for s in SHOWN]

    def vec(d):
        i = pos[d]
        for p in sorted(VECS.glob("shard_*.npy")):
            a = np.load(p, mmap_mode="r")
            if i < len(a):
                return [round(float(x), 3) for x in a[i][:DIMS]]
            i -= len(a)

    out = {"query": QUERY, "query_id": qid, "query_vector": [round(float(x), 3) for x in q[:DIMS]],
           "answer_id": answer, "answer_rank": int((scores > scores[pos[answer]]).sum()) + 1,
           "top3": [{"id": ids[i], "score": float(scores[i]), "text": text[ids[i]][:120]} for i in top],
           "passages": [{"id": d, "text": text[d][:120], "vector": vec(d), "score": float(scores[pos[d]])} for d in shown],
           "n_docs": len(ids), "dims_shown": DIMS}
    print(json.dumps({k: out[k] for k in ("query_vector", "answer_rank", "top3")}, indent=1))
    write_result(OUT, {"status": "COMPLETE", "analysis": "exploratory (illustration)", "dataset": "nq", "results": out,
                       "pins": data["pins"], "receipt": receipt(__file__, {}, started, ("numpy",))})


if __name__ == "__main__":
    main()
