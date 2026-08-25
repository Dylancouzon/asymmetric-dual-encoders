"""Stage 0.2 -- capacity probe. DIAGNOSTIC ONLY, categorically ineligible for any gate.

Deliberately overfits a lookup table on the dev queries with the contrastive objective, trained
to loss plateau on a logged budget. Train and eval queries are the SAME by design: this asks
whether the architecture can express good retrieval in this frozen doc space at all, not
whether it generalizes.

Falsifiable bar (pre-registered): the overfit table must CI-resolve above the BM25 dev row.
If unlimited overfitting on dev cannot beat BM25 on dev, the frozen-tower tax is structural
and the negative-result path is earned.
"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import boot
import dev_eval
import devsuite
from _paths import REPO, WORK
from table import NO_PREFIX, WITH_PREFIX, QueryTable, get_tokenizer, ragged, tokenize
from init_table import get_init

OUT = WORK / "probe"
OUT.mkdir(parents=True, exist_ok=True)
COMPONENTS = ("nq-250k", "cqadup-programmers", "cqadup-physics")


def probe_component(comp, pre, steps=6000, batch=256, n_neg=32768, temp=0.02, lr=1e-2,
                    init="teacher", log=print, plateau_patience=8, plateau_tol=1e-3):
    doc_ids, doc_texts, q_ids, q_texts, qrels, dv = dev_eval.doc_vecs(comp)
    pos_of = {}
    id_to_i = {d: i for i, d in enumerate(doc_ids)}
    for qi, qid in enumerate(q_ids):
        ps = [id_to_i[d] for d, s in qrels.get(qid, {}).items() if s > 0 and d in id_to_i]
        if ps:
            pos_of[qi] = ps
    qi_list = sorted(pos_of)
    tok = get_tokenizer()
    ids_all = tokenize(tok, q_texts, pre)
    V = tok.vocab_size

    dvt = torch.from_numpy(np.ascontiguousarray(dv)).cuda()   # fp16, the whole component corpus
    model = QueryTable(get_init(init, pre, vocab=V), learned_weights=True).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    losses, best, bad, t0 = [], float("inf"), 0, time.time()
    used = 0
    for s in range(1, steps + 1):
        pick = [qi_list[i] for i in rng.integers(0, len(qi_list), batch)]
        f, o, l = ragged([ids_all[i] for i in pick], "cuda")
        qv = model(f, o, l)
        p_i = torch.tensor([pos_of[i][rng.integers(len(pos_of[i]))] for i in pick], device="cuda")
        neg_i = torch.from_numpy(rng.integers(0, len(dv), min(n_neg, len(dv)))).cuda()
        pos_v = dvt.index_select(0, p_i).float()
        neg_v = dvt.index_select(0, neg_i).float()
        s_pos = (qv * pos_v).sum(1, keepdim=True) / temp
        s_neg = (qv @ neg_v.T) / temp
        # a sampled negative that IS this query's positive would be a spurious hard negative
        same = neg_i.unsqueeze(0) == p_i.unsqueeze(1)
        s_neg = s_neg.masked_fill(same, float("-inf"))
        loss = F.cross_entropy(torch.cat([s_pos, s_neg], 1),
                               torch.zeros(len(pick), dtype=torch.long, device="cuda"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        used = s
        if s % 500 == 0:
            recent = float(np.mean(losses[-200:]))
            log(f"    [{comp}] step {s}/{steps} loss {recent:.4f} ({time.time()-t0:.0f}s)")
            if recent > best - plateau_tol:
                bad += 1
                if bad >= plateau_patience:
                    log(f"    [{comp}] loss plateau at step {s}; stopping")
                    break
            else:
                bad, best = 0, recent
    del dvt
    torch.cuda.empty_cache()
    model.eval()
    pq = dev_eval.eval_query_vecs(comp, model.encode(q_texts, pre, tok=tok))
    return pq, {"steps_used": used, "final_loss": float(np.mean(losses[-200:])),
                "budget_steps": steps, "batch": batch, "n_neg": n_neg, "temp": temp, "lr": lr,
                "seconds": round(time.time() - t0, 1), "n_queries_trained": len(qi_list)}


def main(pre_name="noprefix", steps=6000):
    pre = {"noprefix": NO_PREFIX, "prefix": WITH_PREFIX}[pre_name]
    refs = json.loads((dev_eval.DEVRES / "refs.json").read_text())
    per, budgets = {}, {}
    for c in COMPONENTS:
        pq, b = probe_component(c, pre, steps=steps)
        per[c] = pq
        budgets[c] = b
        print(f"  probe {c}: {np.mean(list(pq.values())):.4f}  (bm25 "
              f"{np.mean(list(refs['bm25'][c].values())):.4f})", flush=True)
    m, means = dev_eval.report(per, "[probe] overfit table")
    bm = {c: refs["bm25"][c] for c in COMPONENTS}
    r = boot.paired(per, bm, alternative="greater")
    passed = bool(r["ci95"][0] > 0)
    out = {"_note": "DIAGNOSTIC ONLY -- categorically ineligible for any gate (instructions-m7.md). "
                    "Train and eval queries are identical by design.",
           "preproc": pre_name, "macro": m, "per_component": means, "budgets": budgets,
           "vs_bm25_dev": r, "bar": "overfit table must CI-resolve above the BM25 dev row",
           "passed": passed}
    (REPO / "results" / f"m7_capacity_probe_{pre_name}.json").write_text(json.dumps(out, indent=1))
    print(f"\nprobe macro {m:.4f} vs bm25 macro "
          f"{np.mean([np.mean(list(bm[c].values())) for c in COMPONENTS]):.4f}: "
          f"d={r['delta']:+.4f} CI={r['ci95']} p={r['p_str']} -> {'PASS' if passed else 'FAIL'}", flush=True)
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "noprefix",
         int(sys.argv[2]) if len(sys.argv) > 2 else 6000)
