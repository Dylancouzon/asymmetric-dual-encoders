"""Registered DESCRIPTIVE diagnostic: what is the short-query loss actually made of?

Zero new access. Reads only artifacts that already exist: M7's final-run per-query nDCG on the six
(`results/m7_final_run.json`) and the six's frozen query texts, both already scored and committed.
No reserved, shadow or M9-reserve payload is opened; the LEDGER G2 guard is active and unclaimed,
so an accidental read would raise rather than succeed.

WHY. M7's retention inverts the intuition: best on the LONGEST queries (ArguAna 0.929), worst on
the SHORTEST (trec-covid 0.667, FiQA 0.673). The plan's H3 calls this "short-query loss, partly
recoverable in-class" and routes it to B17. But "best on longest, worst on shortest" is a
BETWEEN-dataset comparison of six points, and at least four things vary along it:

  1. query length -- the hypothesis of interest. A bag-of-token-vectors query averages a fixed
     per-token error; more tokens average more of it away.
  2. subword fragmentation -- WordPiece-30522 predates COVID, so a trec-covid query carries its
     central term split into pieces that carry no lexical identity. This predicts trec-covid as
     the worst set for a reason that has nothing to do with length, and it is the part of the loss
     a multi-word tokenizer (D2) can actually reach.
  3. teacher contamination -- ArguAna and FiQA are on stella's DISCLOSED training list
     (m7/LEDGER.md, Teacher selection). Retention is table/teacher, so an inflated denominator
     DEFLATES retention on exactly two of the six.
  4. n -- trec-covid is 50 queries.

So this computes the WITHIN-dataset relationship, where the dataset fixed effect (and with it the
contamination and task-type confounds) cancels, and reports the between-dataset picture beside it
so the two can be compared rather than conflated.

STATUS: descriptive, exploratory, adopts nothing. It sharpens H3 into separable sub-hypotheses
before B17 spends anything, and it bounds how much of the loss D2's tokenizer can reach. It may
not be cited as evidence for or against any adoption (LEDGER 4.6).
"""
import json
import sys
from pathlib import Path

import numpy as np

import m8base

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_retention_decomposition.json"
SIX = list(m8base.SIX)
# Disclosed on stella's training list -- flagged at the row, per m7/LEDGER.md.
DISCLOSED_OVERLAP = {"arguana", "fiqa"}


def _query_texts():
    """The six's frozen query payloads. These are the DEVELOPMENT-INFORMED six, not the reserved
    four; opening them is not an access (they were scored in M7's final run)."""
    out = {}
    for ds in SIX:
        p = RESULTS / "frozen_eval" / f"{ds}.json"
        d = json.loads(p.read_text())
        if "queries" in d:
            out[ds] = dict(d["queries"])
        else:                                        # {qid: {"text": ...}} or parallel lists
            qs = d.get("qids") or d.get("query_ids")
            ts = d.get("qtexts") or d.get("query_texts") or d.get("texts")
            out[ds] = dict(zip(qs, ts)) if qs and ts else {}
    return out


def _tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        "NovaSearch/stella_en_400M_v5",
        revision="ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20")


def _ols(x, y):
    """Slope, intercept, and the slope's t-stat. Plain OLS -- this is a descriptive summary of a
    scatter, not an inferential claim, and is labelled as such at every use."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 5 or x.std() == 0:
        return None
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    se = float(np.sqrt((resid @ resid) / (len(x) - 2) / ((x - x.mean()) @ (x - x.mean()))))
    return {"slope": float(b), "intercept": float(a), "se": se,
            "t": float(b / se) if se else None, "n": int(len(x))}


def main():
    fr = json.loads((RESULTS / "m7_final_run.json").read_text())["six"]
    table, teacher = fr["int8-table"], fr["teacher-symmetric"]
    texts = _query_texts()
    tok = _tokenizer()

    rows, per_ds = [], {}
    for ds in SIX:
        qids = sorted(set(table[ds]) & set(teacher[ds]) & set(texts.get(ds, {})))
        if not qids:
            per_ds[ds] = {"error": "no query texts recovered from the frozen payload"}
            continue
        recs = []
        for q in qids:
            t = texts[ds][q]
            words = t.split()
            n_sub = len(tok(t, add_special_tokens=False)["input_ids"])
            recs.append({
                "ds": ds, "qid": q,
                "words": len(words),
                "subwords": n_sub,
                "frag": n_sub / max(1, len(words)),     # subwords per whitespace word
                "table": float(table[ds][q]),
                "teacher": float(teacher[ds][q]),
                "gap": float(teacher[ds][q] - table[ds][q]),
            })
        rows += recs
        w = np.array([r["words"] for r in recs])
        f = np.array([r["frag"] for r in recs])
        g = np.array([r["gap"] for r in recs])
        tb = np.array([r["table"] for r in recs])
        te = np.array([r["teacher"] for r in recs])
        per_ds[ds] = {
            "n": len(recs),
            "median_words": float(np.median(w)), "mean_frag": float(f.mean()),
            "macro_table": float(tb.mean()), "macro_teacher": float(te.mean()),
            "retention_macro": float(tb.mean() / te.mean()) if te.mean() else None,
            "mean_gap": float(g.mean()),
            "within_slope_gap_on_words": _ols(w, g),
            "within_slope_gap_on_frag": _ols(f, g),
            "teacher_disclosed_overlap": ds in DISCLOSED_OVERLAP,
        }

    # Within-dataset pooled: centre everything inside each dataset, so the dataset fixed effect --
    # and with it the contamination and task-type confounds -- drops out.
    #
    # THE DIFFICULTY CONTROL. A raw gap-on-length slope is confounded: if long queries are simply
    # harder, the TEACHER loses on them too and the gap widens for a reason that says nothing
    # about the table. So the teacher's own slope is reported beside the table's, and the
    # difference between them is the only quantity that is about the query encoder.
    pooled = {}
    for key in ("words", "frag"):
        acc = {"gap": ([], []), "table": ([], []), "teacher": ([], [])}
        for ds in SIX:
            r = [x for x in rows if x["ds"] == ds]
            if len(r) < 8:
                continue
            x = np.array([v[key] for v in r], float)
            xc = x - x.mean()
            for tgt in acc:
                y = np.array([v[tgt] for v in r], float)
                acc[tgt][0].append(xc)
                acc[tgt][1].append(y - y.mean())
        for tgt, (xs, ys) in acc.items():
            if xs:
                pooled[f"within_dataset_{tgt}_on_{key}"] = _ols(np.concatenate(xs),
                                                                np.concatenate(ys))
    # Collinearity: within dataset, are length and fragmentation the same variable? Reported
    # POOLED and PER DATASET, because a pooled correlation can be dominated by one dataset.
    xs, ys, per_r = [], [], {}
    for ds in SIX:
        r = [x for x in rows if x["ds"] == ds]
        if len(r) < 8:
            continue
        a = np.array([v["words"] for v in r], float)
        b = np.array([v["frag"] for v in r], float)
        per_r[ds] = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None
        xs.append(a - a.mean())
        ys.append(b - b.mean())
    if xs:
        a, b = np.concatenate(xs), np.concatenate(ys)
        pooled["within_dataset_corr_words_frag"] = float(np.corrcoef(a, b)[0, 1])
        pooled["within_dataset_corr_words_frag_per_dataset"] = per_r

    # WHO DRIVES THE POOLED SLOPE. A pooled within-dataset OLS weights each dataset by its own
    # variance in x. ArguAna's queries average 174 words against 2-12 for the other five, so it
    # can carry essentially all the LENGTH variance and the "pooled within-dataset" read becomes
    # the one-dataset read the diagnostic was built to escape -- and that dataset is one of the
    # teacher's two DISCLOSED training sets. This reports the share explicitly and recomputes the
    # slopes with ArguAna removed. (2026-08-29 review finding; it changed the conclusion.)
    variance_share, leave_one_out = {}, {}
    for key in ("words", "frag"):
        var = {}
        for ds in SIX:
            r = [x for x in rows if x["ds"] == ds]
            if len(r) < 8:
                continue
            x = np.array([v[key] for v in r], float)
            var[ds] = float(((x - x.mean()) ** 2).sum())
        tot = sum(var.values()) or 1.0
        variance_share[key] = {d: v / tot for d, v in var.items()}
        for drop in ("arguana",):
            xs2, ys2 = [], []
            for ds in SIX:
                if ds == drop:
                    continue
                r = [x for x in rows if x["ds"] == ds]
                if len(r) < 8:
                    continue
                x = np.array([v[key] for v in r], float)
                y = np.array([v["gap"] for v in r], float)
                xs2.append(x - x.mean())
                ys2.append(y - y.mean())
            if xs2:
                leave_one_out[f"gap_on_{key}_excluding_{drop}"] = _ols(
                    np.concatenate(xs2), np.concatenate(ys2))
    pooled["variance_share_of_x_by_dataset"] = variance_share
    pooled["leave_one_out"] = leave_one_out

    # The between-dataset picture the plan currently reads, stated beside it at n = 6.
    ds_ok = [d for d in SIX if "error" not in per_ds[d]]
    between = {
        "gap_on_median_words": _ols([per_ds[d]["median_words"] for d in ds_ok],
                                    [per_ds[d]["mean_gap"] for d in ds_ok]),
        "gap_on_mean_frag": _ols([per_ds[d]["mean_frag"] for d in ds_ok],
                                 [per_ds[d]["mean_gap"] for d in ds_ok]),
        "_note": "n = 6 datasets. Reported so the between- and within-dataset readings can be "
                 "compared; six points cannot separate length from task type or contamination.",
    }

    # Length strata WITHIN dataset -- the read the plan's 'best on longest / worst on shortest'
    # sentence actually needs.
    strata = {}
    for ds in ds_ok:
        r = [x for x in rows if x["ds"] == ds]
        w = np.array([v["words"] for v in r])
        qs = np.quantile(w, [0.25, 0.5, 0.75])
        buckets = {}
        edges = [(-1, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], 1e9)]
        for i, (lo, hi) in enumerate(edges):
            sel = [v for v in r if lo < v["words"] <= hi]
            if len(sel) < 5:
                continue
            tb = np.mean([v["table"] for v in sel])
            te = np.mean([v["teacher"] for v in sel])
            buckets[f"Q{i+1}"] = {"n": len(sel),
                                  "words_range": [float(max(0, lo)), float(min(hi, max(w)))],
                                  "table": float(tb), "teacher": float(te),
                                  "retention": float(tb / te) if te else None}
        strata[ds] = buckets

    out = {
        "_note": __doc__.strip().splitlines()[0],
        "status": "DESCRIPTIVE / EXPLORATORY. Adopts nothing; may not be cited for an adoption "
                  "(LEDGER 4.6). No new evaluation access: every input was already scored.",
        "source": {"per_query": "results/m7_final_run.json six.int8-table / six.teacher-symmetric",
                   "texts": "results/frozen_eval/<six>.json",
                   "tokenizer": "NovaSearch/stella_en_400M_v5 @ ffeb2b7"},
        "per_dataset": per_ds,
        "pooled_within_dataset": pooled,
        "between_dataset": between,
        "length_strata_within_dataset": strata,
        "caveats": [
            "Retention is table/teacher; ArguAna and FiQA are on stella's DISCLOSED training "
            "list, so their denominators are inflated and their retention is deflated.",
            "trec-covid is n=50: its dataset-level row is the least resolved of the six.",
            "'gap' is teacher minus table per query, not a ratio -- ratios are undefined where "
            "the teacher scores 0 and are unstable where it scores near 0.",
            "OLS slopes are descriptive summaries of a scatter, not inferential claims.",
        ],
    }
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({"pooled_within_dataset": pooled, "between_dataset": between,
                      "per_dataset": {d: {k: per_ds[d].get(k) for k in
                                          ("n", "median_words", "mean_frag", "retention_macro",
                                           "mean_gap", "teacher_disclosed_overlap")}
                                      for d in ds_ok}}, indent=2, default=float))
    print(f"\nwrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
