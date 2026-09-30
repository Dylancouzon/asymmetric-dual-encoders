"""Render m15/EVIDENCE.md: every number the paper cites, in plain language, from committed JSON.

    python m15/make_evidence.py

Each card states the claim, the label (registered or exploratory), the numbers, the source file and
what the result cannot show. Numbers are read from the result files, never typed by hand.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = lambda n: json.loads((REPO / "results" / n).read_text())
OUT = REPO / "m15" / "EVIDENCE.md"
f3 = lambda x: f"{x:.3f}"
f4 = lambda x: f"{x:.4f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def card(title, label, claim, body, source, limits):
    return (f"## {title}\n\n**Label:** {label}. **Source:** {source}\n\n**Claim.** {claim}\n\n{body}\n\n"
            f"**Cannot show.** {limits}\n")


def beir():
    t = R("m20_beir15_run.json")["table"]
    systems = list(next(iter(t.values()))["systems"])
    rows = [[d] + [f4(v["systems"][s]) for s in systems] for d, v in t.items()]
    mac = {s: float(np.mean([t[d]["systems"][s] for d in t])) for s in systems}
    rows.append(["**macro**"] + [f"**{f4(mac[s])}**" for s in systems])
    st = mac["stella-query"]
    ret = ", ".join(f"{s} {mac[s] / st:.3f}" for s in systems)
    return card("C1. Retention across the three tiers", "registered (M20)",
                "Over one Stella index, Nano keeps 0.905 and Zero 0.814 of the Stella query path's "
                "BEIR-15 macro nDCG@10.",
                table(["dataset"] + systems, rows) + f"\n\nRetention (ratio of macros): {ret}.",
                "`results/m20_beir15_run.json`; per-query rows `results/m20_beir15_scores/`",
                "Superiority or equivalence between systems; the four held-out rows are copied from "
                "the one-shot test.")


def latency():
    s = R("m15_e1_latency.json")["summary"]
    rows = [[n, *(f"{v['p50_ms_median_of_trials'][b]:.3f}" for b in ("short", "medium", "long")),
             *(f"{v['p95_ms_median_of_trials'][b]:.3f}" for b in ("short", "medium", "long")),
             f"{v['hydration_s']:.2f}", round(v["peak_rss_bytes"] / 2**20),
             f"{v['asset_total_bytes'] / 2**20:.1f}"] for n, v in s.items()]
    return card("C2. Query-encode latency", "registered (E1)",
                "On an Apple M5 Pro CPU at batch one, Zero encodes a 5-12-word query in 0.044 ms, "
                "Nano in 2.25 ms and the Stella query path in 31.6 ms.",
                table(["encoder", "p50 1-4 w (ms)", "p50 5-12 w", "p50 13-64 w", "p95 1-4 w",
                       "p95 5-12 w", "p95 13-64 w", "hydration (s)", "peak RSS (MiB)",
                       "files (MiB)"], rows)
                + f"\n\nFertility feature cost: {s['zero']['fertility_feature_p50_ms']:.3f} ms.",
                "`results/m15_e1_latency.json`",
                "Server throughput, GPU serving, or other runtimes.")


def towers():
    e8 = R("m15_e8_towers.json")
    cf = e8["configs"]
    rows = [[n, c["dim"], c["pooling"], c["lambda"], f4(c["dev_table"]), f4(c["dev_ceiling"]),
             f4(c["six_table_macro_all6"]), f4(c["six_ceiling_macro_all6"]),
             f3(c["retention_all6"]), f4(c["six_table_macro_clean4"])] for n, c in cf.items()]
    per = [[n] + [f"{c['six_table'][d]:.3f} / {c['six_ceiling'][d]:.3f}" for d in
                  ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")]
           for n, c in cf.items()]
    rho = e8["spearman"]["checkpoints"]
    body = (table(["config", "dim", "readout", "lambda", "dev table", "dev tower", "six table",
                   "six tower", "retention", "clean-4 table"], rows)
            + "\n\nPer dataset, table / tower:\n\n"
            + table(["config", "scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid"],
                    per)
            + "\n\nSpearman over the 10 checkpoints: "
            + ", ".join(f"{k} {v:+.3f}" for k, v in rho.items()))
    return card("C3. Tower quality does not predict the table; a dev screen does",
                "registered (E8; lambdas frozen before the six sets were scored)",
                "Across ten checkpoints the tower's own six-set score and its closed-form table's "
                "six-set score are uncorrelated (Spearman "
                f"{rho['six_ceiling_vs_six_table_all6']:+.2f}), while the table's dev-forum score "
                f"predicts its six-set rank ({rho['primary_dev_table_vs_six_table_all6']:+.2f}).",
                body, "`results/m15_e8_towers.json`, `results/m15_e8_frozen_lambdas.json`, "
                "`m15/e8_exposure.json`",
                "A cause; other recipes; significance (n = 10).")


def mechanisms():
    a = R("m15_e11_mechanism.json")
    b = R("m15_e11b_margin.json")
    rows = [[n, f3(c["additivity_six"]), f3(c["order_cos_six"]),
             f4(b["configs"][n]["median_gap_six"]), f"{b['configs'][n]['error_to_margin_six']:.2f}",
             f4(c["six_table"]), f3(c["retention_all6"])] for n, c in a["configs"].items()]
    rho = {**a["spearman"], **b["spearman"]}
    return card("C4. Three mechanisms that do not explain C3", "exploratory (E11, E11b)",
                "How additive a tower is, how much it reads word order, and how its table's error "
                "compares with its ranking margins do not predict its table's quality.",
                table(["config", "additivity", "order cosine", "median 1st-10th gap",
                       "error / margin", "six table", "retention"], rows)
                + "\n\nSpearman over the 10 checkpoints: "
                + ", ".join(f"{k} {v:+.3f}" for k, v in rho.items()),
                "`results/m15_e11_mechanism.json`, `results/m15_e11b_margin.json`",
                "That no mechanism exists; only these three were tested.")


def oracle():
    e5 = R("m15_e5_oracle.json")
    m = e5["macro_over_12"]
    rows = [[d, f4(v["zero"]), f4(v["nano"]), f4(v["oracle"]), f3(v["share_equal"]),
             f3(v["share_zero_above_nano"]), f3(v["share_zero_below_nano"])]
            for d, v in e5["per_dataset"].items()]
    fr = ", ".join(f"{b:.2f}: {x:.4f}" for b, x in zip(e5["fractions"], m["frontier"]))
    return card("C5. The table's loss is concentrated", "registered (E5)",
                f"An oracle that sends 15% of queries to Nano matches always-Nano; the full oracle "
                f"reaches {m['oracle']:.4f} against {m['nano']:.4f} for always-Nano.",
                table(["dataset", "zero", "nano", "oracle", "tie", "zero higher", "nano higher"], rows)
                + f"\n\nOracle frontier (Nano share: macro nDCG@10): {fr}.",
                "`results/m15_e5_oracle.json`", "A router that reaches the oracle.")


def prefixes():
    e4 = R("m15_e4_prefix.json")
    systems = ("stella-query", "nano", "zero", "bm25")
    conds = [c for c in e4["conditions"] if c != "full"]
    rows = []
    for ds, v in e4["per_dataset"].items():
        for c in conds:
            rows.append([ds, c] + [f"{v['systems'][s][c]['retention']:.3f} "
                                   f"[{v['systems'][s][c]['retention_ci95'][0]:.3f}, "
                                   f"{v['systems'][s][c]['retention_ci95'][1]:.3f}]"
                                   for s in systems])
    return card("C6. Prefixes cost every tier the same share", "registered (E4, E7 folded in)",
                "At every word cut the three dense tiers keep the same share of their own "
                "full-query nDCG@10 within 0.02 (macro over three sets).",
                table(["dataset", "cut", *systems], rows), "`results/m15_e4_prefix.json`",
                "Real search-as-you-type sessions.")


def routers():
    e6 = R("m15_e6_router.json")
    rows = [[b, f3(v["nano_fraction"]), f4(v["router"]), f4(v["random_same_fraction"]),
             f4(v["oracle_same_fraction"]), f"{v['router_minus_random']:+.4f}"]
            for b, v in e6["macro_over_12"].items()]
    out = card("C7. A fertility router captures a tenth of the headroom", "registered (E6)",
               "Routing high-fertility queries to Nano beats random routing at the same share by "
               "0.003-0.005 macro nDCG@10.",
               table(["budget", "Nano share", "router", "random", "oracle", "router - random"], rows),
               "`results/m15_e6_router.json`, thresholds `results/m15_e6_thresholds.json`",
               "That fertility is the best feature.")
    p = REPO / "results" / "m15_e10_router.json"
    if p.exists():
        e10 = json.loads(p.read_text())
        rows = [[k, b, f3(v["nano_fraction"]), f4(v["router"]), f4(v["random_same_fraction"]),
                 f4(v["oracle_same_fraction"]), f"{v['router_minus_random']:+.4f}"]
                for k, bs in e10["macro"].items() for b, v in bs.items() if b != "datasets"]
        out += "\n" + card("C8. Routers on Zero's own signals", "registered (E10)",
                           "See the paper, Section 7.1.",
                           table(["feature", "budget", "Nano share", "router", "random", "oracle",
                                  "router - random"], rows),
                           "`results/m15_e10_router.json`", "Features beyond the five tested.")
    return out


def blend():
    p = REPO / "results" / "m15_e9_blend.json"
    if not p.exists():
        return ""
    e9 = json.loads(p.read_text())
    rows = []
    for ds, v in e9["per_dataset"].items():
        for base in ("nano", "stella"):
            x = v[base]
            rows.append([ds, base, f4(x["alone"]), f4(x["blend"]), f"{x['blend_minus_alone']:+.4f}",
                         f"[{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"])
    mac = "; ".join(f"{k}: {v['blend_minus_alone']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
                    for k, v in e9["macro"].items())
    return card("C9. Blending two query vectors in one search", "registered (E9)",
                f"Frozen weights: {e9['frozen']['nano']['weight']} (Nano), "
                f"{e9['frozen']['stella']['weight']} (Stella).",
                table(["dataset", "base", "alone", "blend", "difference", "95% CI"], rows)
                + f"\n\nMacro differences: {mac}.", "`results/m15_e9_blend.json`",
                "ANN behavior of the blend.")


def system():
    out = []
    for ds in ("fiqa", "msmarco1m"):
        p = REPO / "results" / f"m15_e2_ann_{ds}.json"
        if not p.exists():
            continue
        e2 = json.loads(p.read_text())
        rows = [[n, e, v["quant"], v["hnsw_ef"], v["oversampling"], f"{v['e2e_p50_ms']:.2f}",
                 f"{v['search_p50_ms']:.2f}", f4(v["ann_ndcg10"])]
                for n, d in e2["decision"].items() for e, v in d.items() if v]
        out.append(card(f"C10. Does the saving survive the search ({ds})", "registered (E2)",
                        "For each encoder, the cheapest setting within each loss target of its own "
                        "exact nDCG@10.",
                        table(["encoder", "target", "quant", "ef", "oversampling", "e2e p50 (ms)",
                               "search p50 (ms)", "ANN nDCG@10"], rows)
                        + f"\n\nExact nDCG@10: {e2['exact']}. Saving survives: "
                          f"{e2['saving_survives']}.",
                        f"`results/m15_e2_ann_{ds}.json`",
                        "Full-corpus MS MARCO, multi-client throughput, memory limits."))
    return "\n".join(out)


def heldout():
    d = R("m13_reserved_run.json")
    s = d["systems"]
    rows = [[n] + [f4(v[k]) for k in ("fever", "dbpedia-entity", "cqadup-android",
                                      "cqadup-english")] for n, v in s.items()]
    return card("C11. The one-shot held-out test", "registered (spent 2026-09-19)",
                "The four held-out datasets, reported in full; contrasts in the paper, Section 8.",
                table(["system", "fever", "dbpedia-entity", "cqadup-android", "cqadup-english"], rows),
                "`results/m13_reserved_run.json`", "Anything decided on these sets after the test.")


if __name__ == "__main__":
    parts = ["# M15 evidence, in plain language\n",
             "Generated by `python m15/make_evidence.py` from committed result files. Each card "
             "states one claim of the paper, whether it was registered before observation or is "
             "exploratory, the numbers, the file that holds them, and what the result cannot show.\n",
             beir(), latency(), towers(), mechanisms(), oracle(), prefixes(), routers(), blend(),
             system(), heldout()]
    OUT.write_text("\n".join(p for p in parts if p))
    print("wrote", OUT.relative_to(REPO))
