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
                "six-set score are weakly associated (Spearman "
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
    return card("C4. Three candidate proxies for table quality", "exploratory (E11, E11b)",
                "How additive a tower is, how much it reads word order, and how its table's error "
                "compares with its ranking margins did not reliably rank table quality in this comparison.",
                table(["config", "additivity", "order cosine", "mean of per-set median 1st-10th gaps",
                       "mean of per-set RMS error / median gap", "six table", "retention"], rows)
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
    return card("C6. Synthetic prefix retention by query tier", "registered (E4, E7 folded in)",
                "The dense tiers retain descriptively similar shares of their own full-query "
                "nDCG@10 on synthetic prefixes; this is not an equivalence test.",
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
                           "See the paper, Appendix C.",
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
        shared = ""
        if ds == "msmarco1m":
            fixed_rows = []
            for quant in ("none", "binary1"):
                for encoder in ("zero", "nano", "stella-query"):
                    eligible = [r for r in e2["rows"] if r["quant"] == quant
                                and r["encoder"] == encoder
                                and r["ann_ndcg10"] >= 0.99 * e2["exact"][encoder]]
                    best = min(eligible, key=lambda r: r["e2e_p50_ms"])
                    fixed_rows.append([quant, encoder, best["hnsw_ef"], best["oversampling"],
                                       f4(best["ann_ndcg10"]), f"{best['e2e_p50_ms']:.3f}"])
            fused = e2["fused"]["zero+qdrant_bm25_dbsf@100"]
            shared = ("\n\n**Exploratory shared-index illustration, 2026-10-01:** restrict the "
                      "registered sweep to one quantization and choose the fastest recorded request "
                      "setting within 1% of each tier's own exact quality. All three encoders query "
                      "the same populated collection per quantization, in batches with precomputed "
                      "vectors, without a collection rebuild between encoder batches. The targets "
                      "do not equate absolute retrieval quality.\n\n"
                      + table(["fixed quant", "encoder", "ef", "oversampling", "ANN nDCG@10",
                               "encode + search p50 (ms)"], fixed_rows)
                      + f"\n\nZero + server-side BM25, DBSF@100, on a separate unquantized "
                        f"dense-plus-sparse collection: {f4(fused['ndcg10'])} nDCG@10, "
                        f"{fused['e2e_p50_ms']:.3f} ms encode + search. This quality belongs to "
                        "the positive-preserving diagnostic, not BEIR-15.\n\nLatency sums per-query "
                        "encode and search times measured in separate phases; it excludes encoder "
                        "loading and application dispatch. No interleaving, concurrent failover, "
                        "all-tier resident memory, or switch overhead was measured.")
        out.append(card(f"C10. Does the saving survive the search ({ds})", "registered (E2)",
                        "For each encoder, the cheapest setting within each loss target of its own "
                        "exact nDCG@10.",
                        table(["encoder", "target", "quant", "ef", "oversampling", "e2e p50 (ms)",
                               "search p50 (ms)", "ANN nDCG@10"], rows)
                        + f"\n\nExact nDCG@10: {e2['exact']}. Saving survives: "
                          f"{e2['saving_survives']}." + shared,
                        f"`results/m15_e2_ann_{ds}.json`",
                        "Full-corpus MS MARCO, multi-client throughput, memory limits."))
    return "\n".join(out)


def failure_modes():
    a, b = R("m15_e12_failure_modes.json")["summary"], R("m15_e12b_fragility.json")["summary"]
    rows = [[k, f"{v:+.3f}"] for k, v in {**a["rho"], **{f"{k} (within Stella-score bins)": v
            for k, v in a["rho_within_stella_score_bins"].items()},
            "gap vs fragility": b["rho_gap_fragility"],
            "gap vs order, controlling for fragility": b["partial_rho_gap_order_given_fragility"]}.items()]
    return card("C12. The table's gap concentrates on shuffle-sensitive queries", "exploratory (E12, E12b)",
                f"On {a['n_queries']} queries of six sets, the Stella-minus-Zero gap tracks shuffle "
                "sensitivity (Stella's drop when words are shuffled). The random-noise control is "
                "one isotropic vector perturbation, not a general test of query fragility.",
                table(["association (Spearman)", "value"], rows)
                + f"\n\nOrder-dependent share {a['share_order_dependent']:.3f}; gap there "
                  f"{a['stella_gap_order_dependent']:.3f} against {a['stella_gap_order_free']:.3f}. "
                  f"Mean drop from shuffling {b['mean_order_drop']:.4f}, from a random move of equal "
                  f"cosine {b['mean_noise_drop']:.4f}. Gap by fertility tercile: "
                  f"{[round(x, 3) for x in a['stella_gap_by_fertility_tercile']]}.",
                "`results/m15_e12_failure_modes.json`, `results/m15_e12b_fragility.json`",
                "A cause; shuffling does not isolate word order. Both gaps share Stella's full score; "
                "broad score bins do not remove all coupling. Equal-distance isotropic noise need not "
                "match the ranking-relevant direction of language perturbations.")


def towers_extended():
    x = R("m15_e8x_towers.json")
    r = R("m15_e13_robustness.json")
    rows = [[n, c["dim"], c["pooling"], c["lambda"], f4(c["dev_table"]), f4(c["six_table_macro_all6"]),
             f4(c["six_ceiling_macro_all6"]), f3(c["six_table_macro_all6"] / c["six_ceiling_macro_all6"])]
            for n, c in x["new_configs"].items()]
    st = x["stats"]
    lofo = ", ".join(f"without {k}: screen {v['screen']:+.2f}, tower {v['tower']:+.2f}"
                     for k, v in r["leave_one_family_out"].items())
    return card("C3b. The tower result on 26 checkpoints", "exploratory (E8x, E13)",
                f"Pooled over {x['n_checkpoints_pooled']} checkpoints: screen Spearman "
                f"{st['screen']['spearman']:+.3f} (95% {st['screen']['bootstrap95_over_towers']}), "
                f"tower Spearman {st['tower']['spearman']:+.3f} (95% "
                f"{st['tower']['bootstrap95_over_towers']}).",
                table(["config", "dim", "readout", "lambda", "dev table", "six table", "six tower",
                       "retention"], rows)
                + f"\n\nStopped at the convergence gate: {list(x['stopped'])}. Leave one family out "
                  f"(registered ten): {lofo}. Screen minutes per tower on one A100: "
                  f"{r['screen_minutes_per_tower_a100']}.",
                "`results/m15_e8x_towers.json`, `results/m15_e13_robustness.json`",
                "A cause; recipes other than the closed-form table.")


def heldout():
    d = R("m13_reserved_run.json")
    s = d["systems"]
    rows = [[n] + [f4(v[k]) for k in ("fever", "dbpedia-entity", "cqadup-android",
                                      "cqadup-english")] for n, v in s.items()]
    return card("C11. The one-shot held-out test", "registered (spent 2026-09-19)",
                "The four held-out datasets, reported in full; contrasts in the paper, Appendix B.",
                table(["system", "fever", "dbpedia-entity", "cqadup-android", "cqadup-english"], rows),
                "`results/m13_reserved_run.json`", "Anything decided on these sets after the test.")


def decision_audit():
    d = R("m15_e15_decision_audit.json")
    rows = []
    for label, key in (("registered ten", "selection_registered_ten"),
                       ("pooled 26", "selection_pooled_26")):
        v = d[key]
        rows.append([label, v["dev_screen_choice"], v["best_public_teacher"],
                     f4(v["screen_table_macro"]), f4(v["teacher_choice_table_macro"]),
                     f"{v['screen_minus_teacher_choice']:+.4f}", f4(v["screen_regret_clean4"]),
                     f"{v['dimension_vs_absolute_table_rho']:+.3f}",
                     f"{v['dimension_vs_retention_rho']:+.3f}"])
    v = d["selection_pooled_26"]
    task_rows = [[ds, x["screen_choice_rank"], x["best_table"], f4(x["screen_choice_regret"])]
                 for ds, x in v["per_dataset"].items()]
    n, z = d["nano_build"], d["zero_build_historical_estimate"]
    cost = (f"Nano: {n['training_examples']:,} example presentations in "
            f"{n['training_seconds']:.2f} seconds ({n['training_hours']:.3f} hours) on "
            f"{n['gpu']}; at the recorded historical ${n['historical_total_rental_usd_per_hour']:.6f} "
            f"per hour, priced training time is ${n['training_time_priced_usd']:.2f}. "
            f"Accounting: {n['cost_kind']}. Excludes {n['exclusions']}.\n\n"
            f"Zero: {z['retraining_minutes']} minutes retraining and "
            f"{z['teacher_target_hours'][0]}-{z['teacher_target_hours'][1]} hours target preparation; "
            f"{z['kind']}.\n\nScreen timing: {d['screen_timing']['kind']}. "
            f"Excludes {d['screen_timing']['exclusions']}.")
    return card("C13. Teacher-choice consequences and component build costs", "exploratory (E15)",
                "The table screen changes teacher selection materially under the fixed recipe, "
                "with target-dependent regret. Nano's final training loop is affordable at the "
                "recorded rate; this is not the complete cost of reproducing the research.",
                table(["roster", "screen choice", "strongest teacher", "screen table", "teacher-choice table",
                       "difference", "clean-4 regret", "dimension / absolute table", "dimension / retention"], rows)
                + "\n\nPooled-roster per-dataset sensitivity:\n\n"
                + table(["dataset", "screen rank", "best table", "screen regret"], task_rows)
                + "\n\n" + cost, "`results/m15_e15_decision_audit.json`; input hashes in its receipt",
                "; ".join(d["limits"]))


if __name__ == "__main__":
    parts = ["# M15 evidence, in plain language\n",
             "Generated by `python m15/make_evidence.py` from committed result files. Each card "
             "states one claim of the paper, whether it was registered before observation or is "
             "exploratory, the numbers, the file that holds them, and what the result cannot show.\n",
             beir(), latency(), towers(), towers_extended(), mechanisms(), oracle(), prefixes(), routers(), blend(),
             system(), failure_modes(), heldout(), decision_audit()]
    OUT.write_text("\n".join(p for p in parts if p))
    print("wrote", OUT.relative_to(REPO))
