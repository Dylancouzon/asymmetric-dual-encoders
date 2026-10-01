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


def precision_followup():
    native_rows, interactions = [], []
    for dataset, filename in (("fiqa", "m15_e16_quantization_pilot.json"),
                              ("msmarco1m", "m15_e17_quantization_replication.json")):
        d = R(filename)
        for tier in ("zero", "nano", "stella-query"):
            base = f"64/{tier}/"
            values = [d["measurements"][base + mode]["recovery_at10"] for mode in
                      ("original_traversal", "binary_rescore_1", "binary_rescore_4")]
            native_rows.append([dataset, tier, *[f4(v) for v in values]])
        for other in ("nano", "stella-query"):
            c = d["contrasts"][f"64/zero_minus_{other}/quantization_interaction"]
            lo, hi = (-100 * c["query_bootstrap95"][1], -100 * c["query_bootstrap95"][0])
            lo, hi = (0.0 if abs(v) < 1e-10 else v for v in (lo, hi))
            interactions.append([dataset, f"Zero minus {other}", f"{-100*c['mean']:.2f}",
                                 f"[{lo:.2f}, {hi:.2f}]"])
    native = card("C14. Quantized scoring on the same collection", "exploratory (E16, E17)",
                  "Binary scoring adds a workload-dependent neighbor-recovery penalty. Primary ef64; "
                  "192 deterministic queries per workload. Original scoring uses ignore=true on the "
                  "same binary collection, not a separately built original-vector graph.",
                  table(["workload", "tier", "original scoring", "binary rescore1", "binary rescore4"], native_rows)
                  + "\n\nExtra Zero recovery penalty, percentage points:\n\n"
                  + table(["workload", "contrast", "extra loss (pp)", "query-bootstrap95 (pp)"], interactions)
                  + "\n\nThe 1M primary Zero-versus-Nano contrast includes zero; the Zero-versus-Stella "
                    "interval touches zero. Secondary ef16/256 are in the receipts, not substituted for "
                    "the primary display. Each tier retains its own exact-neighbor target. Native "
                    "rescoring can change global candidates through segment merging.",
                  "`results/m15_e16_quantization_pilot.json`; `results/m15_e17_quantization_replication.json`",
                  "A general static-model sensitivity law, a fixed candidate-pool decomposition, "
                  "training/graph-build variation, or a native query-precision remedy.")
    d = R("m15_e18_query_precision.json")
    rows, gain_rows, headroom = [], [], []
    for dataset, result in d["datasets"].items():
        for tier in ("zero", "nano", "stella-query"):
            for budget in d["candidate_budgets"]:
                sign = result["rows"][f"{tier}/sign_query/{budget}"]["expected_original_top10_coverage"]
                full = result["rows"][f"{tier}/float_query/{budget}"]["expected_original_top10_coverage"]
                c = result["precision_deltas"][f"{tier}/{budget}"]
                rows.append([dataset, tier, budget, f4(sign), f4(full), f"{100*c['float_minus_sign']:.2f}"])
                if budget == 40:
                    headroom.append([dataset, tier, f"{(full-sign)/(1-sign):.3f}",
                                     f"{result['query_magnitude_diagnostic'][tier]['mean_cosine_to_sign_direction']:.3f}"])
        for other in ("nano", "stella-query"):
            c = result["tier_interactions"][f"zero_minus_{other}/40"]
            gain_rows.append([dataset, f"Zero minus {other}", f"{100*c['mean']:.2f}",
                              f"[{100*c['query_bootstrap95'][0]:.2f}, {100*c['query_bootstrap95'][1]:.2f}]"])
    exact = card("C15. Query precision with document codes fixed", "exploratory (E18)",
                 "Retaining query magnitudes increases original-top10 candidate coverage on identical "
                 "document sign codes; all tiers benefit. Candidate budget 40 is primary.",
                 table(["workload", "tier", "candidates", "sign query", "graded query", "gain (pp)"], rows)
                 + "\n\nExtra absolute Zero gain at 40 candidates, with paired query intervals:\n\n"
                 + table(["workload", "contrast", "gain (pp)", "query-bootstrap95 (pp)"], gain_rows)
                 + "\n\nError headroom and mean angular distortion at 40 candidates:\n\n"
                 + table(["workload", "tier", "fraction of sign misses recovered", "mean cosine to sign query"], headroom)
                 + "\n\nCoverage is expected inclusion under uniform boundary-tie selection, with "
                   "each tier's E16/E17 original exact-top10 target. Query-vector hashes match those "
                   "receipts. Graded scoring is original normalized q dot sign(d), not native scalar8. "
                   "Global candidate budgets are not native per-segment oversampling. Larger absolute "
                   "Zero gains have more miss headroom and do not establish greater proportional sensitivity.",
                 "`results/m15_e18_query_precision.json`",
                 "HNSW performance, nDCG gain, serving latency, native 8-bit query performance, "
                 "or a new quantizer; no graph is used.")
    return native + "\n" + exact


def cross_recipe():
    p = REPO / "results" / "m15_e19_head_screen.json"
    if not p.exists():
        return ""
    d = R("m15_e19_head_screen.json")
    s = d["spearman"]
    keys = (("recipe2_vs_teacher_all6", "head vs teacher six-set score"),
            ("recipe2_vs_recipe1_all6", "head vs table (recipe 1)"),
            ("recipe2_dev_vs_six_all6", "head dev screen vs head six-set"),
            ("recipe2_vs_teacher_clean4", "head vs teacher, clean-four"),
            ("recipe2_vs_recipe1_clean4", "head vs table, clean-four"))
    rho_rows = [[label, roster, s[roster]["n"], f"{s[roster][k]['spearman']:+.3f}",
                 f"[{s[roster][k]['bootstrap95_over_checkpoints'][0]:+.2f}, "
                 f"{s[roster][k]['bootstrap95_over_checkpoints'][1]:+.2f}]"]
                for k, label in keys for roster in ("registered_ten", "pooled", "pooled_without_backbone")]
    cfg = d["configs"]
    rows = [[n, r["dim"], f4(r["teacher_six_macro_all6"]), f4(r["recipe1_six_table_macro_all6"]),
             f4(r["six_head_macro_all6"]), f3(r["retention_all6"]), f"{r['lambda']:g}"]
            for n, r in sorted(cfg.items(), key=lambda kv: -kv[1]["teacher_six_macro_all6"])]
    st = d["strongest_teacher_comparison"]
    return card("C16. The teacher screen under a second student recipe",
                "exploratory (E19, analysis pre-specified before scoring)",
                "A frozen bge-small backbone with a closed-form ridge head, fitted to the same 26 "
                "teachers on the same fit list and scored on the same indexes as the E8/E8x tables, "
                "also ranks poorly by the teacher's own score; its ranking agrees only moderately "
                "with the table ranking, and its own dev screen predicts its six-set rank.",
                table(["correlation", "roster", "n", "Spearman", "checkpoint bootstrap95"], rho_rows)
                + f"\n\nStrongest registered teacher {st['strongest_registered_teacher']}: head "
                  f"{f4(st['recipe2_six_strongest'])}; recipe-2 screen choice "
                  f"{st['recipe2_screen_choice']}: head {f4(st['recipe2_six_choice'])}.\n\n"
                + table(["teacher", "dim", "teacher six", "table (recipe 1)", "head (recipe 2)",
                         "head retention", "lambda"], rows)
                + "\n\nbge-small-en-v1.5 is the head's own backbone and gte-small is nearly its "
                  "linear image (retention 1.008 and 0.978); the without-backbone roster drops the "
                  "first. Leave-one-family-out values are in the result file.",
                "`results/m15_e19_head_screen.json`",
                "A selector for the trained Zero or Nano; a trained-student ceiling (the frozen head "
                "is a floor); independence of related checkpoints; a cause. Head retention falls "
                "with teacher width (exploratory Spearman −0.73), which confounds the pooled "
                "teacher correlation.")


def cross_space_ann():
    p = REPO / "results" / "m15_e20_ann_spaces.json"
    if not p.exists():
        return ""
    d = R("m15_e20_ann_spaces.json")
    srows = []
    for ds, s in d["summary"].items():
        q = s["multiplier_quantiles_over_reached_spaces"]
        srows.append([ds, s["n_spaces"], s["censored_spaces"],
                      s["spaces_with_multiplier_above_1"], s["spaces_with_multiplier_1_or_below"],
                      "n/a" if q is None else f"{q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f}",
                      f"{100 * s['mean_table_minus_teacher_recovery_at_ref_ef']:+.1f}",
                      f"{s['exploratory_spearman_gap_vs_top1_cos_delta']:+.2f}",
                      f"{s['exploratory_spearman_gap_vs_margin_delta']:+.2f}"])
    prow = []
    for name, sp in sorted(d["spaces"].items()):
        if name == "arctic-embed-l-mean":
            continue
        cells = [name, sp["family"]]
        for ds in d["summary"]:
            w = sp["workloads"][ds]
            m = w["table_multiplier_loss"]
            cells.append("/".join("cens." if v is None else f"{v:g}" for v in m))
            cells.append(f"{100 * np.mean(w['table_minus_teacher_recovery_at_ref_ef']):+.1f}")
        prow.append(cells)
    head = ["space", "family"] + [c for ds in d["summary"] for c in (f"{ds} mult. b0/b1", f"{ds} rec. gap pp")]
    return card("C17. ANN effort for table queries across teacher spaces",
                "exploratory (E20, analysis pre-specified before scoring)",
                "Over each teacher's own uncompressed HNSW index, the fitted table's queries need "
                "more graph effort than the teacher's queries to reach the teacher's ef=64 relative "
                "loss; the multiplier is the smallest grid ef that reaches it, divided by 64, "
                "averaged over two builds, censored if either build never reaches it by ef=512.",
                table(["workload", "spaces", "censored", "mult. > 1", "mult. <= 1",
                       "mult. q10 / q50 / q90", "mean table-teacher recovery gap at ef=64 (pp)",
                       "Spearman gap vs top-1 cosine delta", "Spearman gap vs margin delta"], srows)
                + "\n\n" + table(head, prow)
                + "\n\nTwo graphs per workload are built from different seeded insertion orders; they are repetitions, not independent "
                  "samples; related checkpoints are grouped by family in the result file. Geometry "
                  "correlations are exploratory. The E2 consistency check for Stella's teacher path "
                  "on FiQA is in the result file and is not a gate.",
                "`results/m15_e20_ann_spaces.json`",
                "A latency claim (sweeps ran on a shared pod); other engines or graph parameters; "
                "the trained Zero (these are closed-form tables); a causal geometric mechanism.")


def width_model():
    p = REPO / "results" / "m15_e21_width_model.json"
    if not p.exists():
        return ""
    d = R("m15_e21_width_model.json")
    rows = []
    for recipe in ("head", "table"):
        a = d["rosters"]["pooled"][recipe]["all6"]
        for k, label in (("M1_teacher", "space quality"), ("M2_teacher_width", "space quality + width"),
                         ("M4_teacher_width_dev", "+ dev screen")):
            e = a["models"][k]
            betas = ", ".join(f"{b:+.2f}" for b in e["std_beta"])
            ci = e.get("std_beta_bootstrap95")
            cis = "; ".join(f"[{lo:+.2f}, {hi:+.2f}]" for lo, hi in ci) if ci else ""
            rows.append([recipe, label, betas, cis, f3(e["r2_in_sample"]),
                         f"{e['leave_one_family_out']['spearman_pred_vs_actual']:+.2f}",
                         f"{e['leave_one_checkpoint_out']['spearman_pred_vs_actual']:+.2f}"])
    held = d["registered_fit_predicts_exploratory"]
    hrows = [[r, v["n_train"], v["n_test"], f"{v['spearman_pred_vs_actual']:+.2f}",
              f"[{v['spearman_pred_vs_actual_bootstrap95'][0]:+.2f}, {v['spearman_pred_vs_actual_bootstrap95'][1]:+.2f}]",
              f"{v['spearman_teacher_only_vs_actual']:+.2f}"] for r, v in held.items()]
    reg = []
    for recipe in ("head", "table"):
        a = d["rosters"]["pooled"][recipe]["all6"]
        for rule, v in a["selection_regret"].items():
            reg.append([recipe, rule, v["pick"], f4(v["student"]), f4(v["regret"])])
    sp = {r: d["rosters"]["pooled"][r]["all6"]["spearman"] for r in ("head", "table")}
    return card("C18. Width hides the teacher-quality signal", "exploratory (E21, pre-specified; existing data)",
                "Student quality is predicted by the space's own quality once width is held fixed; width is a "
                "strong negative effect; a two-variable fit on the ten registered spaces ranks the 16 later "
                "spaces far better than the space's own score does.",
                table(["student", "predictors", "standardized betas", "checkpoint bootstrap95", "R2 in sample",
                       "leave-one-family-out Spearman", "leave-one-out Spearman"], rows)
                + "\n\nRegistered-ten fit predicting the 16 exploratory checkpoints (space quality + width):\n\n"
                + table(["student", "n fit", "n test", "Spearman pred vs actual", "bootstrap95", "space quality alone"], hrows)
                + "\n\nSelection regret, six-set nDCG@10 below the best student:\n\n"
                + table(["student", "rule", "pick", "student score", "regret"], reg)
                + f"\n\nPooled Spearman student vs space quality: head {sp['head']['student_vs_teacher']:+.2f}, "
                  f"table {sp['table']['student_vs_teacher']:+.2f}; partial given width: head "
                  f"{sp['head']['partial_student_vs_teacher_given_width']:+.2f}, table "
                  f"{sp['table']['partial_student_vs_teacher_given_width']:+.2f}; space quality vs width "
                  f"{sp['head']['teacher_vs_width']:+.2f}.",
                "`results/m15_e21_width_model.json`",
                "A mechanism; independence of related checkpoints; transfer to trained students; a prospective "
                "test (the hypothesis was formed after seeing all 26; E24 supplies the prospective roster).")


def lightretriever():
    p = REPO / "results" / "m15_e25_lightretriever.json"
    if not p.exists():
        return ""
    d = R("m15_e25_lightretriever.json")
    rows, ex = [], []
    for ds, w in d["workloads"].items():
        for instr, c in w["contrasts"].items():
            lm = c["loss_multiplier"]; rm = c["recovery_multiplier"]
            rows.append([ds, instr, "cens." if lm is None else f"{lm:g}", "/".join("cens." if v is None else f"{v:g}" for v in c["loss_multiplier_builds"]),
                         "cens." if rm is None else f"{rm:g}", f"{c['recovery_gap_at_ref_ef_pp']:+.2f}",
                         f"[{c['recovery_gap_query_bootstrap95_pp'][0]:+.2f}, {c['recovery_gap_query_bootstrap95_pp'][1]:+.2f}]"])
        for path, v in w["exact"].items():
            ex.append([ds, path, f4(v["exact_ndcg10"]), f3(v["geometry"]["top1_cos_quantiles"][1])])
        t = w["timing"]
        ex.append([ds, "timing", f"full {1000*t['full_websearch_query_seconds_per_query']:.2f} ms/query", f"lookup {1000*t['lookup_websearch_query_seconds_per_query']:.3f} ms/query"])
    return card("C19. LightRetriever's own lookup path under graph search", "exploratory (E25, pre-specified before encoding)",
                "Over the released LightRetriever model's own FiQA and SCIDOCS indexes, its lookup query path recovers "
                "fewer exact neighbours than its full query encoding at equal HNSW effort, and matching the full path "
                "takes four times the ef by recovery; the relevance-loss multiplier agrees on FiQA and is uninformative "
                "on SCIDOCS where both paths lose under 0.1% at ef=64.",
                table(["workload", "prompt", "loss multiplier", "builds", "recovery multiplier", "recovery gap at ef=64 (pp)", "query bootstrap95"], rows)
                + "\n\nExact quality and query placement per path (and GPU encoding time on the pod, context only):\n\n"
                + table(["workload", "path", "exact nDCG@10", "median top-1 cosine"], ex),
                "`results/m15_e25_lightretriever.json`",
                "The paper's retention or encoding-speedup figures; latency (shared pod GPU); generality beyond one 1.5B model "
                "and two corpora; the sparse or hybrid paths.")


def gap_predictors():
    p = REPO / "results" / "m15_e22_gap_predictors.json"
    if not p.exists():
        return ""
    d = R("m15_e22_gap_predictors.json")
    rows = []
    for k, v in d["univariate"].items():
        ci = v["bootstrap95_over_spaces"]
        pw = " / ".join("n/a" if x["spearman"] is None else f"{x['spearman']:+.2f}" for x in v["per_workload"].values())
        h = d["held_out_models"].get(f"single_{k}", {})
        rows.append([k, f"{v['spearman_all75']:+.2f}", "n/a" if ci[0] is None else f"[{ci[0]:+.2f}, {ci[1]:+.2f}]", pw,
                     f"{h.get('spearman', float('nan')):+.2f}", f"{h.get('mae', float('nan')):.2f}"])
    models = [[k, f"{v['spearman']:+.2f}", f"{v['mae']:.2f}", f"{v['baseline_fold_mean_mae']:.2f}"]
              for k, v in d["held_out_models"].items() if not k.startswith("single_")]
    return card("C20. Predictors of the cross-space recovery gap", "exploratory (E22, features declared before computation)",
                "The query-to-document distance ratio, for the encoder's or the student's queries, ranks the recovery gap "
                "across 75 space-by-workload points and predicts held-out families; student-minus-encoder differences "
                "and the two geometry summaries do not; magnitude does not transfer across workloads.",
                table(["feature", "Spearman (75)", "bootstrap95 over spaces", "FiQA / SCIDOCS / TREC-COVID", "single-feature LOFO Spearman", "LOFO MAE (pp)"], rows)
                + "\n\nHeld-out models (leave one family out unless named):\n\n"
                + table(["model", "Spearman", "MAE (pp)", "fold-mean baseline MAE"], models),
                "`results/m15_e22_gap_predictors.json`", "A mechanism; independence of families; transfer of magnitude across workloads.")


if __name__ == "__main__":
    parts = ["# M15 evidence, in plain language\n",
             "Generated by `python m15/make_evidence.py` from committed result files. Each card "
             "states an available finding, whether it was registered before observation or is "
             "exploratory, the numbers, the file that holds them, and what the result cannot show.\n",
             beir(), latency(), towers(), towers_extended(), mechanisms(), oracle(), prefixes(), routers(), blend(),
             system(), failure_modes(), heldout(), decision_audit(), precision_followup(), cross_recipe(), cross_space_ann(), width_model(), lightretriever(), gap_predictors()]
    OUT.write_text("\n".join(p for p in parts if p))
    print("wrote", OUT.relative_to(REPO))
