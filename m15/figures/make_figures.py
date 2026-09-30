"""Regenerate every paper figure from committed JSON: `python m15/figures/make_figures.py`.

Colors follow one fixed assignment: Stella query path blue, Nano orange, Zero aqua, reference
systems gray. Each figure is written as PDF (for LaTeX) and PNG (for the Markdown draft).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
R = lambda name: json.loads((REPO / "results" / name).read_text())
C = {"stella": "#2a78d6", "nano": "#eb6834", "zero": "#1baf7a", "ref": "#8a8986", "ink": "#0b0b0b",
     "muted": "#52514e"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink"],
                     "xtick.color": C["muted"], "ytick.color": C["muted"], "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "lines.linewidth": 2})


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def beir_macros():
    t = R("m20_beir15_run.json")["table"]
    systems = list(next(iter(t.values()))["systems"])
    return {s: float(np.mean([t[d]["systems"][s] for d in t])) for s in systems}, t


def f1_frontier():
    mac, _ = beir_macros()
    e1 = R("m15_e1_latency.json")["summary"]
    st = mac["stella-query"]
    pts = [("Stella query path", "stella-query", "stella-query", C["stella"], True),
           ("Nano", "nano", "nano-dense", C["nano"], True),
           ("Zero", "zero", "zero-dense", C["zero"], True),
           ("bge-small (own index)", "bge-small", "bge-small-en-v1.5", C["ref"], False),
           ("LEAF (own index)", "leaf-query", "leaf-ir-asym", C["ref"], False)]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for label, e, s, color, filled in pts:
        x = e1[e]["p50_ms_median_of_trials"]["medium"]
        y = mac[s] / st
        ax.scatter(x, y, s=60, color=color if filled else "white", edgecolor=color, zorder=3,
                   linewidth=1.8)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(7, -3), fontsize=8,
                    color=C["ink"])
    ax.set_xscale("log")
    ax.set_xlabel("Query encode latency, p50 (ms, M5 Pro CPU, log scale)")
    ax.set_ylabel("BEIR-15 nDCG@10 retention\n(vs Stella query path)")
    ax.set_ylim(0.78, 1.02)
    ax.set_xlim(0.02, 120)
    save(fig, "f1_frontier")


def f2_per_dataset():
    _, t = beir_macros()
    rows = sorted(((d, v["systems"]["zero-dense"] / v["systems"]["stella-query"],
                    v["systems"]["nano-dense"] / v["systems"]["stella-query"]) for d, v in t.items()),
                  key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    y = np.arange(len(rows))
    for i, (d, z, n) in enumerate(rows):
        ax.plot([z, n], [i, i], color="#d9d8d3", linewidth=2, zorder=1)
    ax.scatter([r[1] for r in rows], y, color=C["zero"], s=36, zorder=3, label="Zero")
    ax.scatter([r[2] for r in rows], y, color=C["nano"], s=36, zorder=3, label="Nano")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Retention of the Stella query path's nDCG@10")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "f2_per_dataset")


SHORT = {"stella-400M-v5": "stella", "arctic-embed-l": "arctic-l", "arctic-embed-m-v1.5": "arctic-m",
         "bge-base-en-v1.5": "bge-base", "bge-large-en-v1.5": "bge-large", "e5-base-v2": "e5-base",
         "e5-large-v2": "e5-large", "gte-base-en-v1.5": "gte-base", "gte-large-en-v1.5": "gte-large",
         "mxbai-embed-large-v1": "mxbai"}
PAIRS = [("bge-base-en-v1.5", "bge-large-en-v1.5"), ("e5-base-v2", "e5-large-v2"),
         ("gte-base-en-v1.5", "gte-large-en-v1.5"), ("arctic-embed-m-v1.5", "arctic-embed-l")]


def f3_towers():
    e8 = R("m15_e8_towers.json")
    cf = e8["configs"]
    rho = e8["spearman"]["checkpoints"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    panels = [("six_ceiling_macro_all6", "Tower's own nDCG@10, six public sets",
               f"Tower quality does not predict the table\nSpearman {rho['six_ceiling_vs_six_table_all6']:+.2f}"),
              ("dev_table", "Table nDCG@10 on two dev forums",
               f"A cheap screen of the table does\nSpearman {rho['primary_dev_table_vs_six_table_all6']:+.2f}")]
    for ax, (xkey, xlabel, title) in zip(axes, panels):
        for a, b in PAIRS:
            ax.plot([cf[a][xkey], cf[b][xkey]],
                    [cf[a]["six_table_macro_all6"], cf[b]["six_table_macro_all6"]],
                    color="#d9d8d3", linewidth=1.2, zorder=1)
        for name, short in SHORT.items():
            x, y = cf[name][xkey], cf[name]["six_table_macro_all6"]
            color = C["stella"] if name == "stella-400M-v5" else C["muted"]
            ax.scatter(x, y, s=34, color=color, zorder=3)
            ax.annotate(short, (x, y), textcoords="offset points", xytext=(5, 2), fontsize=7.5,
                        color=C["ink"])
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].set_ylabel("Table nDCG@10, six public sets")
    save(fig, "f3_towers")


def f4_system():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=False)
    for ax, ds in zip(axes, ("msmarco1m", "fiqa")):
        p = REPO / "results" / f"m15_e2_ann_{ds}.json"
        if not p.exists():
            ax.set_title(f"{ds}: pending")
            continue
        e2 = json.loads(p.read_text())
        for enc, key in (("zero", "zero"), ("nano", "nano"), ("stella-query", "stella")):
            rows = [r for r in e2["rows"] if r["encoder"] == enc]
            # Pareto front: lowest end-to-end p50 at each quality loss.
            rows = sorted(rows, key=lambda r: r["e2e_p50_ms"])
            front, best = [], 1.0
            for r in rows:
                if r["ann_ndcg_loss_rel"] < best:
                    front.append(r)
                    best = r["ann_ndcg_loss_rel"]
            ax.plot([r["e2e_p50_ms"] for r in front], [100 * r["ann_ndcg_loss_rel"] for r in front],
                    marker="o", markersize=4, color=C[key], label=enc)
        ax.set_xscale("log")
        ax.set_xlabel("End-to-end p50 (ms): encode + search")
        ax.set_ylabel("nDCG@10 lost to ANN (% of own exact)")
        ax.set_title(ds, fontsize=9, loc="left")
        ax.legend(frameon=False)
    save(fig, "f4_system")


def f5_routing():
    e5 = R("m15_e5_oracle.json")["macro_over_12"]
    fr = np.array(e5["frontier"])
    b = np.linspace(0, 1, len(fr))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1, 1.1]})
    ax.plot(b, fr, color=C["ink"], label="Oracle, largest gains first")
    ax.plot([0, 1], [e5["zero"], e5["nano"]], color=C["ref"], linestyle="--", label="Random routing")
    ax.axhline(e5["stella_query"], color=C["stella"], linewidth=1, linestyle=":")
    ax.annotate("Stella query path", (0.02, e5["stella_query"]), textcoords="offset points",
                xytext=(0, 3), fontsize=7.5, color=C["stella"])
    ax.set_xlabel("Share of queries sent to Nano")
    ax.set_ylabel("Macro nDCG@10, 12 BEIR sets")
    ax.set_title("Upper bound from label-aware routing", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    # Right: share of the oracle's advantage over random routing each signal recovers, per budget
    # (macro of router, random-at-same-share and oracle-at-same-share over its evaluation sets).
    e6 = R("m15_e6_router.json")["macro_over_12"]
    e10 = R("m15_e10_router.json")["macro"]
    signals = [("Fertility (12 sets)", e6), ("Pooled norm (12)", e10["f2_pooled_norm"]),
               ("Word count (12)", e10["f3_words"]), ("Zero's 1st-10th margin (6)", e10["f4_margin"]),
               ("Zero-BM25 top-10 agreement (6)", e10["f5_agreement"])]
    markers = {"0.1": "o", "0.25": "s", "0.5": "^"}
    for k, (name, m) in enumerate(signals):
        for bud, mk in markers.items():
            v = m[bud]
            head = v["oracle_same_fraction"] - v["random_same_fraction"]
            if head <= 0:
                continue
            eff = (v["router"] - v["random_same_fraction"]) / head
            bx.scatter(eff, k, marker=mk, s=36, color=C["zero"] if k >= 3 else C["nano"],
                       zorder=3, label=f"Nano budget {int(float(bud) * 100)}%" if k == 0 else None)
    bx.axvline(0, color=C["ref"], linewidth=1)
    bx.set_yticks(range(len(signals)), [s[0] for s in signals])
    bx.invert_yaxis()
    bx.set_xlim(-0.05, 1.0)
    bx.set_xlabel("Share of oracle gain recovered")
    bx.set_title("What a router recovers", fontsize=9, loc="left")
    bx.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "f5_routing")


if __name__ == "__main__":
    for f in (f1_frontier, f2_per_dataset, f3_towers, f4_system, f5_routing):
        f()
        print("wrote", f.__name__)
