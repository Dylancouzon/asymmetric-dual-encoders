"""Rebuild the dense slides as visuals. Keeps each slide's <aside>; Sol rewrites notes afterwards."""
import json as _json
import math
import re
from pathlib import Path

S = Path(__file__).resolve().parents[1] / "deck/project/slides"
GRAY, VIO, RED, TEAL, ORG = "#c9cede", "#9494ff", "#ff516b", "#3fc1a5", "#f5a623"
MUT, LINE, WHITE = "#9aa0b5", "#303547", "#fcfdff"
MONO = "font-family:JetBrains Mono,monospace"


# The distill slide's own note (the zero-train backup keeps its own).
ORIG_DISTILL = "<aside>[~2 min] [Start at the teacher targets.]\n\nI needed a table whose pooled vectors would make sense to an index full of Stella passages. I began with Stella's encoding of each token. I also gave rare terms more initial weight using inverse document frequency. That gives each row a starting point before I've asked it to represent a whole query.\n\n[Follow the two training phases.]\n\nFor the first pass I saved Stella's vectors and used them as targets. On paired examples I also matched how Stella distributed its scores across candidate passages. A query vector can look close to the teacher's and still put a different passage first. Matching scores gives the table a chance to learn which of those differences matters.\n\nThen I trained with query and passage pairs to put the relevant passage ahead of negatives in a frozen bank. I masked known positives and candidates Stella scored close to or better than the positive. Some of those could answer the question too. At export I folded the learned weights into the rows that the encoder reads.\n\nThe desktop retrain estimate assumes I have the teacher outputs ready. For a new teacher, I estimated eight to 12 hours just to prepare Zero's targets. I can reuse those targets across attempts. Nano can train on Stella's outputs alone, which gives me a way to train a contextual encoder on text without relevance labels.</aside>"


def aside(sid):
    m = re.search(r"<aside>.*?</aside>", (S / f"{sid}.html").read_text(), re.S)
    return m.group(0) if m else "<aside></aside>"


def write(sid, cls, title, body, src="", note=None):
    a = note if note is not None else aside(sid)
    src_html = f'<p class="src">{src}</p>\n' if src else ""
    (S / f"{sid}.html").write_text(
        f'<section id="{sid}" class="s{cls}" data-transition="fade">\n'
        f'<div class="head"><h2>{title}</h2></div>\n{body}\n{src_html}{a}\n</section>\n')


def t(x, y, s, size=28, fill=WHITE, anchor="start", weight=400, mono=False):
    size = {24: 30, 26: 30, 28: 32, 30: 34}.get(size, size)   # readable from the back of the room
    f = f' style="{MONO}"' if mono else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{f}>{s}</text>')


# 3. cloud: CPU per million queries (machine-normalized), dollars as labels, ratios as the outcome
g = ['<svg width="1664" height="560" viewBox="0 0 1664 560">']
rows = [("Stella", 123.0, "123 vCPU-h", GRAY), ("Nano", 4.786, "4.8 vCPU-h", VIO),
        ("Zero", 0.03296, "2 vCPU-min", RED)]
x0, x1, lo, hi = 200, 1060, math.log10(0.01), math.log10(300)
X = lambda v: x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)
g.append(t(x0, 30, "CPU time to encode 1M queries (log scale)", 26, MUT))
for i, (n, v, lab, c) in enumerate(rows):
    y = 80 + i * 150
    g.append(t(0, y + 52, n, 40, c, "start", 600))
    g.append(f'<rect x="{x0}" y="{y + 10}" width="{X(v) - x0:.0f}" height="60" rx="8" fill="{c}"/>')
    g.append(t(X(v) + 16, y + 52, lab, 34, WHITE, "start", 600))
for d, lab in ((0.01, "0.01"), (0.1, "0.1"), (1, "1"), (10, "10"), (100, "100 vCPU-h")):
    g.append(t(X(d), 520, lab, 24, MUT, "middle", 400, True))
g.append(t(1664, 230, "26×", 96, VIO, "end", 700))
g.append(t(1664, 380, "3,700×", 96, RED, "end", 700))
g.append(t(1664, 440, "less CPU than Stella", 30, GRAY, "end"))
g.append("</svg>")
write("cloud", " foot", "Encoding 1M Queries: $6.07, $0.24, or $0.0016",
      '<div style="margin-top:40px">' + "".join(g) + "</div>",
      "GCP c4-standard-8 (8 vCPU), us-east1, on-demand $0.395/h, at peak throughput: batch 1, 8 processes × 1 ONNX "
      "Runtime thread (Zero: NumPy), 30 s closed loop on NQ queries. vCPU-hours = 8 / queries per second × 1M / 3600. "
      "Encoding only. Exploratory.")

# 4. distill (merged with zero-train): frozen targets, two steps, one table
box = lambda x, w, title, sub, c: (
    f'<rect x="{x}" y="60" width="{w}" height="220" rx="16" fill="#151b2b" stroke="{c}" stroke-width="3"/>'
    + t(x + w / 2, 130, title, 34, WHITE, "middle", 600)
    + "".join(t(x + w / 2, 180 + 36 * k, s, 24, GRAY, "middle") for k, s in enumerate(sub)))
arrow = lambda x: f'<path d="M{x} 170 l40 0" stroke="{MUT}" stroke-width="4"/><path d="M{x + 40} 160 l14 10 l-14 10z" fill="{MUT}"/>'
g = ['<svg width="1664" height="420" viewBox="0 0 1664 420">',
     box(0, 360, "Stella targets", ["queries + passages", "computed once"], GRAY), arrow(372),
     box(430, 380, "Align", ["match Stella's", "vectors and scores"], WHITE), arrow(822),
     box(880, 380, "Rank", ["positives vs", "32,768 negatives"], WHITE), arrow(1272),
     box(1330, 334, "31 MB table", ["int8 rows", "+ row scales"], RED),
     t(0, 360, "align: 1 − cos(z, Ê_S) + KL(p_S ‖ p_z)  ·  rank: InfoNCE  ·  Ê_D(d) never changes", 30, WHITE, "start", 400, True),
     "</svg>"]
chips = ('<div class="chips" style="margin-top:28px;gap:24px;white-space:nowrap"><p class="chip">one RTX 3080</p>'
         '<p class="chip red">≈ 20 min to train</p></div>')
write("distill", " foot", "Zero Trains on a Desktop GPU Against a Frozen Index",
      '<div style="margin-top:32px">' + "".join(g) + "</div>" + chips,
      "Zero: phase 1 16,000 steps × 512, phase 2 2,500 steps InfoNCE, τ 0.02; retrain time is a historical estimate; "
      "a new teacher needs 8-12 h of target encoding. All training sources permit commercial derived weights.",
      note=ORIG_DISTILL)

# 5. nano: three layers feed one projection into Stella's space
g = ['<svg width="1664" height="520" viewBox="0 0 1664 520">']
for k in range(12):
    y = 440 - k * 34
    hot = k + 1 in (4, 8, 12)
    g.append(f'<rect x="0" y="{y}" width="300" height="26" rx="5" fill="{VIO if hot else "#232a3d"}"/>')
    if hot:
        g.append(t(318, y + 21, f"L{k + 1}", 24, VIO, "start", 600, True))
        g.append(f'<path d="M370 {y + 13} C 480 {y + 13}, 520 260, 600 260" fill="none" stroke="{VIO}" stroke-width="3"/>')
g.append(t(150, 500, "bge-small, 12 layers", 26, GRAY, "middle"))
g.append(f'<rect x="600" y="200" width="300" height="120" rx="14" fill="#1d1d3d" stroke="{VIO}" stroke-width="3"/>')
g.append(t(750, 252, "mean-pool ⊕", 32, WHITE, "middle", 600))
g.append(t(750, 292, "3 × 384 = 1,152", 26, GRAY, "middle", 400, True))
g.append(f'<path d="M900 260 l90 0" stroke="{MUT}" stroke-width="4"/><path d="M990 250 l14 10 l-14 10z" fill="{MUT}"/>')
g.append(f'<rect x="1010" y="200" width="250" height="120" rx="14" fill="#151b2b" stroke="{WHITE}" stroke-width="3"/>')
g.append(t(1135, 252, "linear W", 32, WHITE, "middle", 600))
g.append(t(1135, 292, "1,152 → 1,024", 26, GRAY, "middle", 400, True))
g.append(f'<path d="M1260 260 l90 0" stroke="{MUT}" stroke-width="4"/><path d="M1350 250 l14 10 l-14 10z" fill="{MUT}"/>')
g.append(f'<circle cx="1500" cy="260" r="130" fill="none" stroke="{GRAY}" stroke-width="3" stroke-dasharray="8 8"/>')
g.append(t(1500, 252, "Stella's", 30, WHITE, "middle", 600))
g.append(t(1500, 290, "space", 30, WHITE, "middle", 600))
g.append("</svg>")
write("nano", " foot", "Nano Reads Three Layers to Reach Stella's Space",
      '<div style="margin-top:32px">' + "".join(g) + "</div>"
      '<div class="chips" style="gap:24px"><p class="chip vio">34.5M parameters</p><p class="chip">3.49M training queries</p>'
      '<p class="chip">57.3 A100-hours ≈ $95, once</p></div>',
      "Loss: ‖f(x) − Ê_S(x)‖² on queries (query targets) and passages (document targets), 3:1. Head initialized by ridge. "
      "No relevance labels; MS MARCO used for validation only.")

# 6. bench v7: Constella on Stella's index among the 10 most-downloaded open retrievers; six BEIR sets, query FLOPs
_FF = _json.loads(Path("/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/m15_talk_flops_field.json").read_text())["results"]
_TOP = _json.loads(Path("/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/m15_talk_model_downloads.json").read_text())["top10"]
_FL = _json.loads(Path("/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/m15_talk_flops.json").read_text())["results"]
_B15 = _json.loads(Path("/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/m20_beir15_run.json").read_text())["table"]
_SIX = ("scifact", "nfcorpus", "scidocs", "trec-covid", "fiqa", "arguana")
six = lambda s: sum(_B15[k]["systems"][s] for k in _SIX) / 6
W, H, L, R, T, B = 1664, 640, 130, 60, 40, 100
ZX, BRK, X0 = 250, 360, 440                         # Zero's lookup lane, the axis break, the transformer lane
xl, xh, yl, yh = math.log10(0.12), math.log10(40), 0.26, 0.62
PX = lambda gf: X0 + (math.log10(gf) - xl) / (xh - xl) * (W - X0 - R)
PY = lambda v: T + (yh - v) / (yh - yl) * (H - T - B)
g = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
for v in (0.30, 0.40, 0.50, 0.60):
    for a, b in ((L, BRK - 20), (BRK + 20, W - R)):     # the gap marks the break between lookup and transformers
        g.append(f'<line x1="{a}" x2="{b}" y1="{PY(v):.0f}" y2="{PY(v):.0f}" stroke="{LINE}" stroke-width="2"/>')
    g.append(t(L - 16, PY(v) + 9, f"{v:.2f}", 26, MUT, "end", 400, True))
g.append(t(ZX, H - B + 44, "lookup", 26, MUT, "middle", 400, True))
for gf, lab in ((0.3, "0.3"), (1, "1"), (3, "3"), (10, "10"), (30, "30")):
    g.append(t(PX(gf), H - B + 44, lab, 26, MUT, "middle", 400, True))
g.append(t((X0 + W - R) / 2, H - 8, "GFLOPs to encode one query (log)", 26, GRAY, "middle"))
g.append(t(L, 22, "nDCG@10, six BEIR datasets", 26, GRAY))
g.append(t(W - R, 22, "10 most-downloaded open models", 26, MUT, "end"))
LAB = {"minilm-l6": ("MiniLM-L6", -22, 10, "end"), "minilm-l12": ("MiniLM-L12", 22, 18, "start"),
       "gte-small": ("gte-small", -22, 10, "end"), "e5-small-v2": ("e5-small", 22, 10, "start"),
       "bge-small-en-v1.5": ("bge-small", 22, 10, "start"), "bge-base-en-v1.5": ("bge-base", 22, 10, "start"),
       "contriever": ("contriever", 22, 10, "start"), "gte-large-en-v1.5": ("gte-large-v1.5", -22, 10, "end"),
       "mxbai-embed-large-v1": ("mxbai-large", 0, -26, "middle"), "bge-large-en-v1.5": ("bge-large", 0, 50, "middle")}
assert set(_TOP) == set(LAB), _TOP
for n in _TOP:
    r, (lab, dx, dy, anc) = _FF[n], LAB[n]
    x, y = PX(r["median_flops"] / 1e9), PY(r["six_ndcg10"])
    g.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="12" fill="#7d849c"/>')
    g.append(t(x + dx, y + dy, lab, 28, GRAY, anc))
for n, sys_, gf, c, dx, dy, anc in (("Stella", "stella-query", _FL["stella"]["median_flops"] / 1e9, WHITE, 0, -34, "middle"),
                                     ("Nano", "nano-dense", _FL["nano"]["median_flops"] / 1e9, VIO, 0, -34, "middle"),
                                     ("Zero", "zero-dense", None, RED, 0, -34, "middle")):
    x, y = (ZX if gf is None else PX(gf)), PY(six(sys_))
    g.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="20" fill="{c}"/>')
    g.append(t(x + dx, y + dy, n, 40, c, anc, 700))
g.append("</svg>")
write("bench", " foot", "One Index, Three Price Points",
      '<div style="margin-top:24px">' + "".join(g) + "</div>",
      "Exact search, mean nDCG@10 over SciFact, NFCorpus, SciDocs, TREC-COVID, FiQA, and ArguAna. Stella's training "
      "included FiQA and ArguAna; on the other four, Zero (0.410) trails BM25 (0.441). Models: the 10 most downloaded on "
      "Hugging Face (30 days to 2026-11-21) of the 25 the paper evaluated. FLOPs: analytic, each model's query prefix. Exploratory.",
      note="<aside>[~2 min] [Give the room a moment. Point to the grey dots.] These are the ten most-downloaded open "
      "embedding models I evaluated, each searching its own index. Higher means better ranking on six BEIR datasets. "
      "Further right means more compute for every query.\n\n[Point to Nano.] Nano ranks with bge-base and bge-large at the "
      "cost of the small models, and it searches the index I already have. [Point to Zero, far left.] Zero is off the "
      "transformer scale. It runs no network at all, and it still ranks above both MiniLM models, including the most "
      "downloaded one. If I add a BM25 search and fuse the two lists, Zero reaches 0.489, for the price of a sparse search and "
      "a merge on every request.\n\nOne caveat. Stella's training data overlapped with FiQA and ArguAna, two of these six "
      "datasets. On the other four, Zero trails BM25, 0.410 to 0.441. I'd judge each one on my own traffic. So how does a "
      "table with no transformer get this close? Here's what Zero does with an actual query.</aside>")

# 8. zero-mech: real rows Zero gathers for one query, pooled, next to Stella's vector

SD = _json.loads(Path("/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/m15_talk_slide_data.json").read_text())
ex = SD["zero_example"]


def strip(x, y, vals, w=760, h=46):
    m = max(abs(v) for v in vals) or 1
    cw = w / len(vals)
    out = []
    for k, v in enumerate(vals):
        a = min(abs(v) / m, 1)
        c = RED if v > 0 else TEAL
        out.append(f'<rect x="{x + k * cw:.1f}" y="{y}" width="{cw - 2:.1f}" height="{h}" fill="{c}" opacity="{0.12 + 0.88 * a:.2f}"/>')
    return "".join(out)


g = ['<svg width="1664" height="640" viewBox="0 0 1664 640">']
for i, r in enumerate(ex["rows"]):
    y = 20 + i * 64
    g.append(t(200, y + 34, r["token"], 32, MUT if r["token"].startswith("[") else WHITE, "end", 600, True))
    g.append(strip(230, y, r["row"]))
g.append(t(230, 500, "one table row per token (first 48 of 1,024 dims)", 26, MUT))
g.append(f'<path d="M1010 40 Q1060 40 1060 120 L1060 340 Q1060 420 1010 420" fill="none" stroke="{MUT}" stroke-width="3"/>')
g.append(t(1100, 210, "√count-weighted", 30, WHITE, "start", 600))
g.append(t(1100, 252, "mean, then L2", 30, WHITE, "start", 600))
g.append(t(1100, 330, "Zero", 32, RED, "start", 600))
g.append(strip(1100, 344, ex["zero_vector"], w=560))
g.append(t(1100, 450, "Stella", 32, GRAY, "start", 600))
g.append(strip(1100, 464, ex["stella_vector"], w=560))
g.append(t(1100, 580, f"cosine {ex['cos_zero_stella']:.2f}", 44, WHITE, "start", 700))
g.append("</svg>")
write("zero-mech", " foot", "Zero Builds a Query Vector by Lookup",
      '<div style="margin-top:24px">' + "".join(g) + "</div>",
      "Released Zero table (int8 rows, fp32 scale per row, dequantized), query \"cheap flights paris to rome\". "
      "Colour: sign and size of each value, scaled per strip. Stella: its query vector with prompt. results/m15_talk_slide_data.json.")

# 10. worst: slope chart, teacher score vs its fitted table, ten registered models
REG = [("gte-base-en-v1.5", .5331, .3252), ("arctic-embed-m-v1.5", .5263, .3279), ("bge-base-en-v1.5", .5259, .3529),
       ("e5-base-v2", .4669, .2930), ("gte-large-en-v1.5", .5970, .2455), ("stella-400M-v5", .5745, .3974),
       ("mxbai-embed-large-v1", .5368, .2605), ("bge-large-en-v1.5", .5329, .2845), ("arctic-embed-l", .5289, .3034),
       ("e5-large-v2", .4735, .2585)]
lx, rx, top, bot = 420, 1240, 80, 560
YL = lambda v: top + (0.61 - v) / (0.61 - 0.23) * (bot - top)   # one nDCG scale for both columns
YR = YL
g = ['<svg width="1664" height="640" viewBox="0 0 1664 640">']
g.append(t(lx, 40, "model itself", 30, GRAY, "middle", 600))
g.append(t(rx, 40, "its lookup table", 30, GRAY, "middle", 600))
for x in (lx, rx):
    g.append(f'<line x1="{x}" y1="{top - 10}" x2="{x}" y2="{bot + 10}" stroke="{LINE}" stroke-width="3"/>')
for n, a, b in sorted(REG, key=lambda r: r[0] in ("gte-large-en-v1.5", "stella-400M-v5")):
    c, w, op = (RED, 6, 1) if n == "gte-large-en-v1.5" else (WHITE, 6, 1) if n == "stella-400M-v5" else ("#4a5068", 3, 1)
    g.append(f'<line x1="{lx}" y1="{YL(a):.0f}" x2="{rx}" y2="{YR(b):.0f}" stroke="{c}" stroke-width="{w}" opacity="{op}"/>')
    g.append(f'<circle cx="{lx}" cy="{YL(a):.0f}" r="8" fill="{c}"/><circle cx="{rx}" cy="{YR(b):.0f}" r="8" fill="{c}"/>')
for n, a, b, c in (("gte-large", .5970, .2455, RED), ("Stella", .5745, .3974, WHITE)):
    g.append(t(lx - 24, YL(a) + 11, f"{n} {a:.3f}", 32, c, "end", 600))
    g.append(t(rx + 24, YR(b) + 11, f"{b:.3f}", 32, c, "start", 600))
g.append(t(rx + 24, 620, "nDCG@10, six BEIR sets", 26, MUT))
g.append("</svg>")
write("worst", " foot", "The Best Retriever Made the Weakest Table",
      '<div style="margin-top:24px">' + "".join(g) + "</div>",
      "Ten registered models, mean nDCG@10 over six BEIR datasets, exact search: each model, then its fitted lookup "
      "table (min ‖XW − Y‖² + λ‖W − W₀‖², 4-7 min on one A100). Rank correlation across all 26 models: +0.09.")

# 11. live: prepared queries, no call for Zoom input
write("live", " demo", "Live: An Auto Router on One Index",
      '<div class="live"><div class="dot"></div><p>LIVE</p></div>'
      '<p class="sub" style="font-size:40px;margin-top:24px">2.68M passages, real NQ questions, judged answers, on this laptop.</p>',
      note="<aside>" + '[~4 min] [Before the talk: run DATASET=nq .venv/bin/python m15/talk/demo/studio/app.py, open http://127.0.0.1:8767, wait until the subtitle stops saying "starting", then drag traffic to about 1,500 once to warm it up and back to 40. Leave routing on Auto, quality floor 0.50, p95 target 150 ms. Keys: space pauses, up and down arrows step the traffic by 1.5x.]\n\n[Share the studio.] This is a search service running on this laptop: the 2.68 million passages, one Stella index, real NQ questions arriving, every answer scored against the judged answers. I give the router two rules. Quality first: the average score of what it serves stays above this floor. Then latency: stay under this target. And it never drops a query.\n\n[At 40 a second.] With quiet traffic it spends the spare time on Stella with the widest search, the best ranking we have.\n\n[Drag traffic to about 600.] Fifteen times the traffic. Within a second it moves the traffic to Nano. Latency stays under the line; quality dips a little.\n\n[Drag to about 1,500.] Now it mixes in Zero, and watch the search sliders: it narrows the HNSW search on its own, per request. Same index, nothing re-encoded.\n\n[Drag to about 3,500.] Past this laptop\'s ceiling, about 2,600 a second in one process. It keeps the floor and serves every query, so the queue grows and latency climbs. That is the trade I asked for: quality first.\n\n[Drag back to 40.] The backlog drains in a few seconds, and Stella earns its traffic back.\n\n[Optional, MORE, Agent tasks.] An agent firing ten searches per task: now the gauge times the slowest of the ten.\n\nThe questions and judgments are real; the traffic pattern is simulated, and the capacity points are this laptop with one Edge shard in one process. On your hardware they move, which is the point: set your floor and your target, then measure.' + "</aside>")

# 12. close: how simple it is, with FastEmbed
code = ("from fastembed import TextEmbedding\n\n"
        "index = TextEmbedding(\"Qdrant/stella-en-400M-v5-doc-onnx\")\n"
        "query = TextEmbedding(\"Qdrant/constella-zero\")  <span class=\"mut\"># or constella-nano</span>\n\n"
        "client.upload_collection(\"docs\", list(index.embed(docs)))\n"
        "client.query_points(\"docs\", query=next(query.embed([q])))")
write("close", " alt", "Keep Your Index. <span class=\"tr\">Make the Query Encoder a Dial.</span>",
      f'<p class="code" style="white-space:pre;font-size:32px;margin-top:40px;padding:32px 40px;font-variant-ligatures:none">{code}</p>'
      '<p class="code" style="font-size:32px;margin-top:24px;display:inline-block;font-variant-ligatures:none">pip install "fastembed&gt;=0.9.0"</p>')

# ---- v5 overrides (owner review 2026-11-21) ----
import random as _random

# zero-mech v5: token rows on the left, one arrow, the result on the right
g = ['<svg width="1664" height="640" viewBox="0 0 1664 640">']
for i, r in enumerate(ex["rows"]):
    y = 30 + i * 62
    g.append(t(190, y + 34, r["token"], 32, MUT if r["token"].startswith("[") else WHITE, "end", 600, True))
    g.append(strip(215, y, r["row"], w=700))
g.append(f'<path d="M960 245 l150 0" stroke="{WHITE}" stroke-width="5"/><path d="M1110 230 l24 15 l-24 15z" fill="{WHITE}"/>')
g.append(t(1035, 300, "weighted", 28, WHITE, "middle", 600))
g.append(t(1035, 336, "mean", 28, WHITE, "middle", 600))
g.append(t(1170, 150, "Zero", 34, RED, "start", 600))
g.append(strip(1170, 166, ex["zero_vector"], w=490))
g.append(t(1170, 290, "Stella", 34, GRAY, "start", 600))
g.append(strip(1170, 306, ex["stella_vector"], w=490))
g.append(t(1170, 440, f"cosine {ex['cos_zero_stella']:.2f}", 52, WHITE, "start", 700))
g.append("</svg>")
write("zero-mech", " foot", "Zero Builds a Query Vector by Lookup",
      '<div style="margin-top:24px">' + "".join(g) + "</div>",
      "Released Zero table (int8 rows, fp32 scale per row, dequantized), query \"cheap flights paris to rome\"; "
      "√count-weighted mean, then L2. First 48 of 1,024 dimensions; colour: sign and size of each value. "
      "Stella: its query vector with prompt.")

# worst v5: bump chart with names, rank as a retriever vs rank as a lookup table
NAMES = {"gte-large-en-v1.5": "gte-large-v1.5", "stella-400M-v5": "Stella", "mxbai-embed-large-v1": "mxbai-large",
         "gte-base-en-v1.5": "gte-base-v1.5", "bge-large-en-v1.5": "bge-large", "arctic-embed-l": "arctic-l",
         "arctic-embed-m-v1.5": "arctic-m", "bge-base-en-v1.5": "bge-base", "e5-large-v2": "e5-large", "e5-base-v2": "e5-base"}
left = sorted(REG, key=lambda r: -r[1]); right = sorted(REG, key=lambda r: -r[2])
lx, rx, top, step = 560, 1100, 110, 50
g = ['<svg width="1664" height="640" viewBox="0 0 1664 640">',
     t(lx, 50, "as a retriever", 32, GRAY, "end", 600), t(rx, 50, "as a lookup table", 32, GRAY, "start", 600)]
for n, a, b in sorted(REG, key=lambda r: r[0] in ("gte-large-en-v1.5", "stella-400M-v5")):
    i, j = left.index((n, a, b)), right.index((n, a, b))
    c, w = (RED, 6) if n == "gte-large-en-v1.5" else (WHITE, 6) if n == "stella-400M-v5" else ("#3a4157", 3)
    y1, y2 = top + i * step, top + j * step
    g.append(f'<line x1="{lx + 24}" y1="{y1 - 10}" x2="{rx - 24}" y2="{y2 - 10}" stroke="{c}" stroke-width="{w}"/>')
    hot = c != "#3a4157"
    g.append(t(lx, y1, f"{NAMES[n]}  {i + 1}", 30 if hot else 28, c if hot else MUT, "end", 600 if hot else 400))
    g.append(t(rx, y2, f"{j + 1}  {NAMES[n]}", 30 if hot else 28, c if hot else MUT, "start", 600 if hot else 400))
g.append("</svg>")
write("worst", " foot", "The Best Retriever Made the Weakest Table",
      '<div style="margin-top:16px">' + "".join(g) + "</div>",
      "Ten registered models ranked by mean nDCG@10 over six BEIR datasets, exact search: the model itself, then its fitted "
      "lookup table (gte-large-en-v1.5 0.597 → 0.246; Stella 0.575 → 0.397). Rank correlation across all 26 models: +0.09.")

# recovery v5: chart only
REC = {"Stella": ([.913, .934, .948, .954, .957, .960], GRAY), "Nano": ([.867, .894, .914, .926, .933, .935], VIO),
       "Zero": ([.772, .813, .857, .878, .894, .901], RED)}
W, H, L, R, T, B = 1664, 640, 130, 200, 70, 100
RX = lambda i: L + i * (W - L - R) / 5
RY = lambda v: T + (0.98 - v) / (0.98 - 0.75) * (H - T - B)
g = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
for v in (0.75, 0.80, 0.85, 0.90, 0.95):
    g.append(f'<line x1="{L}" x2="{W - R}" y1="{RY(v):.0f}" y2="{RY(v):.0f}" stroke="{LINE}" stroke-width="2"/>')
    g.append(t(L - 16, RY(v) + 10, f"{v:.2f}", 30, MUT, "end", 400, True))
for i, e in enumerate((32, 64, 128, 256, 512, 1024)):
    g.append(t(RX(i), H - B + 46, str(e), 30, MUT, "middle", 400, True))
g.append(t((L + W - R) / 2, H - 6, "hnsw_ef (oversampling 4)", 30, GRAY, "middle"))
g.append(t(L, 34, "share of its own exact top 10 found", 30, GRAY))
g.append(f'<line x1="{RX(0)}" y1="{RY(.913):.0f}" x2="{RX(5)}" y2="{RY(.913):.0f}" stroke="{WHITE}" stroke-width="2" stroke-dasharray="8 8"/>')
for n, (ys, c) in REC.items():
    g.append(f'<polyline points="{" ".join(f"{RX(i):.0f},{RY(v):.0f}" for i, v in enumerate(ys))}" fill="none" stroke="{c}" stroke-width="6"/>')
    g.append(t(RX(5) + 24, RY(ys[-1]) + 12, n, 36, c, "start", 600))
for i, v in ((0, .913), (5, .901)):
    g.append(f'<circle cx="{RX(i):.0f}" cy="{RY(v):.0f}" r="22" fill="none" stroke="{WHITE}" stroke-width="4"/>')
g.append("</svg>")
write("recovery", " foot", "Zero Needs a Wider Search, and Still Trails",
      '<div style="margin-top:16px">' + "".join(g) + "</div>",
      "Qdrant Edge, HNSW m=16, ef_construct=100, binary quantization with fp16 rescoring, one build, M5 Pro. "
      "Across 25 models in the paper, lookup tables needed a median ef of 256 to match their teacher at 64. Exploratory.",
      note="<aside>[~2.5 min] [Before the chart: the two kinds of settings.] Qdrant's HNSW index has two kinds of settings. "
      "The first kind is fixed when the index is built. m is how many links each passage keeps to its neighbours in the graph; "
      "I used 16, the default. ef_construct is how many candidates Qdrant checks when it chooses those links; I used 100. "
      "Together they shape the graph. Changing either one means rebuilding the index. I built this graph once, from Stella's "
      "passage vectors.\n\nThe second kind I can change on every request. hnsw_ef is how many candidates the search keeps "
      "while it walks the graph. A wider ef finds more of the true nearest neighbours and takes more time. Oversampling works "
      "with quantization. This index keeps a one-bit binary code for every passage in RAM and the full vectors on disk. Qdrant "
      "first fetches four times the results I asked for using the binary codes, then rescores those with the full vectors.\n\n"
      "[Point to the curves.] Each line asks one question: of this encoder's own exact top 10, how much does the search find? "
      "[Point to the two circles.] Zero at ef 1024 still finds less than Stella at ef 32. The graph's links were chosen with "
      "Stella's passage vectors. Stella's queries land where those links lead. Zero's query vectors land a little off, so the "
      "walk needs more room. Across 25 models in the paper, the median lookup table needed ef 256 to match its teacher at 64.\n\n"
      "So I don't change the graph per encoder. I give each encoder its own ef and oversampling at query time, on the same "
      "graph. The encoder and the search width become one decision, and the right answer changes with traffic. That's the "
      "decision I handed to a router.</aside>")

# close v5: VS Code Dark+ colored example
def _code(lines):
    K, F, S_, C_, V, T_, P_ = "#C586C0", "#DCDCAA", "#CE9178", "#6A9955", "#9CDCFE", "#4EC9B0", "#D4D4D4"
    sp = lambda col, s: f'<span style="color:{col}">{s}</span>'
    return lines(sp, K, F, S_, C_, V, T_, P_)
code = _code(lambda sp, K, F, S_, C_, V, T_, P_: "\n".join([
    sp(K, "from") + " fastembed " + sp(K, "import") + " " + sp(T_, "TextEmbedding"),
    "",
    sp(C_, "# 1. Index once with Stella's document model"),
    sp(V, "stella") + " = " + sp(T_, "TextEmbedding") + "(" + sp(S_, '"Qdrant/stella-en-400M-v5-doc-onnx"') + ")",
    sp(V, "client") + "." + sp(F, "upload_collection") + "(" + sp(S_, '"wiki"') + ", " + sp(F, "list") + "(" + sp(V, "stella") + "." + sp(F, "embed") + "(" + sp(V, "passages") + ")))",
    "",
    sp(C_, "# 2. Query the same index with Zero, or swap in Nano"),
    sp(V, "zero") + " = " + sp(T_, "TextEmbedding") + "(" + sp(S_, '"Qdrant/constella-zero"') + ")",
    sp(V, "query") + " = " + sp(F, "next") + "(" + sp(V, "zero") + "." + sp(F, "embed") + "([" + sp(S_, '"how far is the moon?"') + "]))",
    sp(V, "hits") + " = " + sp(V, "client") + "." + sp(F, "query_points") + "(" + sp(S_, '"wiki"') + ", " + sp(V, "query") + "=" + sp(V, "query") + ")",
]))
write("close", " alt", "Use the Constella Models <span class=\"tr\">Today</span>",
      f'<pre style="margin-top:36px;background:#1E1E1E;color:#D4D4D4;border-radius:16px;padding:36px 44px;font-family:JetBrains Mono,monospace;'
      f'font-size:32px;line-height:1.5;font-variant-ligatures:none;border:2px solid #303547">{code}</pre>'
      '<p style="margin-top:24px;font-family:JetBrains Mono,monospace;font-size:32px;color:#D4D4D4;font-variant-ligatures:none">'
      '<span style="color:#DCDCAA">pip</span> install <span style="color:#CE9178">"fastembed&gt;=0.9.0"</span></p>')

# thanks: four QR codes
def _qr(name):
    f = Path(__file__).parent / f"qr_{name}.svg"
    if not f.exists():
        return None
    svg = f.read_text()
    m = re.search(r'width="(\d+)" height="(\d+)"', svg)
    return svg.replace(m.group(0), f'width="300" height="300" viewBox="0 0 {m.group(1)} {m.group(2)}"', 1)
cards = []
for name, label in (("linkedin", "LinkedIn"), ("paper", "Paper preview"), ("modelcard", "Model card"), ("fastembed", "FastEmbed")):
    q = _qr(name)
    body = q if q else '<div style="width:300px;height:300px;border:3px dashed #656b7f;border-radius:12px;display:flex;align-items:center;justify-content:center"><p style="font-size:28px;color:#9aa0b5">URL needed</p></div>'
    cards.append(f'<div class="col" style="gap:20px;align-items:center"><div style="background:#fff;border-radius:16px;padding:12px">{body}</div>'
                 f'<p style="font-size:36px;font-weight:600;color:#fcfdff">{label}</p></div>')
write("thanks", " alt", "Thank You",
      '<div class="row" style="gap:64px;margin-top:64px;justify-content:center">' + "".join(cards) + "</div>",
      note="<aside>[~0.5 min] [Leave this up through Q&amp;A.] Thank you. The paper preview, the model card and FastEmbed are all here, and you can find me on LinkedIn. Happy to take questions.</aside>")

# cloud-e2e: the whole request on one VM (warm cache), dollars per million
_E2E = _json.loads(Path("/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/m15_talk_throughput_nq_gcp-c4-standard-8-warm_e2e.json").read_text())["results"]["e2e"]
_best = {n: max(by.values(), key=lambda r: r["qps"]) for n, by in _E2E.items()}
g = ['<svg width="1664" height="560" viewBox="0 0 1664 560">']
x0, xw = 200, 760
mx = _best["stella"]["usd_per_million"]
g.append(t(x0, 30, "$ per 1M queries, encode + search", 30, MUT))
for i, (n, c) in enumerate((("stella", GRAY), ("nano", VIO), ("zero", RED))):
    y, v = 80 + i * 150, _best[n]["usd_per_million"]
    g.append(t(0, y + 52, n.capitalize(), 40, c, "start", 600))
    g.append(f'<rect x="{x0}" y="{y + 10}" width="{max(8, v / mx * xw):.0f}" height="60" rx="8" fill="{c}"/>')
    g.append(t(x0 + max(8, v / mx * xw) + 16, y + 52, f"${v:.2f}", 36, WHITE, "start", 600))
r = mx / _best["zero"]["usd_per_million"]
g.append(t(1664, 250, f"{r:.0f}×", 120, RED, "end", 700))
g.append(t(1664, 310, "cheaper than Stella", 32, GRAY, "end"))
g.append("</svg>")
write("cloud-e2e", " foot", "The Whole Request on a Cloud VM: 29× Cheaper",
      '<div style="margin-top:40px">' + "".join(g) + "</div>",
      "GCP c4-standard-8, $0.395/h, NQ, Qdrant Edge in one process, best of 1-8 threads, each encoder at its selected "
      "setting, warm page cache. Nano and Zero both stop near 600-650 queries/s: the search process sets the ceiling. Exploratory.",
      note="<aside>[~1.25 min] That was one laptop, one query at a time. So I put the same index on the cloud machine from earlier and paid for the whole request, encoding plus search. [Point to the two short bars.] Encoding alone made Zero look three thousand times cheaper than Stella. With search in the bill, both small encoders land around 17 or 18 cents per million, about 29 times cheaper than Stella. That's still a big saving. It's just a different number from the one on the encoding slide. [Point to the ratio.] Nano and Zero tie here because the search process becomes the ceiling, around six hundred queries a second in one process. More processes or shards would move that ceiling. A cheaper encoder wouldn't. And Zero spends more of that search budget than the others, because it needs a wider search to find its own neighbors.</aside>")

# ---- v8 overrides (owner review 2026-11-22) ----
_R = "/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/"


def box(x, y, w, h, c, lines, fill="#151b2b"):
    """A rounded box: first line large in the box colour, the rest small grey."""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" stroke="{c}" stroke-width="3"/>']
    for k, s_ in enumerate(lines):
        out.append(t(x + 32, y + 56 + k * 44, s_, 40 if k == 0 else 30, c if k == 0 else GRAY, "start", 600 if k == 0 else 400))
    return "".join(out)


def arr(x1, y1, x2, y2, c=MUT):
    """A straight arrow ending in a small head at (x2, y2)."""
    a = math.atan2(y2 - y1, x2 - x1)
    hx, hy = x2 - 18 * math.cos(a), y2 - 18 * math.sin(a)
    px, py = 9 * math.sin(a), -9 * math.cos(a)
    return (f'<line x1="{x1}" y1="{y1}" x2="{hx:.0f}" y2="{hy:.0f}" stroke="{c}" stroke-width="4"/>'
            f'<path d="M{x2} {y2} L{hx + px:.0f} {hy + py:.0f} L{hx - px:.0f} {hy - py:.0f}z" fill="{c}"/>')


# cost v10: one Stella encoder, two paths; real NQ passages and query, real vector values, one HNSW walk
_X2 = _json.loads(Path(_R + "m15_talk_slide2_example.json").read_text())["results"]
_rng = _random.Random(7)


def _vec(v, n=3):
    return "[" + ", ".join(f"{x:.3f}".replace("-", "−") for x in v[:n]) + ", …]"


def _curve(x1, y1, x2, y2, c, w=5):
    mx = (x1 + x2) / 2
    a = f'<path d="M{x1} {y1} C {mx} {y1}, {mx} {y2}, {x2 - 16} {y2}" fill="none" stroke="{c}" stroke-width="{w}"/>'
    return a + f'<path d="M{x2} {y2} l-18 -10 l0 20z" fill="{c}"/>'


g = ['<svg width="1664" height="660" viewBox="0 0 1664 660">']
# inputs: three real passages (once) and one real question (every query)
for k, p in enumerate(_X2["passages"]):
    y = 20 + k * 100
    title, rest = p["text"].split(" ", 2)[:2] if p["text"].startswith("June") else (p["text"].split(" ")[0], "")
    title = {0: "June Gable", 1: "Catch-22", 2: "Mitral valve"}[k]
    body = p["text"][len(title) + 1:][:34] + "…"
    g.append(f'<rect x="0" y="{y}" width="440" height="84" rx="12" fill="#151b2b" stroke="#3a4157" stroke-width="2"/>')
    g.append(t(20, y + 36, title, 28, WHITE, "start", 700))
    g.append(t(20, y + 68, body, 22, MUT))
g.append(t(0, 352, "2.68M passages", 30, GRAY, "start", 600))
g.append(f'<rect x="0" y="432" width="440" height="120" rx="20" fill="#1a2233" stroke="{RED}" stroke-width="3"/>')
g.append(f'<circle cx="38" cy="468" r="12" fill="none" stroke="{RED}" stroke-width="4"/><line x1="47" y1="477" x2="58" y2="488" stroke="{RED}" stroke-width="4"/>')
g.append(t(76, 478, "who played joey's agent", 27, WHITE, "start", 600))
g.append(t(76, 520, "estelle on friends", 27, WHITE, "start", 600))
# one encoder, drawn as a small network
EX, EY, EW, EH = 520, 150, 260, 330
g.append(f'<rect x="{EX}" y="{EY}" width="{EW}" height="{EH}" rx="22" fill="#11182a" stroke="{WHITE}" stroke-width="3"/>')
cols = [[(EX + 45 + c * 57, EY + 70 + r * 50) for r in range(5)] for c in range(4)]
for a, b in zip(cols, cols[1:]):
    for (x1, y1) in a:
        for (x2, y2) in b:
            g.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#3a4157" stroke-width="1.5"/>')
for col in cols:
    for (x, y) in col:
        g.append(f'<circle cx="{x}" cy="{y}" r="9" fill="{GRAY}"/>')
g.append(t(EX + EW / 2, EY + 46, "Stella", 36, WHITE, "middle", 700))
g.append(t(EX + EW / 2, EY + EH + 44, "435M parameters", 26, MUT, "middle"))
g += [_curve(440, 160, EX, 260, MUT), _curve(440, 492, EX, 400, RED)]
# outputs: real vector values
VX = 840
for k, p in enumerate(_X2["passages"]):
    g.append(t(VX, 150 + k * 46, _vec(p["vector"]), 25, GRAY, "start", 400, True))
g.append(t(VX, 72, "once, 5.3 GPU-hours", 30, GRAY, "start", 700))
g.append(t(VX, 452, "every query, 20.5 ms", 30, RED, "start", 700))
g.append(t(VX, 500, _vec(_X2["query_vector"]), 25, RED, "start", 600, True))
g += [_curve(EX + EW, 260, VX - 12, 190, MUT, 4), _curve(EX + EW, 400, VX - 12, 490, RED, 4)]
# HNSW graph: deterministic points, links to nearest neighbours, shortest walk from the entry to June Gable
GX0, GX1, GY0, GY1 = 1300, 1640, 40, 590
pts = [(1330, 500), (1560, 160)] + [(_rng.uniform(GX0, GX1), _rng.uniform(GY0, GY1)) for _ in range(40)]
pts = [p for i, p in enumerate(pts) if i < 2 or min(math.dist(p, q) for q in pts[:i]) > 48]
nb = {i: sorted(range(len(pts)), key=lambda j: math.dist(pts[i], pts[j]))[1:4] for i in range(len(pts))}
edges = {tuple(sorted((i, j))) for i in nb for j in nb[i]}
for i, j in edges:
    g.append(f'<line x1="{pts[i][0]:.0f}" y1="{pts[i][1]:.0f}" x2="{pts[j][0]:.0f}" y2="{pts[j][1]:.0f}" stroke="#3a4157" stroke-width="2"/>')
adj = {i: set() for i in range(len(pts))}
for i, j in edges:
    adj[i].add(j); adj[j].add(i)
prev, todo = {0: None}, [0]
while todo and 1 not in prev:
    i = todo.pop(0)
    for j in sorted(adj[i]):
        if j not in prev:
            prev[j] = i; todo.append(j)
path, i = [], 1
while i is not None:
    path.append(i); i = prev.get(i)
for a, b in zip(path, path[1:]):
    g.append(f'<line x1="{pts[a][0]:.0f}" y1="{pts[a][1]:.0f}" x2="{pts[b][0]:.0f}" y2="{pts[b][1]:.0f}" stroke="{RED}" stroke-width="5"/>')
for k, (x, y) in enumerate(pts):
    on = k in path
    g.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{16 if k == 1 else 9}" fill="{WHITE if k == 1 else (RED if on else "#7d849c")}"/>')
g.append(t(1560, 126, "June Gable", 30, WHITE, "middle", 700))
g.append(t(1650, 640, "Qdrant HNSW index, 0.23 ms", 28, GRAY, "end", 600))
g += [_curve(1236, 196, 1310, 250, MUT, 4), _curve(1236, 490, 1314, 500, RED, 4), "</svg>"]
write("cost", " foot", "One Encoder, Two Deadlines",
      '<div style="margin-top:8px">' + "".join(g) + "</div>",
      f"BEIR NQ, 2,681,468 passages. Real NQ test query; its judged passage ranks first in exact search (score "
      f"{_X2['passages'][0]['score']:.2f}). Vectors: first 3 of 1,024 values. Median times on the M5 Pro. Graph schematic. Exploratory.")

# arch v8: the paper's Figure 1 layout; one ingestion path, three query paths, each with its own search width
_FL = _json.loads(Path(_R + "m15_talk_flops.json").read_text())["results"]
g = ['<svg width="1664" height="660" viewBox="0 0 1664 660">',
     box(0, 0, 420, 110, WHITE, ["Documents"]), arr(420, 55, 490, 55),
     box(490, 0, 560, 110, WHITE, ["Stella document encoder"]),
     f'<path d="M1050 55 L1400 55 L1400 172" fill="none" stroke="{MUT}" stroke-width="4"/>',
     f'<path d="M1400 190 L1391 172 L1409 172z" fill="{MUT}"/>',
     f'<rect x="1150" y="190" width="514" height="470" rx="20" fill="#11182a" stroke="{WHITE}" stroke-width="3"/>',
     t(1407, 380, "One Stella index", 44, WHITE, "middle", 700),
     t(1407, 432, "same vectors", 32, GRAY, "middle"), t(1407, 474, "same HNSW graph", 32, GRAY, "middle"),
     t(0, 196, "each request picks one", 30, MUT)]
for i, (n, sub, c) in enumerate((("Stella", "25 GFLOPs per query", GRAY), ("Nano", "0.54 GFLOPs per query", VIO),
                                 ("Zero", "28 KFLOPs per query", RED))):
    y = 230 + i * 150
    g.append(box(0, y, 560, 120, c, [n, sub]))
    g.append(arr(560, y + 60, 650, y + 60))
    g.append(f'<rect x="650" y="{y + 25}" width="190" height="70" rx="35" fill="none" stroke="{c}" stroke-width="3"/>')
    g.append(t(745, y + 72, "own ef", 30, c, "middle", 600, True))
    g.append(arr(840, y + 60, 1150, 425 + (i - 1) * 60))
g.append("</svg>")
write("arch", " foot", "Three Query Encoders Search One Index",
      '<div style="margin-top:24px">' + "".join(g) + "</div>",
      "FLOPs: analytic estimate, 2·P·T + attention, median over 100 real 5-12 word queries (Stella includes its prompt); "
      "Zero: 2 FLOPs per gathered value plus normalization. ef: Qdrant's hnsw_ef, set per request.")

# order v8: what a lookup table cannot see, shown on its own pipeline
def pill(x, y, s_, c=WHITE, hot=False):
    w = 26 + 21 * len(s_)
    fill = "#3a1a26" if hot else "#1a2233"
    return (f'<rect x="{x}" y="{y}" width="{w}" height="58" rx="12" fill="{fill}" stroke="{RED if hot else "#3a4157"}" stroke-width="3"/>'
            + t(x + w / 2, y + 41, s_, 32, c, "middle", 600, True)), w


def row(x, y, toks, hot=()):
    out = []
    for s_ in toks:
        p, w = pill(x, y, s_, RED if s_ in hot else WHITE, s_ in hot)
        out.append(p)
        x += w + 14
    return "".join(out)


g = ['<svg width="1664" height="600" viewBox="0 0 1664 600">']
cols = (("Order", "a mean has no order", (["paris", "to", "rome"], ["rome", "to", "paris"]), (), "same vector"),
        ("Context", "one row per token", (["river", "bank"], ["bank", "loan"]), ("bank",), "same row for bank"),
        ("New words", "rows learned elsewhere", (["qdrant"], ["q", "##dran", "##t"]), ("q", "##dran", "##t"), "pieces, averaged"))
for i, (head, sub, (a, b), hot, out) in enumerate(cols):
    x = i * 572
    g.append(t(x, 60, head, 52, RED, "start", 700))
    g.append(t(x, 112, sub, 32, GRAY))
    g.append(row(x, 190, a, () if head == "New words" else hot))
    g.append(row(x, 290, b, hot))
    g.append(f'<line x1="{x}" y1="400" x2="{x + 500}" y2="400" stroke="{LINE}" stroke-width="2"/>')
    g.append(t(x, 470, out, 38, WHITE, "start", 700))
g.append("</svg>")
write("order", " foot", "What a Lookup Table Cannot See",
      '<div style="margin-top:24px">' + "".join(g) + "</div>",
      "Zero: Stella's WordPiece tokenizer (30,522 tokens), one learned row per token, √count-weighted mean, then L2. Token "
      "splits shown are the real tokenizer output. On 24 query pairs where direction changes the answer, Zero ranked the right "
      "passage first in both directions for 0 pairs, Stella 4, Nano 2 (results/m15_talk_word_order.json). Exploratory.",
      note="<aside>[~2 min] [Point to the three headings.] Zero is fast because it skips the transformer. Here is what that costs. "
      "Zero does three things: it splits the query into tokens, looks up one row per token, and averages the rows.\n\n"
      "[Order.] An average has no order. Paris to Rome and Rome to Paris have the same tokens, so Zero builds exactly the same "
      "vector and returns the same list. I tested 24 pairs where direction changes the answer. Zero got none of them right in "
      "both directions. Stella got four, so this is hard for everyone, but Zero cannot win one.\n\n[Context.] Each token has one "
      "row, whatever sits next to it. Bank in river bank and bank in bank loan is the same row. A transformer mixes tokens with "
      "attention, so its bank depends on its neighbours. Zero only has the other rows in the average to separate the two.\n\n"
      "[New words.] The vocabulary is fixed at 30,522 pieces. A word it has never seen, like qdrant, splits into pieces whose "
      "rows were learned from other words. A transformer can put those pieces back together; Zero averages them.\n\nSo Zero "
      "works best on short keyword-like queries, and I'd test longer, phrased questions separately.</aside>")

# split v8: Stella's bar runs off the slide; one millisecond axis for all three
W, L0, AX = 1664, 230, 2.2
SX = lambda v: L0 + v / AX * 820
g = [f'<svg width="{W}" height="560" viewBox="0 0 {W} 560" style="overflow:visible">',
     f'<rect x="{SX(0)}" y="10" width="28" height="20" rx="4" fill="{TEAL}"/>', t(SX(0) + 38, 30, "encode", 26, MUT),
     f'<rect x="{SX(0) + 190}" y="10" width="28" height="20" rx="4" fill="{ORG}"/>', t(SX(0) + 228, 30, "search", 26, MUT)]
for v in (0, 1, 2):
    g.append(f'<line x1="{SX(v)}" x2="{SX(v)}" y1="70" y2="480" stroke="{LINE}" stroke-width="2"/>')
    g.append(t(SX(v), 520, str(v), 26, MUT, "middle", 400, True))
g.append(t(SX(2.2), 520, "ms", 26, MUT, "end", 400, True))
y = 380
g.append(t(0, y + 54, "Stella", 44, GRAY, "start", 600))
g.append(f'<rect x="{SX(0)}" y="{y + 10}" width="2000" height="64" fill="{TEAL}"/>')
g.append(t(1640, y + 56, "20.8 ms  →", 40, "#0b0f19", "end", 700))
for i, (n, tot, enc, c) in enumerate([("Nano", 2.04, 0.807, VIO), ("Zero", 0.86, 0.041, RED)]):
    y = 80 + i * 150
    g.append(t(0, y + 54, n, 44, c, "start", 600))
    g.append(f'<rect x="{SX(0)}" y="{y + 10}" width="{SX(tot * enc) - SX(0):.0f}" height="64" fill="{TEAL}"/>')
    g.append(f'<rect x="{SX(tot * enc):.0f}" y="{y + 10}" width="{SX(tot) - SX(tot * enc):.0f}" height="64" fill="{ORG}"/>')
    g.append(t(SX(tot) + 18, y + 56, f"{tot} ms", 40, WHITE, "start", 600))
g.append("</svg>")
write("split", " foot", "The Whole Request Is 2.4× Faster, Not 51×",
      '<div style="margin-top:32px">' + "".join(g) + "</div>",
      "BEIR NQ, 3,452 queries, 2.68M passages, one Qdrant Edge shard in-process, M5 Pro, sequential. Segments: median encode "
      "and search shares of the median request. Each encoder at its fastest setting within 1% of its own exact nDCG@10 "
      "(Stella 0.631, Nano 0.579); Zero misses that at every tested point (best shown: ef 1024, os 8, nDCG@10 0.510). Exploratory.")

# recipe v8: how auto-routing decides, in three steps
_ED = _json.loads(Path(_R + "m15_talk_edge_nq.json").read_text())["results"]
_RP = _json.loads(Path(_R + "m15_talk_router_plan.json").read_text())
nd = lambda n, ef: next(s["ndcg10"] for s in _ED[n]["settings"] if s["ef"] == ef and s["oversampling"] == 4.0)
g = ['<svg width="1664" height="640" viewBox="0 0 1664 640">']
for i, (num, head) in enumerate((("01", "Measure"), ("02", "Set"), ("03", "Route"))):
    x = i * 590
    g.append(f'<line x1="{x}" y1="0" x2="{x + 500}" y2="0" stroke="{RED}" stroke-width="4"/>')
    g.append(t(x, 56, num, 30, RED, "start", 600, True))
    g.append(t(x + 70, 58, head, 48, WHITE, "start", 700))
for k, ef in enumerate((128, 256, 512)):                      # 01: the measured menu
    g.append(t(200 + k * 120, 150, str(ef), 28, MUT, "middle", 400, True))
for j, (n, c) in enumerate((("stella", GRAY), ("nano", VIO), ("zero", RED))):
    yy = 210 + j * 64
    g.append(t(0, yy, n.capitalize(), 34, c, "start", 600))
    for k, ef in enumerate((128, 256, 512)):
        g.append(t(200 + k * 120, yy, f"{nd(n, ef):.3f}"[1:], 32, WHITE, "middle", 400, True))
g.append(t(0, 150, "ef", 28, MUT, "start", 400, True))
g.append(t(0, 420, "nDCG@10 on judged queries", 30, GRAY))
g.append(t(0, 462, "+ encode and search time", 30, GRAY))
for k, (lab, val) in enumerate((("quality floor", "0.50"), ("p95 target", "150 ms"))):   # 02: the two rules
    yy = 110 + k * 170
    g.append(f'<rect x="590" y="{yy}" width="420" height="140" rx="16" fill="#151b2b" stroke="{GRAY}" stroke-width="3"/>')
    g.append(t(622, yy + 50, lab, 30, GRAY))
    g.append(t(622, yy + 112, val, 56, WHITE, "start", 700, True))
g.append(t(590, 462, "quality first, then latency", 30, GRAY))
p15 = next(p for p in _RP["plans"] if p["rate"] == 1500)        # 03: one plan, re-made every 250 ms
g.append(t(1180, 200, "every 250 ms", 52, WHITE, "start", 700))
g.append(t(1180, 262, "at 1,500 searches/s:", 30, GRAY))
bx, bw = 1180, 484
for n, c in (("nano", VIO), ("zero", RED)):
    w = bw * p15["shares"][n]
    g.append(f'<rect x="{bx:.0f}" y="290" width="{w:.0f}" height="70" fill="{c}"/>')
    g.append(t(bx + w / 2, 337, n.capitalize(), 30, "#0b0f19", "middle", 700))
    bx += w
g.append(t(1180, 410, f"ef {p15['ef']}", 34, WHITE, "start", 600, True))
g.append(t(1180, 462, "best average nDCG that fits", 30, GRAY))
g.append("</svg>")
write("recipe", " foot", "How the Router Decides",
      '<div style="margin-top:40px">' + "".join(g) + "</div>",
      "Quality: exact-judged NQ queries, Qdrant Edge, oversampling 4 (results/m15_talk_edge_nq.json). Floor and target: the "
      "studio defaults. Plan: allocator.py on this laptop's measured encode capacity (results/m15_talk_router_plan.json). Exploratory.",
      note="<aside>[~2 min] Here's how the router you're about to see makes its choice. Three steps.\n\n[01 Measure.] Before "
      "any traffic arrives, I score every encoder at every search width on judged queries. That's this menu: the ranking "
      "quality I get from each pair. At startup the studio also times each encoder and each search on this machine.\n\n"
      "[02 Set.] Then I give it two rules, in order. Quality first: the average score of what it serves stays above the floor. "
      "Then latency: stay under the p95 target if the machine can.\n\n[03 Route.] Every quarter of a second it looks at the "
      "traffic and picks the mix of encoders and one search width that gives the best average quality and still fits on this "
      "machine. At 1,500 searches a second that's about three quarters Nano, one quarter Zero, at ef 512.\n\nThis only works "
      "because all three search one index. Switching an encoder costs nothing: no second index, no re-encode. To do this on "
      "your data you need the same three things: a few hundred judged queries from your traffic, the quality of each encoder "
      "and ef on them, and timings on your hardware.</aside>")

# plan: what the router picks as traffic grows (planner output)
W, H, L, R, T, B = 1664, 600, 120, 40, 90, 90
lo, hi = math.log10(40), math.log10(2500)
PX = lambda r: L + (math.log10(r) - lo) / (hi - lo) * (W - L - R)
PY = lambda v: T + (1 - v) * (H - T - B)
P_ = _RP["plans"]
g = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
base = [0.0] * len(P_)
for n, c in (("stella", GRAY), ("nano", VIO), ("zero", RED)):
    top = [b + p["shares"][n] for b, p in zip(base, P_)]
    pts = [f"{PX(p['rate']):.0f},{PY(v):.0f}" for p, v in zip(P_, top)]
    pts += [f"{PX(p['rate']):.0f},{PY(v):.0f}" for p, v in reversed(list(zip(P_, base)))]
    g.append(f'<polygon points="{" ".join(pts)}" fill="{c}"/>')
    base = top
g.append(t(PX(55), PY(0.45), "Stella", 44, "#0b0f19", "start", 700))
g.append(t(PX(600), PY(0.45), "Nano", 44, "#0b0f19", "middle", 700))
g.append(t(PX(2350), PY(0.82), "Zero", 44, "#0b0f19", "end", 700))
spans = {}
for p in P_:
    spans.setdefault(p["ef"], []).append(p["rate"])
efs = list(spans)
edge = [PX(P_[0]["rate"])] + [PX(math.sqrt(spans[a][-1] * spans[b][0])) for a, b in zip(efs, efs[1:])] + [PX(P_[-1]["rate"])]
for k, ef in enumerate(efs):                                   # one search width per span of traffic, split at the switch
    a, b = edge[k] + 4, edge[k + 1] - 4
    g.append(f'<line x1="{a:.0f}" x2="{b:.0f}" y1="{T - 40}" y2="{T - 40}" stroke="{WHITE}" stroke-width="4"/>')
    last = k == len(efs) - 1
    g.append(t(W if last else (a + b) / 2, T - 56, f"ef {ef}" if k == 0 else str(ef), 30, WHITE, "end" if last else "middle", 600, True))
for r in (40, 100, 300, 1000, 2500):
    g.append(t(PX(r), H - B + 44, f"{r:,}", 30, MUT, "end" if r == 2500 else "middle", 400, True))
g.append(t((L + W - R) / 2, H - 4, "searches per second", 30, GRAY, "middle"))
for v in (0, 0.5, 1):
    g.append(t(L - 16, PY(v) + 10, f"{v:.0%}", 28, MUT, "end", 400, True))
g.append("</svg>")
q0, q1 = P_[0]["quality"], P_[-1]["quality"]
write("plan", " foot", "Quiet Hours Get Stella, Peaks Get Zero",
      '<div style="margin-top:16px">' + "".join(g) + "</div>",
      f"Share of traffic per encoder planned by allocator.py: NQ quality and search time per preset, M5 Pro encode capacity "
      f"(Stella 80, Nano 1,400, Zero 17,000 encodes/s), floor 0.50, headroom 0.8. Average nDCG@10 falls from {q0:.3f} to "
      f"{q1:.3f}. Planner output, not a load test (results/m15_talk_router_plan.json). Exploratory.",
      note=f"<aside>[~1 min] This is the plan the router makes for this laptop, from quiet to busy. [Point left.] At 40 "
      f"searches a second, everything goes to Stella with the widest search. [Point to the purple.] From about a hundred a "
      f"second Stella can't keep up, so Nano takes over. [Point right.] Past a thousand, Zero mixes in and the search narrows. "
      f"The average quality drops from {q0:.2f} to {q1:.2f}, and it never goes below the floor I set. Nothing is dropped and "
      f"nothing is re-encoded. Let me show you it running.</aside>")

# takeaways: four lines to remember
_TK = (("Measure the whole request", "not just the encoder"), ("Pick the encoder and its ef together", "cheaper vectors need a wider search"),
       ("Set the quality floor first", "it decides which encoder ships"), ("One index, any query encoder", "so every request can choose"))
g = ['<svg width="1664" height="640" viewBox="0 0 1664 640">']
for i, (a, b) in enumerate(_TK):
    y = 60 + i * 150
    g.append(t(0, y + 40, f"0{i + 1}", 44, RED, "start", 700, True))
    g.append(t(120, y + 40, a, 52, WHITE, "start", 700))
    g.append(t(120, y + 92, b, 34, GRAY))
g.append("</svg>")
write("takeaways", "", "What to Take Home", '<div style="margin-top:24px">' + "".join(g) + "</div>")

# byo: already have a frozen index? train a Zero or Nano for it (details in the paper)
_ST = (("Keep your document encoder and index", "nothing is re-encoded"),
       ("Encode real queries with your query model", "its vectors are the training targets"),
       ("Train Zero, Nano, or both", "Zero ≈ 20 min on one GPU, Nano ≈ $95"),
       ("Judge them on your own queries", "exact nDCG@10, then sweep ef"))
g = ['<svg width="1180" height="620" viewBox="0 0 1180 620">']
for i, (a, b) in enumerate(_ST):
    y = 40 + i * 150
    g.append(t(0, y + 40, f"0{i + 1}", 44, RED, "start", 700, True))
    g.append(t(110, y + 40, a, 46, WHITE, "start", 700))
    g.append(t(110, y + 90, b, 32, GRAY))
g.append("</svg>")
_q = _qr("paper")
write("byo", " foot", "Already Have an Index? Train Your Own",
      '<div class="row" style="gap:80px;margin-top:24px;align-items:center">' + "".join(g)
      + f'<div class="col" style="gap:20px;align-items:center"><div style="background:#fff;border-radius:16px;padding:12px">{_q}</div>'
      '<p style="font-size:34px;font-weight:600;color:#fcfdff">Full recipe: the paper</p></div></div>',
      "Costs are Constella's, for a Stella index: Zero trained on one RTX 3080 once its targets existed (computing Stella's "
      "targets took 8-12 h); Nano took 57.3 A100-hours. Training sources must allow commercial use of derived weights. Exploratory.")

# Speaker notes: talking_points.md is the one source; it replaces every note written above.
_TP = (Path(__file__).parent / "talking_points.md").read_text()
for sec in re.split(r"^## ", _TP, flags=re.M)[1:]:
    head, body = sec.split("\n", 1)
    sid, mins = [x.strip() for x in head.split("·")]
    f = S / f"{sid}.html"
    html = f.read_text()
    text = f"[{mins}]\n" + body.strip().replace("&", "&amp;").replace("<", "&lt;")
    note = "<aside>" + text.replace("\n", "<br>\n") + "</aside>"       # the deck page shows <br> as a line break
    assert html.count("<aside>") == 1, sid
    f.write_text(re.sub(r"<aside>.*?</aside>", lambda m: note, html, flags=re.S))

for f in sorted(S.glob("*.html")):                             # backup notes not in talking_points.md: same line breaks
    html = f.read_text()
    m = re.search(r"<aside>(.*?)</aside>", html, re.S)
    if m and "<br>" not in m.group(1) and "\n" in m.group(1).strip():
        f.write_text(html.replace(m.group(0), "<aside>" + m.group(1).strip().replace("\n", "<br>\n") + "</aside>"))

for sid in ("plan", "solved", "takeaways"):                                  # backup slides stay out of the main run
    f = S / f"{sid}.html"
    f.write_text(re.sub(r'<section id="(\w[\w-]*)" class="([^"]*)"[^>]*>', r'<section id="\1" class="\2" hidden>', f.read_text(), count=1))

print("rebuilt")
