"""Rebuild the dense slides as visuals. Keeps each slide's <aside>; Sol rewrites notes afterwards."""
import math
import re
from pathlib import Path

S = Path(__file__).resolve().parents[1] / "deck/project/slides"
GRAY, VIO, RED, TEAL, ORG = "#c9cede", "#9494ff", "#ff516b", "#3fc1a5", "#f5a623"
MUT, LINE, WHITE = "#9aa0b5", "#303547", "#fcfdff"
MONO = "font-family:JetBrains Mono,monospace"


# The merged distill note is built from the committed original, so reruns stay idempotent.
ORIG_DISTILL = re.search(r"<aside>.*?</aside>", Path("/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/deck/project/slides/distill.html").read_text(), re.S).group(0)


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


# 1. cost: one block paid once, a stream of query ticks paid forever
g = ['<svg width="1664" height="560" viewBox="0 0 1664 560">']
g.append(f'<rect x="0" y="100" width="440" height="200" rx="16" fill="#1d2433" stroke="{GRAY}" stroke-width="3"/>')
g.append(t(220, 170, "2.68M passages", 34, WHITE, "middle", 600))
g.append(t(220, 222, "5.3 GPU-hours", 34, WHITE, "middle", 600))
g.append(t(220, 270, "once", 30, GRAY, "middle"))
for i in range(60):
    x = 490 + i * 19.5
    h = 200
    g.append(f'<rect x="{x:.0f}" y="100" width="10" height="{h}" rx="3" fill="{RED}" opacity="{0.35 + 0.65 * (i % 5 == 0)}"/>')
g.append(t(490, 350, "Stella per query: 31.6 ms · 25 GFLOPs", 34, WHITE, "start", 600))
g.append(t(490, 396, "1M queries ≈ 10.3 hours of encoding", 30, GRAY))
g.append(f'<line x1="0" y1="450" x2="1664" y2="450" stroke="{LINE}" stroke-width="2"/>')
g.append(t(0, 516, "C(t) = N · c_D + Q(t) · (c_Q + c_search)", 40, WHITE, "start", 400, True))
g.append("</svg>")
write("cost", " foot", "Encode the Corpus Once, Pay for Every Query",
      '<div style="margin-top:40px">' + "".join(g) + "</div>",
      "2,681,468 NQ passages, Stella fp32, one RTX 3080, steady 141 passages/s. Query: Stella with its prompt, "
      "median of 31 tokens, M5 Pro CPU, 4 threads, p50; 10.3 h uses the mean (37 ms).")

# 2. arch: three encoders converge on one index; FLOPs as the proof
g = ['<svg width="1664" height="640" viewBox="0 0 1664 640">']
enc = [("Stella", "435M-parameter transformer", "25 GFLOPs / query", GRAY, 40),
       ("Nano", "34.5M · bge-small + linear head", "0.54 GFLOPs / query", VIO, 250),
       ("Zero", "token lookup table · 30,522 × 1024", "28 KFLOPs / query", RED, 460)]
for name, what, fl, c, y in enc:
    g.append(f'<rect x="0" y="{y}" width="620" height="150" rx="16" fill="#151b2b" stroke="{c}" stroke-width="3"/>')
    g.append(t(32, y + 58, name, 44, c, "start", 600))
    g.append(t(32, y + 100, what, 28, GRAY))
    g.append(t(32, y + 134, fl, 26, WHITE, "start", 600, True))
    g.append(f'<path d="M620 {y + 75} C 820 {y + 75}, 860 320, 1010 320" fill="none" stroke="{c}" stroke-width="4"/>')
g.append(f'<rect x="1010" y="170" width="654" height="300" rx="20" fill="#11182a" stroke="{WHITE}" stroke-width="3"/>')
g.append(t(1337, 240, "One Stella index", 44, WHITE, "middle", 600))
g.append(t(1337, 296, "2.68M vectors · 1024-d", 30, GRAY, "middle"))
g.append(t(1337, 342, "HNSW · binary codes in RAM", 30, GRAY, "middle"))
g.append("</svg>")
write("arch", " foot", "Three Query Encoders Search One Index",
      '<div style="margin-top:24px">' + "".join(g) + "</div>",
      "FLOPs: analytic estimate, 2·P·T + attention, median over 100 real 5-12 word queries (Stella includes its prompt); "
      "Zero: 2 FLOPs per gathered value plus normalization.")

# 3. cloud: CPU per million queries (machine-normalized), dollars as labels, ratios as the outcome
g = ['<svg width="1664" height="560" viewBox="0 0 1664 560">']
rows = [("Stella", 123.0, "123 vCPU-h · $6.07", GRAY), ("Nano", 4.786, "4.8 vCPU-h · $0.24", VIO),
        ("Zero", 0.03296, "2 vCPU-min · $0.0016", RED)]
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
     box(430, 380, "1 · Align", ["match Stella's", "vectors and scores"], WHITE), arrow(822),
     box(880, 380, "2 · Rank", ["positives vs", "32,768 negatives"], WHITE), arrow(1272),
     box(1330, 334, "31 MB table", ["int8 rows", "+ row scales"], RED),
     t(0, 360, "align: 1 − cos(z, Ê_S) + KL(p_S ‖ p_z)  ·  rank: InfoNCE  ·  Ê_D(d) never changes", 30, WHITE, "start", 400, True),
     "</svg>"]
chips = ('<div class="chips" style="margin-top:28px;gap:24px;white-space:nowrap"><p class="chip">RTX 3080 · 10 GB</p>'
         '<p class="chip red">≈ 20 min with prepared targets</p></div>')
write("distill", " foot", "Zero Trains on a Desktop GPU Against a Frozen Index",
      '<div style="margin-top:32px">' + "".join(g) + "</div>" + chips,
      "Zero: phase 1 16,000 steps × 512, phase 2 2,500 steps InfoNCE, τ 0.02; retrain time is a historical estimate; "
      "a new teacher needs 8-12 h of target encoding. All training sources permit commercial derived weights.",
      note=ORIG_DISTILL.replace("</aside>", " " + re.sub(r"</?aside>", "", aside("zero-train")) + "</aside>"))

# 5. nano: three layers feed one projection into Stella's space
g = ['<svg width="1664" height="520" viewBox="0 0 1664 520">']
for k in range(12):
    y = 440 - k * 34
    hot = k + 1 in (4, 8, 12)
    g.append(f'<rect x="0" y="{y}" width="300" height="26" rx="5" fill="{VIO if hot else "#232a3d"}"/>')
    if hot:
        g.append(t(318, y + 21, f"L{k + 1}", 24, VIO, "start", 600, True))
        g.append(f'<path d="M370 {y + 13} C 480 {y + 13}, 520 260, 600 260" fill="none" stroke="{VIO}" stroke-width="3"/>')
g.append(t(150, 500, "bge-small · 12 × 384", 26, GRAY, "middle"))
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
g.append(t(750, 400, "+0.0217 dev nDCG@10 vs last layer only", 28, VIO, "middle", 600))
g.append("</svg>")
write("nano", " foot", "Nano Reads Three Layers to Reach Stella's Space",
      '<div style="margin-top:32px">' + "".join(g) + "</div>"
      '<div class="chips" style="gap:24px"><p class="chip vio">34.5M parameters</p><p class="chip">3.49M training queries</p>'
      '<p class="chip">57.3 A100-hours ≈ $95, once</p></div>',
      "Loss: ‖f(x) − Ê_S(x)‖² on queries (query targets) and passages (document targets), 3:1. Head initialized by ridge. "
      "No relevance labels; MS MARCO used for validation only.")

# 6. bench: Constella against models that need their own index; absolute BEIR-15 nDCG@10
W, H, L, R, T, B = 1664, 600, 140, 80, 30, 90
xl, xh, yl, yh = math.log10(0.02), math.log10(60), 0.38, 0.58
PX = lambda v: L + (math.log10(v) - xl) / (xh - xl) * (W - L - R)
PY = lambda v: T + (yh - v) / (yh - yl) * (H - T - B)
g = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
for v in (0.40, 0.45, 0.50, 0.55):
    g.append(f'<line x1="{L}" x2="{W - R}" y1="{PY(v):.0f}" y2="{PY(v):.0f}" stroke="{LINE}" stroke-width="2"/>')
    g.append(t(L - 16, PY(v) + 9, f"{v:.2f}", 26, MUT, "end", 400, True))
for d, lab in ((0.03, "0.03"), (0.3, "0.3"), (3, "3"), (30, "30")):
    g.append(t(PX(d), H - B + 44, lab, 26, MUT, "middle", 400, True))
g.append(t((L + W - R) / 2, H - 8, "median query encode time, ms (log)", 26, GRAY, "middle"))
g.append(t(L, 22, "BEIR-15 nDCG@10", 26, GRAY))
g.append(f'<line x1="{L}" x2="{W - R}" y1="{PY(0.4006):.0f}" y2="{PY(0.4006):.0f}" stroke="{MUT}" stroke-width="3" stroke-dasharray="4 10"/>')
g.append(t(W - R, PY(0.4006) - 14, "BM25 · 0.401", 28, MUT, "end"))
for n, x, y, dx, dy, anc in (("LEAF", 1.39, 0.5402, 0, -30, "middle"), ("bge-small", 2.57, 0.5171, 34, 10, "start")):
    g.append(f'<rect x="{PX(x) - 13:.0f}" y="{PY(y) - 13:.0f}" width="26" height="26" fill="none" stroke="{MUT}" stroke-width="4" transform="rotate(45 {PX(x):.0f} {PY(y):.0f})"/>')
    g.append(t(PX(x) + dx, PY(y) + dy, f"{n} · own index", 26, MUT, anc))
pts = [("Stella", 31.6, 0.5614, GRAY, -34), ("Nano", 2.25, 0.5081, VIO, 62), ("Zero", 0.044, 0.4572, RED, 62)]
for n, x, y, c, dy in pts:
    g.append(f'<circle cx="{PX(x):.0f}" cy="{PY(y):.0f}" r="18" fill="{c}"/>')
    g.append(t(PX(x), PY(y) + dy, n, 34, c, "middle", 600))
g.append(t(PX(0.044) + 34, PY(0.4933) + 10, "+ BM25 fusion · 0.493", 28, RED, "start", 600))
g.append(f'<line x1="{PX(0.044):.0f}" y1="{PY(0.4572) - 20:.0f}" x2="{PX(0.044):.0f}" y2="{PY(0.4933) + 6:.0f}" stroke="{RED}" stroke-width="3" stroke-dasharray="6 6"/>')
g.append("</svg>")
write("bench", " foot", "One Index, Three Price Points",
      '<div style="margin-top:24px">' + "".join(g) + "</div>",
      "Exact search, nDCG@10, unweighted mean of 15 BEIR datasets (registered descriptive evaluation). Circles share "
      "Stella's index; diamonds use their own index. Encode: M5 Pro CPU p50. Fusion adds a sparse branch (DBSF over "
      "100 candidates per branch).")

# 7. split: whole-request bars on one millisecond axis
W, L0, AX = 1664, 230, 2.2
SX = lambda v: L0 + v / AX * 820
g = [f'<svg width="{W}" height="580" viewBox="0 0 {W} 580">']
g.append(t(SX(0), 30, "median request, ms", 26, MUT))
g.append(f'<rect x="{SX(0) + 300}" y="10" width="28" height="20" rx="4" fill="{TEAL}"/>')
g.append(t(SX(0) + 338, 30, "encode", 26, MUT))
g.append(f'<rect x="{SX(0) + 490}" y="10" width="28" height="20" rx="4" fill="{ORG}"/>')
g.append(t(SX(0) + 528, 30, "search", 26, MUT))
g.append(t(1420, 30, "nDCG@10", 26, MUT))
for i, (n, tot, enc, nd, c, star) in enumerate([("Nano", 2.04, 0.807, "0.579", VIO, "99.2% of exact"),
                                                ("Zero", 0.86, 0.041, "0.510", RED, "98.5% of exact")]):
    y = 80 + i * 150
    g.append(t(0, y + 54, n, 44, c, "start", 600))
    g.append(f'<rect x="{SX(0)}" y="{y + 10}" width="{SX(tot * enc) - SX(0):.0f}" height="64" fill="{TEAL}"/>')
    g.append(f'<rect x="{SX(tot * enc):.0f}" y="{y + 10}" width="{SX(tot) - SX(tot * enc):.0f}" height="64" fill="{ORG}"/>')
    g.append(t(SX(tot) + 18, y + 56, f"{tot} ms", 40, WHITE, "start", 600))
    g.append(t(1420, y + 56, nd, 40, WHITE, "start", 600))
    g.append(t(1420, y + 100, star, 32 if star.startswith("98") else 26, RED if star.startswith("98") else GRAY, "start", 600 if star.startswith("98") else 400))
y = 440
g.append(t(0, y + 54, "Stella", 44, GRAY, "start", 600))
g.append(t(SX(0), y + 54, "20.8 ms (off scale)", 34, GRAY))
g.append(t(1420, y + 56, "0.631", 40, WHITE, "start", 600))
g.append(t(1420, y + 96, "99.0% of exact", 26, GRAY))
for v in (0, 1, 2):
    g.append(f'<line x1="{SX(v)}" x2="{SX(v)}" y1="80" y2="360" stroke="{LINE}" stroke-width="2"/>')
    g.append(t(SX(v), 395, str(v), 24, MUT, "middle", 400, True))
g.append("</svg>")
write("split", " foot", "The Whole Request Is 2.4× Faster, Not 51×",
      '<div style="margin-top:32px">' + "".join(g) + "</div>",
      "BEIR NQ, 3,452 queries, 2.68M passages, one Qdrant Edge shard in-process, M5 Pro, sequential. Segments: median encode "
      "and search shares of the median request. Each encoder at its fastest setting within 1% of its own exact nDCG@10; "
      "Zero misses that at every tested point (best shown: ef 1024, os 8). Exploratory.")
# 8. zero-mech: real rows Zero gathers for one query, pooled, next to Stella's vector
import json as _json
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

# 9. order: the angle between a query and its reversal, per encoder
oc = SD["order_cosine"]
ang = {k: math.degrees(math.acos(min(1.0, v["median_cos"]))) for k, v in oc.items()}
g = ['<svg width="1664" height="560" viewBox="0 0 1664 560">']
for i, (k, n, c) in enumerate((("stella-query", "Stella", GRAY), ("nano", "Nano", VIO), ("zero", "Zero", RED))):
    ox, oy, Lr = 60 + i * 560, 470, 440
    a = math.radians(ang[k])
    g.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + Lr}" y2="{oy}" stroke="{c}" stroke-width="6"/>')
    g.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + Lr * math.cos(a):.1f}" y2="{oy - Lr * math.sin(a):.1f}" stroke="{c}" stroke-width="6" opacity="0.55"/>')
    g.append(t(ox, 380, n, 44, c, "start", 600))
    g.append(t(ox, 300, f"{ang[k]:.1f}°", 88, WHITE, "start", 700))
g.append(t(0, 60, "\"paris to rome\" vs \"rome to paris\": median angle between the two query vectors, 24 pairs", 30, GRAY))
g.append("</svg>")
write("order", " foot", "Reverse the Query, and Zero's Vector Does Not Move",
      '<div style="margin-top:32px">' + "".join(g) + "</div>",
      "Median over 24 frozen reversal pairs (m15/talk/word_order_pairs.json) of the angle between the two query "
      "vectors; Zero's are identical. results/m15_talk_slide_data.json. Exploratory.")

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
write("live", " demo", "Live: Four Rankers on One Shard",
      '<div class="live"><div class="dot"></div><p>LIVE</p></div>'
      '<p class="sub" style="font-size:40px;margin-top:24px">Stella, Nano, Zero and BM25 on one Qdrant Edge shard, on this laptop.</p>')

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

# cost v5: the index as a field of passages, queries arriving forever, formula terms colored to match
_rng = _random.Random(7)
g = ['<svg width="1664" height="620" viewBox="0 0 1664 620">']
cx, cy, rad = 330, 230, 200
for _ in range(420):
    a, r = _rng.uniform(0, 2 * math.pi), rad * math.sqrt(_rng.random())
    g.append(f'<circle cx="{cx + r * math.cos(a):.0f}" cy="{cy + r * math.sin(a):.0f}" r="4" fill="{GRAY}" opacity="0.55"/>')
g.append(t(cx, 480, "2.68M passages", 36, WHITE, "middle", 600))
g.append(t(cx, 524, "5.3 GPU-hours, once", 32, GRAY, "middle"))
for k in range(9):
    y = 70 + k * 40
    x0 = 1250 - (k % 3) * 60
    g.append(f'<line x1="{x0}" y1="{y}" x2="{cx + rad + 40}" y2="{cy + (y - cy) * 0.35:.0f}" stroke="{RED}" stroke-width="3" opacity="{0.35 + 0.08 * k:.2f}"/>')
    g.append(f'<circle cx="{x0}" cy="{y}" r="7" fill="{RED}"/>')
g.append(t(1290, 210, "every query", 36, RED, "start", 600))
g.append(t(1290, 254, "31.6 ms · 25 GFLOPs", 32, WHITE))
g.append(f'<text x="832" y="600" font-size="48" fill="{WHITE}" text-anchor="middle" style="{MONO}">C(t) = '
         f'<tspan fill="{GRAY}">N · c_D</tspan> + <tspan fill="{RED}">Q(t) · (c_Q + c_search)</tspan></text>')
g.append("</svg>")
write("cost", " foot", "Encode the Corpus Once, Pay for Every Query",
      '<div style="margin-top:16px">' + "".join(g) + "</div>",
      "2,681,468 NQ passages, Stella fp32, one RTX 3080, steady 141 passages/s. Query: Stella with its prompt, median of "
      "31 tokens, M5 Pro CPU, 4 threads, p50. N passages, c_D per passage, Q(t) queries by time t.")

# zero-mech v5: token rows on the left, one arrow, the result on the right
g = ['<svg width="1664" height="640" viewBox="0 0 1664 640">']
for i, r in enumerate(ex["rows"]):
    y = 30 + i * 62
    g.append(t(190, y + 34, r["token"], 32, MUT if r["token"].startswith("[") else WHITE, "end", 600, True))
    g.append(strip(215, y, r["row"], w=700))
g.append(t(215, 500, "one table row per token", 28, MUT))
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

# order v5: cosine on a zoomed axis
g = ['<svg width="1664" height="560" viewBox="0 0 1664 560">']
x0, x1, lo, hi = 120, 1560, 0.990, 1.000
CX = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
g.append(f'<line x1="{x0}" y1="380" x2="{x1}" y2="380" stroke="{LINE}" stroke-width="4"/>')
for v in (0.990, 0.992, 0.994, 0.996, 0.998, 1.000):
    g.append(f'<line x1="{CX(v):.0f}" y1="370" x2="{CX(v):.0f}" y2="390" stroke="{MUT}" stroke-width="3"/>')
    g.append(t(CX(v), 434, f"{v:.3f}", 30, MUT, "middle", 400, True))
g.append(t((x0 + x1) / 2, 500, "cosine between a query and its reversal (median of 24 pairs)", 30, GRAY, "middle"))
for k, n, c, ylab in (("stella-query", "Stella", GRAY, 250), ("nano", "Nano", VIO, 150), ("zero", "Zero", RED, 250)):
    v = oc[k]["median_cos"]
    g.append(f'<line x1="{CX(v):.0f}" y1="{ylab + 20}" x2="{CX(v):.0f}" y2="368" stroke="{c}" stroke-width="4"/>')
    g.append(f'<circle cx="{CX(v):.0f}" cy="380" r="16" fill="{c}"/>')
    lab = "1 (identical)" if k == "zero" else f"{v:.4f}"
    g.append(t(CX(v), ylab - 36, n, 40, c, "middle" if k != "zero" else "end", 600))
    g.append(t(CX(v), ylab + 4, lab, 32, WHITE, "middle" if k != "zero" else "end", 600))
g.append(t(x0, 60, "\"cheapest flights from Paris to Rome\" ↔ \"… Rome to Paris\"", 32, WHITE, "start", 400, True))
g.append("</svg>")
write("order", " foot", "Reverse the Query, and Zero's Vector Does Not Move",
      '<div style="margin-top:24px">' + "".join(g) + "</div>",
      "24 frozen reversal pairs (m15/talk/word_order_pairs.json); Stella with its prompt. results/m15_talk_slide_data.json. Exploratory.")

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
g.append(t(L, 34, "share of its own exact top 10 found · NQ, 2.68M passages", 30, GRAY))
g.append(f'<line x1="{RX(0)}" y1="{RY(.913):.0f}" x2="{RX(5)}" y2="{RY(.913):.0f}" stroke="{WHITE}" stroke-width="2" stroke-dasharray="8 8"/>')
for n, (ys, c) in REC.items():
    g.append(f'<polyline points="{" ".join(f"{RX(i):.0f},{RY(v):.0f}" for i, v in enumerate(ys))}" fill="none" stroke="{c}" stroke-width="6"/>')
    g.append(t(RX(5) + 24, RY(ys[-1]) + 12, n, 36, c, "start", 600))
for i, v in ((0, .913), (5, .901)):
    g.append(f'<circle cx="{RX(i):.0f}" cy="{RY(v):.0f}" r="22" fill="none" stroke="{WHITE}" stroke-width="4"/>')
g.append(t(RX(3), RY(0.80), "Zero at ef 1024: 0.901", 34, RED, "middle", 600))
g.append(t(RX(3), RY(0.80) + 46, "Stella at ef 32: 0.913", 34, WHITE, "middle", 600))
g.append("</svg>")
write("recovery", " foot", "Zero Needs a Wider Search, and Still Trails",
      '<div style="margin-top:16px">' + "".join(g) + "</div>",
      "Qdrant Edge, HNSW m=16, ef_construct=100, binary quantization with fp16 rescoring, one build, M5 Pro. "
      "Across 25 models in the paper, lookup tables needed a median ef of 256 to match their teacher at 64. Exploratory.")

# recipe v5: the method only; the worked example moves to "solved"
write("recipe", "", "The Monday Recipe",
      '<div class="fxbox" style="margin-top:24px;padding:36px 48px">'
      '<p class="fx" style="font-size:40px;white-space:pre">(e, ef, os)* = argmin p50<span class="mut">_total</span></p>'
      '<p class="fx" style="font-size:40px;white-space:pre">        s.t.  nDCG@10 ≥ <span class="tr">floor</span>,  p95<span class="mut">_total</span> ≤ <span class="tr">budget</span></p></div>'
      '<div class="steps" style="margin-top:56px">'
      '<div class="step"><p class="sn">01</p><p class="st">Judge queries</p><p class="sd">sampled from your traffic</p></div>'
      '<div class="step"><p class="sn">02</p><p class="st">Exact nDCG@10</p><p class="sd">each encoder, your index</p></div>'
      '<div class="step"><p class="sn">03</p><p class="st">Sweep ef × os</p><p class="sd">time encode + search</p></div>'
      '<div class="step"><p class="sn">04</p><p class="st">Ship the winner</p><p class="sd">with its ef and os</p></div></div>')

# solved: the worked example on tonight's index
c = 'padding:18px 28px'
hl = c + ';background:#2a2a5c;font-weight:600;color:#fcfdff'
write("solved", " foot", "Solved on Tonight's Index",
      '<p class="lead" style="margin-top:8px">Budget: p95 ≤ <span class="tr">3 ms</span>. Which encoder ships?</p>'
      f'<div class="dtab" style="margin-top:40px;width:1300px;display:grid;grid-template-columns:300px 520px 480px;font-size:44px">'
      f'<p class="hd" style="{c}">floor</p><p class="hd" style="{c}">winner</p><p class="hd" style="{c}">p95</p>'
      f'<p style="{c}">0.50</p><p style="{c}"><span class="tr">Zero</span> · ef 512</p><p style="{c}">0.96 ms</p>'
      f'<p style="{hl}">0.55</p><p style="{hl}"><span style="color:#b9b9ff">Nano</span> · ef 64</p><p style="{hl}">2.12 ms</p>'
      f'<p style="{c}">0.62</p><p style="{c}" class="mut">none fits</p><p style="{c}" class="mut">Stella 34.6 ms</p></div>',
      "BEIR NQ, 3,452 queries, 2.68M passages, one Stella index in Qdrant Edge, M5 Pro, sequential. Oversampling 4; "
      "nDCG@10 floor; fastest median among settings with ef ≥ 64. Exploratory.",
      note="<aside></aside>")

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
write("close", " alt", "Keep Your Index. <span class=\"tr\">Make the Query Encoder a Dial.</span>",
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
g.append(t(1664, 400, "encoding alone: 3,700×", 32, MUT, "end"))
g.append("</svg>")
write("cloud-e2e", " foot", "The Whole Request on a Cloud VM: 29× Cheaper",
      '<div style="margin-top:40px">' + "".join(g) + "</div>",
      "GCP c4-standard-8, $0.395/h, NQ, Qdrant Edge in one process, best of 1-8 threads, each encoder at its selected "
      "setting, warm page cache. Nano and Zero both stop near 600-650 queries/s: the search process sets the ceiling. Exploratory.",
      note="<aside>[~1.25 min] That was one laptop, one query at a time. So I put the same index on the cloud machine from earlier and paid for the whole request, encoding plus search. [Point to the two short bars.] Encoding alone made Zero look three thousand times cheaper than Stella. With search in the bill, both small encoders land around 17 or 18 cents per million, about 29 times cheaper than Stella. That's still a big saving. It's just a different number from the one on the encoding slide. [Point to the ratio.] Nano and Zero tie here because the search process becomes the ceiling, around six hundred queries a second in one process. More processes or shards would move that ceiling. A cheaper encoder wouldn't. And Zero spends more of that search budget than the others, because it needs a wider search to find its own neighbors.</aside>")

print("rebuilt")
