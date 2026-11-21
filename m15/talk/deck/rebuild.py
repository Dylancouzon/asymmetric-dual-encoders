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
g.append(t(490, 80, "every uncached query · forever", 28, RED, "start", 600, True))
g.append(t(490, 350, "Stella per query: 31.6 ms · 25 GFLOPs", 34, WHITE, "start", 600))
g.append(t(490, 396, "1M queries ≈ 10.3 CPU-hours, before any search", 30, GRAY))
g.append(f'<line x1="0" y1="450" x2="1664" y2="450" stroke="{LINE}" stroke-width="2"/>')
g.append(t(0, 516, "C(t) = N · c_D + Q(t) · (c_Q + c_search)", 40, WHITE, "start", 400, True))
g.append(t(1664, 516, "agents multiply Q(t)", 30, RED, "end", 600))
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
g.append(t(1337, 420, "built once · never re-encoded", 30, TEAL, "middle", 600))
g.append(t(1337, 560, "↑ passages → Stella document encoder (once)", 26, MUT, "middle"))
g.append("</svg>")
write("arch", " foot", "Three Query Encoders Search One Index",
      '<div style="margin-top:24px">' + "".join(g) + "</div>",
      "FLOPs: analytic estimate, 2·P·T + attention, median over 100 real 5-12 word queries (Stella includes its prompt); "
      "Zero: 2 FLOPs per gathered value plus normalization.")

# 3. cloud: dollars per million on a log scale, VMs as squares
g = ['<svg width="1664" height="560" viewBox="0 0 1664 560">']
rows = [("Stella", 6.07, "$6.07", 56, GRAY), ("Nano", 0.236, "$0.24", 3, VIO), ("Zero", 0.0016, "$0.0016", 1, RED)]
x0, x1, lo, hi = 200, 1000, math.log10(0.001), math.log10(10)
X = lambda v: x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)
g.append(t(x0, 30, "encoding cost per 1M queries, at peak throughput (log scale)", 26, MUT))
g.append(t(1100, 30, "VMs to encode 1,000 queries/s", 26, MUT))
for i, (n, v, lab, vms, c) in enumerate(rows):
    y = 80 + i * 150
    g.append(t(0, y + 52, n, 40, c, "start", 600))
    g.append(f'<rect x="{x0}" y="{y + 10}" width="{X(v) - x0:.0f}" height="60" rx="8" fill="{c}"/>')
    g.append(t(X(v) + 16, y + 52, lab, 36, WHITE, "start", 600))
    for k in range(vms):
        cx, cy = 1100 + (k % 14) * 38, y + (k // 14) * 26 - 10
        g.append(f'<rect x="{cx}" y="{cy + 10}" width="30" height="20" rx="4" fill="{c}"/>')
    if vms == 1:
        g.append(t(1150, y + 28, "1, at 1.5% of its capacity", 28, WHITE))
for d in (0.001, 0.01, 0.1, 1, 10):
    g.append(t(X(d), 520, f"${d:g}", 24, MUT, "middle", 400, True))
g.append("</svg>")
write("cloud", " foot", "A Million Queries: $6.07, $0.24, or a Sixth of a Cent",
      '<div style="margin-top:40px">' + "".join(g) + "</div>"
      '<p class="lead" style="margin-top:8px"><span class="tr">Not yet counted:</span> graph search.</p>',
      "GCP c4-standard-8, us-east1, on-demand $0.395/h. Encoding only, batch 1, 8 processes × 1 ONNX Runtime thread "
      "(Zero: NumPy), 30 s closed loop on NQ queries; dollars assume a saturated VM. Exploratory.")

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
     t(0, 360, "min_θ  E_x ‖ f_θ(x) − Ê_S(x) ‖²   ·   the index Ê_D(d) never changes", 32, WHITE, "start", 400, True),
     "</svg>"]
chips = ('<div class="chips" style="margin-top:28px;gap:24px;white-space:nowrap"><p class="chip">RTX 3080 · 10 GB</p>'
         '<p class="chip red">≈ 20 min retrain</p><p class="chip">Nano: no relevance labels</p></div>')
write("distill", " foot", "Train Against a Frozen Index, on a Desktop GPU",
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

# 6. bench: the dial, encode time vs relevance kept, one index
W, H, L, R, T, B = 1664, 600, 140, 80, 30, 90
xl, xh, yl, yh = math.log10(0.02), math.log10(60), 0.75, 1.02
PX = lambda v: L + (math.log10(v) - xl) / (xh - xl) * (W - L - R)
PY = lambda v: T + (yh - v) / (yh - yl) * (H - T - B)
g = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
for v in (0.8, 0.9, 1.0):
    g.append(f'<line x1="{L}" x2="{W - R}" y1="{PY(v):.0f}" y2="{PY(v):.0f}" stroke="{LINE}" stroke-width="2"/>')
    g.append(t(L - 16, PY(v) + 9, f"{v:.0%}", 26, MUT, "end", 400, True))
for d, lab in ((0.03, "0.03"), (0.3, "0.3"), (3, "3"), (30, "30")):
    g.append(t(PX(d), H - B + 44, lab, 26, MUT, "middle", 400, True))
g.append(t((L + W - R) / 2, H - 8, "median query encode time, ms (log)", 26, GRAY, "middle"))
g.append(t(L, 22, "BEIR-15 nDCG@10, share of Stella", 26, GRAY))
pts = [("Stella", 31.6, 1.0, GRAY, "100%"), ("Nano", 2.25, 0.905, VIO, "90.5%"), ("Zero", 0.044, 0.814, RED, "81.4%")]
g.append(f'<polyline points="{" ".join(f"{PX(x):.0f},{PY(y):.0f}" for _, x, y, _, _ in pts)}" fill="none" stroke="{LINE}" stroke-width="3" stroke-dasharray="10 8"/>')
for n, x, y, c, lab in pts:
    g.append(f'<circle cx="{PX(x):.0f}" cy="{PY(y):.0f}" r="18" fill="{c}"/>')
    g.append(t(PX(x), PY(y) - 34, f"{n} · {lab}", 32, c, "middle", 600))
g.append(f'<circle cx="{PX(0.044):.0f}" cy="{PY(0.879):.0f}" r="16" fill="none" stroke="{RED}" stroke-width="4"/>')
g.append(t(PX(0.044) + 30, PY(0.879) + 10, "Zero + BM25 · 87.9%", 28, RED))
g.append("</svg>")
write("bench", " foot", "One Index, Three Price Points",
      '<div style="margin-top:24px">' + "".join(g) + "</div>",
      "Exact search, nDCG@10, unweighted mean of 15 BEIR datasets (registered descriptive evaluation); Stella 0.5614. "
      "Encode: M5 Pro CPU p50. Fusion adds a sparse branch (DBSF over 100 candidates per branch).")

# 7. split: whole-request bars on one millisecond axis
W, L0, AX = 1664, 230, 2.2
SX = lambda v: L0 + v / AX * 820
g = [f'<svg width="{W}" height="520" viewBox="0 0 {W} 520">']
g.append(t(SX(0), 30, "median request, ms", 26, MUT))
g.append(f'<rect x="{SX(0) + 300}" y="10" width="28" height="20" rx="4" fill="{TEAL}"/>')
g.append(t(SX(0) + 338, 30, "encode", 26, MUT))
g.append(f'<rect x="{SX(0) + 490}" y="10" width="28" height="20" rx="4" fill="{ORG}"/>')
g.append(t(SX(0) + 528, 30, "search", 26, MUT))
g.append(t(1420, 30, "nDCG@10", 26, MUT))
for i, (n, tot, enc, nd, c, star) in enumerate([("Nano", 2.04, 0.807, "0.579", VIO, ""),
                                                ("Zero", 0.86, 0.041, "0.510", RED, "*")]):
    y = 80 + i * 150
    g.append(t(0, y + 54, n, 44, c, "start", 600))
    g.append(f'<rect x="{SX(0)}" y="{y + 10}" width="{SX(tot * enc) - SX(0):.0f}" height="64" fill="{TEAL}"/>')
    g.append(f'<rect x="{SX(tot * enc):.0f}" y="{y + 10}" width="{SX(tot) - SX(tot * enc):.0f}" height="64" fill="{ORG}"/>')
    g.append(t(SX(tot) + 18, y + 56, f"{tot} ms", 40, WHITE, "start", 600))
    g.append(t(1420, y + 56, nd + star, 40, WHITE, "start", 600))
y = 380
g.append(t(0, y + 54, "Stella", 44, GRAY, "start", 600))
g.append(f'<rect x="{SX(0)}" y="{y + 10}" width="{SX(AX) - SX(0) + 40:.0f}" height="64" fill="{TEAL}" opacity="0.55"/>')
g.append(t(SX(AX) + 56, y + 56, "20.8 ms →", 40, WHITE, "start", 600))
g.append(t(1420, y + 56, "0.631", 40, WHITE, "start", 600))
for v in (0, 1, 2):
    g.append(f'<line x1="{SX(v)}" x2="{SX(v)}" y1="80" y2="470" stroke="{LINE}" stroke-width="2"/>')
    g.append(t(SX(v), 505, str(v), 24, MUT, "middle", 400, True))
g.append("</svg>")
write("split", " foot", "The Whole Request Is 2.4× Faster, Not 51×",
      '<div style="margin-top:32px">' + "".join(g) + "</div>",
      "BEIR NQ, 3,452 queries, 2.68M passages, one Qdrant Edge shard in-process, M5 Pro, sequential. Segments: median encode "
      "and search shares of the median request. Each encoder at its fastest setting within 1% of its own exact nDCG@10; "
      "*Zero misses that at every tested point (best shown: ef 1024, os 8). Exploratory.")
print("rebuilt")
