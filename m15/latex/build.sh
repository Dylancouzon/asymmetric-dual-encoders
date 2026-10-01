#!/bin/sh
# Build the arXiv LaTeX source and PDF from m15/PAPER.md: sh m15/latex/build.sh
set -e
cd "$(dirname "$0")"
mkdir -p figures
cp ../figures/*.pdf figures/
# Drop the internal draft banner; point images at the PDF figures.
paper_title=$(sed -n '1s/^# //p' ../PAPER.md)
sed -e '1{/^# /d;}' -e '/^\*\*Draft v[0-9]*/d' -e 's#(figures/\([a-z0-9_]*\)\.png)#(figures/\1.pdf)#g' ../PAPER.md > paper.md
# Use each bold "**Figure N.** ..." paragraph as its figure's LaTeX caption (pandoc takes the
# caption from the image's alt text) and drop the duplicate paragraph.
python3 - <<'PY'
import re
s = open("paper.md").read()
s = re.sub(r"!\[[^\]]*\]\((figures/[^)]+)\)\n\n\*\*Figure \d+\.\*\* ([^\n]+)",
           lambda m: "![" + m.group(2).replace("[", "(").replace("]", ")") + "](" + m.group(1) + ")", s)
open("paper.md", "w").write(s)
PY
pandoc paper.md -s -o paper.tex --from markdown+tex_math_dollars+pipe_tables \
  -V documentclass=article -V fontsize=10pt -V geometry:margin=1in -V colorlinks=true \
  -V author="Dylan Couzon (Qdrant)" -V date="October 2026" \
  --metadata title="$paper_title"
# Keep compact tables and their captions together, and figures within their research sections.
python3 - <<'PY'
import re
from pathlib import Path
p = Path("paper.tex")
s = p.read_text()
s = s.replace(r"\usepackage{longtable,booktabs,array}",
              r"\usepackage{longtable,booktabs,array}" + "\n" + r"\usepackage{needspace,placeins}")
# Captions include the method; reserve enough room for them and each compact table.
for number, lines in ((1, 16), (2, 14), (3, 13)):
    caption = rf"\textbf{{Table {number}.}}"
    s = s.replace(caption, rf"\Needspace{{{lines}\baselineskip}}" + "\n" + caption)
for heading in ("4. Lexical fusion", "6. Serving compatibility"):
    needle = r"\subsection{" + heading
    s = s.replace(needle, r"\FloatBarrier" + "\n" + needle)
s = s.replace(r"\subsubsection{3.2 Query precision", r"\FloatBarrier" + "\n" +
              r"\subsubsection{3.2 Query precision")
s = s.replace("The strongest registered teacher,", r"\FloatBarrier" + "\n" +
              "The strongest registered teacher,")
s = s.replace("Our results establish four findings:",
              r"\Needspace{8\baselineskip}" + "\n" + "Our results establish four findings:")
s = s.replace("The originally reserved four,", r"\Needspace{20\baselineskip}" + "\n" +
              "The originally reserved four,")
p.write_text(s)
PY
tectonic -X compile paper.tex --keep-logs > build.log 2>&1 || { tail -30 build.log; exit 1; }
echo "built $(pwd)/paper.pdf"
