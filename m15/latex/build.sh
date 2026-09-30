#!/bin/sh
# Build the arXiv LaTeX source and PDF from m15/PAPER.md: sh m15/latex/build.sh
set -e
cd "$(dirname "$0")"
mkdir -p figures
cp ../figures/*.pdf figures/
# Drop the internal draft banner; point images at the PDF figures.
sed -e '/^# Screen the Student/d' -e '/^\*\*Draft v[0-9]*/d' -e 's#(figures/\([a-z0-9_]*\)\.png)#(figures/\1.pdf)#g' ../PAPER.md > paper.md
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
  --metadata title="Screen the Student: Cheap Query Encoders for a Frozen Document Index"
tectonic -X compile paper.tex --keep-logs > build.log 2>&1 || { tail -30 build.log; exit 1; }
echo "built $(pwd)/paper.pdf"
