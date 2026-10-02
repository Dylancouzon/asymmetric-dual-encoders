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
s = re.sub(r"!\[[^\]]*\]\((figures/[^)]+)\)\n\n\*\*Figure [A-Z]?\d+\.\*\* ([^\n]+)",
           lambda m: "![" + m.group(2).replace("[", "(").replace("]", ")") + "](" + m.group(1) + ")", s)
open("paper.md", "w").write(s)
PY
pandoc paper.md -s -o paper.tex --from markdown+tex_math_dollars+pipe_tables \
  -V documentclass=article -V fontsize=10pt -V geometry:margin=1in -V colorlinks=true \
  -V author="Constella authors" -V date="October 2026" \
  --metadata title="$paper_title"
# Keep compact tables and their captions together, and figures within their research sections.
python3 - <<'PY'
import json
import re
from pathlib import Path
p = Path("paper.tex")
s = p.read_text()
# Keep the author roster in one source and lay it out in two columns with contact details.
roster = json.loads(Path("../AUTHORS.json").read_text())
def tex_escape(value):
    return re.sub(r"([&%$#_{}])", r"\\\1", value)
author_cells = [tex_escape(a["name"]) + r"\\" + "\n" +
                r"{\small\href{mailto:" + a["email"] + r"}{\nolinkurl{" + a["email"] + "}}}"
                for a in roster["authors"]]
author_rows = [" & ".join(r"\begin{tabular}{c}" + cell + r"\end{tabular}"
                          for cell in author_cells[i:i+2])
               for i in range(0, len(author_cells), 2)]
author_block = (r"\author{\begin{tabular}{cc}" + "\n" +
                (r"\\[0.8em]" + "\n").join(author_rows) + "\n" +
                r"\end{tabular}\\[0.6em]" + tex_escape(roster["affiliation"]) + "}")
s = s.replace(r"\author{Constella authors}", author_block)
s = s.replace(r"\hypersetup{", r"\hypersetup{" + "\n  pdfauthor={" +
              ", ".join(tex_escape(a["name"]) for a in roster["authors"]) + "},", 1)
s = s.replace(r"\usepackage{longtable,booktabs,array}",
              r"\usepackage{longtable,booktabs,array}" + "\n" + r"\usepackage{needspace,placeins}")
s = s.replace(r"\begin{document}",
              r"\widowpenalty=10000" + "\n" + r"\clubpenalty=10000" + "\n" + r"\begin{document}")
# Captions include the method; reserve enough room for them and each compact table.
for number, lines in ((1, 16), (2, 14), (3, 13), ("B", 18), ("C", 22), ("D", 24), ("E", 20), ("F", 24)):
    caption = rf"\textbf{{Table {number}.}}"
    s = s.replace(caption, rf"\Needspace{{{lines}\baselineskip}}" + "\n" + caption)
s = s.replace(r"\subsection{Appendix A.",
              r"\FloatBarrier" + "\n" + r"\setcounter{figure}{0}" + "\n" +
              r"\renewcommand{\thefigure}{A\arabic{figure}}" + "\n" + r"\subsection{Appendix A.")
s = s.replace(r"\subsection{Appendix B.", r"\FloatBarrier" + "\n" + r"\subsection{Appendix B.")
for heading in (r"\subsubsection{5.3 Is the effort predictable?}",
                r"\subsection{7. Limitations}", r"\textbf{Variation across datasets.}"):
    s = s.replace(heading, r"\FloatBarrier" + "\n" + heading)
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
for paragraph, lines in (("For the other seven features,", 23),
                         ("The original Zero tests used", 17),
                         ("The graph-free precision control uses", 16)):
    s = s.replace(paragraph, rf"\Needspace{{{lines}\baselineskip}}" + "\n" + paragraph)
p.write_text(s)
PY
tectonic -X compile paper.tex --keep-logs > build.log 2>&1 || { tail -30 build.log; exit 1; }
echo "built $(pwd)/paper.pdf"
