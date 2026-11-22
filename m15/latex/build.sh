#!/bin/sh
# Rebuild the manuscript from m15/PAPER.md: sh m15/latex/build.sh
set -eu
cd "$(dirname "$0")"
mkdir -p figures
cp ../figures/manuscript/*.png figures/

python3 - <<'PY'
import json
import re
from pathlib import Path

source = Path("../PAPER.md").read_text()
title_line, body = source.split("\n", 1)
title = title_line.removeprefix("# ")
roster = json.loads(Path("../AUTHORS.json").read_text())
names = [author["name"] for author in roster["authors"]]
byline = ", ".join(names[:-1]) + ", and " + names[-1] + ". " + roster["affiliation"] + "."
metadata = {"title": title, "author": byline, "lang": "en", "secnumdepth": 0}
header = "\n".join(f"{name}: {json.dumps(value)}" for name, value in metadata.items())

# Let LaTeX keep images and their printed captions together.
body = re.sub(
    r"!\[[^\n]*\]\(figures/manuscript/([^\n()]+)\)\n\n\*\*Figure B?\d+\.\*\* ([^\n]+)",
    lambda match: f"![{match[2]}](figures/{match[1]}){{width=100%}}",
    body,
)
body = re.sub(
    r"(?m)^\*\*Table \d+\.\*\* ",
    "Table: ",
    body,
)
exponents = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
body = re.sub(
    r"10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
    lambda match: "$10^{" + match[1].translate(exponents) + "}$",
    body,
)
body = body.replace("λ", r"$\lambda$")
Path("paper.md").write_text(f"---\n{header}\n---\n{body}")
PY

pandoc paper.md --standalone --from markdown+tex_math_dollars+pipe_tables --to latex \
  -V documentclass=article -V fontsize=10pt -V geometry:margin=1in \
  -V colorlinks=true -o paper.tex

python3 - <<'PY'
from pathlib import Path

paper = Path("paper.tex")
text = paper.read_text()
text = text.replace(
    r"\begin{document}",
    "\n".join((r"\usepackage{float,needspace}", r"\widowpenalty=10000",
               r"\clubpenalty=10000", r"\begin{document}")),
)
text = text.replace(r"\begin{figure}", r"\begin{figure}[H]")
text = text.replace(
    r"\subsection{Appendix B.",
    "\n".join((r"\setcounter{figure}{0}", r"\renewcommand{\thefigure}{B\arabic{figure}}",
               r"\subsection{Appendix B.")),
)
table_starts = text.split(r"\begin{longtable}")
assert len(table_starts) == 4, "Expected three manuscript tables"
text = table_starts[0] + "".join(
    rf"\Needspace{{{rows}\baselineskip}}" + "\n" + r"\begin{longtable}" + section
    for rows, section in zip((12, 16, 11), table_starts[1:])
)
text = text.replace("Against BM25, the lexical baseline,",
                    r"\Needspace{12\baselineskip}" + "\nAgainst BM25, the lexical baseline,")
text = text.replace(r"\textbf{Lookup table.}",
                    r"\Needspace{9\baselineskip}" + "\n" + r"\textbf{Lookup table.}")
paper.write_text(text)
PY

tectonic -X compile paper.tex --keep-logs > build.log 2>&1 || { tail -30 build.log; exit 1; }
echo "built $(pwd)/paper.pdf"
