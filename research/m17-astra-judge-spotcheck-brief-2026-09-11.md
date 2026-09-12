# Brief: spot-check the 308-pair alias training sample (M17 ruling A6)

You are a judge, read-only. Open exactly one file, with `sed -n`/`nl -ba`/`cat`:
`results/m17_alias_spotcheck_sample.jsonl` (308 JSON lines). Do not open anything else, do not
search the tree, do not write files.

Context: these are a seeded random 2 % sample of 15,393 bulk alias training pairs mined by rule
from admitted sources: `R2_abbreviation_expansion` (an `Expansion (ABBR)` definition found in one
document, initial-matched) and `R1_k8s_glossary_aka` (a Kubernetes glossary "aka" form). Each
row has `form_a`, `form_b` (the two forms), and `view_a`, `view_b` (the same carrier sentence with
one form substituted for the other). The registered rule: if MORE THAN 5 % of the sample (16 or
more pairs) is wrong, the abbreviation filter is tightened and the pool rebuilt.

**A pair is WRONG if any of these hold:** the two forms do not denote the same thing (the
"expansion" is not what the abbreviation stands for, or the initials do not match the phrase);
the expansion is truncated or over-extended (missing a word of the name, or dragging in words
that are not part of it); the short form is a common word or number rather than an abbreviation;
or substituting one form for the other in the carrier sentence changes its meaning. Ordinary
grammatical awkwardness after substitution (an article that no longer fits) is NOT wrong.
Judge from the row text alone.

**Output.** After one paragraph on how you applied the rule, print ONE fenced block labelled
`jsonl` with exactly 308 lines, in file order:

```jsonl
{"line": <1-based line number>, "pair_id": "<pair_id>", "wrong": true|false, "note": "<optional, at most 10 words; required when wrong>"}
```

Then: the count of wrong pairs, the share of 308, whether it exceeds the 5 % threshold, a
breakdown of wrong pairs by `rule` and by the failure kind above, and the most common failure
pattern in one sentence (this is what would tune the filter). Finish with the list of files you
read (it must be that one file).
