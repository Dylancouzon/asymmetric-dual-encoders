You are an adversarial reviewer (read-only) of a proposed shape for an arXiv IR preprint. The proposal is in the file proposal.md in the scratchpad directory given below; read it first.

Repository (read-only): /Users/dylanc/Documents/GitHub/asymetric-dual-encoders. The paper branch files are readable with `git show m15-whitepaper:m15/PAPER.md` and `git show m15-whitepaper:m15/EVIDENCE_INDEX.md`.

HARD ACCESS RULES. Breaking them damages a research record:
- Never read or list `results/frozen_eval/untouched-*`, `work/` (anything), any reserved qrels cache, or raw queries/qrels of FEVER, DBpedia-entity, cqadupstack-android, cqadupstack-english.
- No recursive content search (grep -r, rg, find) across `results/` or `work/`. You may open these exact result files: results/m20_beir15_run.json, results/m13_reserved_run.json, results/m13_build_record.json, results/m13_serving_costs.json. Recursive search is fine inside m7src m9src m10src m13src m20src bench scripts and the markdown files in m7/ m9/ m10/ m12/ m13/ m20/ research/ (not research/archive).
- No web searches naming the four reserved datasets. No edits, no commits, no training or evaluation runs.

Goal the findings are measured against: a preprint that is unimpeachable on evidence and access discipline, focused on what is new, with a registered headline that is honestly reported next to the held-out result.

Essential-only rule: report only what is essential to that goal. That means a claim or number in the proposal that is wrong, a planned statement that would violate the registration (m20/REGISTRATION.md, m20/beir15_registry.json `reporting`, CLAUDE.md evidence rules), a planned measurement that is invalid or cannot answer its question, a missing disclosure a hostile IR reviewer would use to reject the paper, and scope that is over-engineered for the decision. Wording and style are out of scope. If you find nothing essential, say so plainly.

Check specifically:
1. Every number in proposal.md against its source (results/m20_beir15_run.json for BEIR-15 macros; results/m13_reserved_run.json for R1/R2; m21/BENCHMARKS.md or m13/STATUS.md for the registered contrasts; m7/RECIPE.md and results/m13_build_record.json for the training-data claims; bench/edge_prototype_pair.py for the claim that the M9 system-latency "nano" was an untrained MiniLM and the index synthetic).
2. Whether the contact-partition plan is valid and symmetric, and whether the proposal's partition macros would be read as a significance claim the registry forbids.
3. Whether "retention against the single-prompt stella query path" is a defensible denominator given the published card uses per-task instructions (card 58.97 vs ours 56.14).
4. Whether E1-E5 can answer what they claim (e.g., recall@10 vs exact on a 1M subset; one server lifetime swap; memory limits), and what is missing or unnecessary.
5. The strongest reason a hostile reviewer would reject the paper as proposed, and the minimal fix.

Output: a numbered list of findings, each with severity P0 (blocks), P1 (must fix before drafting), P2 (should fix), the evidence (file:line), and the minimal fix. Then a one-line verdict. Under 1,000 words.
