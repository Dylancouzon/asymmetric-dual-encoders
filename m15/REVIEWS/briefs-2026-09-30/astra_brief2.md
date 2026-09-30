You are an adversarial reviewer (read-only) for the shape of an arXiv research preprint. Read spine.md in the scratchpad directory named below first. The owner has just ruled that the paper is a research paper (findings, implications, possibilities), not a benchmark or competitor report; the registered comparator contrasts stay in the evaluation section as the pre-registered test.

Repository (read-only): /Users/dylanc/Documents/GitHub/asymetric-dual-encoders.

HARD ACCESS RULES:
- Never read or list `results/frozen_eval/untouched-*`, anything under `work/`, any reserved qrels cache, or raw queries/qrels of FEVER, DBpedia-entity, cqadupstack-android, cqadupstack-english.
- No recursive content search (grep -r, rg over a directory, find) across `results/` or `work/`. You may open these exact files: results/m20_beir15_run.json, results/m13_serving_costs.json, results/m13_build_record.json, results/m7_absorb_check.json, results/m7_learnability_report.json, results/m7_offfamily_report.json. Recursive search is fine inside m7src m8src m9src m10src m13src m20src bench scripts and the markdown in m7/ m8/ m9/ m10/ m12/ m13/ m20/ research/ (not research/archive).
- No web searches naming the four reserved datasets. No edits, no commits, no runs of training or evaluation.

Goal: the strongest research paper the committed evidence supports, unimpeachable to a hostile IR reviewer. Essential-only: report what is essential to that goal (a candidate finding that the evidence does not support as stated, a number that is wrong, a planned use-case claim that would be speculation, a measurement that cannot answer its question, a missing disclosure that would sink the paper). No wording or style notes. Say so plainly if you find nothing essential.

Tasks:
1. For each of the seven candidate findings in spine.md, verify its numbers against source (file:line) and rate the evidence: STRONG (registered or exact, with a source), ADEQUATE (descriptive, clear source, disclosed limits), WEAK (small n, dev-only, confounded), or UNSUPPORTED. For finding 2 check the tower/teacher denominators in results/m7_learnability_report.json and results/m7_offfamily_report.json and research/constella-in-plain-english.md; say exactly what the 43%..72% and Spearman 0.000 numbers are over.
2. Which one finding should lead a research paper, judged on evidence strength times novelty to an expert. Name a runner-up.
3. For the use cases (high QPS, search-as-you-type, edge/on-device, per-request tiering on one index, building a new query encoder for an existing index): what evidence exists in the repo today (search source dirs and markdown for throughput, QPS, prefix, keystroke, autocomplete, multi-thread scaling), and for each the minimal, valid measurement that would support one paragraph of claims. Say what each measurement cannot show. In particular: is truncating full BEIR queries to prefixes a valid proxy for search-as-you-type? What is the correct throughput measurement for "high QPS"?
4. Any finding in spine.md that a hostile reviewer would attack first, with the minimal fix.

Output: numbered sections, file:line evidence, under 1,200 words.
