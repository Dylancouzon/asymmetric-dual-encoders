# Owner decisions and preferences for the paper

Updated 2026-10-01. This is the durable record of Dylan's instructions for iterations on the M15 paper. Read it with `instructions-m15.md` and `HANDOFF.md` before revising. Entries below paraphrase the owner's messages; tentative ideas remain tentative. Agent recommendations and scientific conclusions belong in reviews and evidence, not in this record.

## Purpose and audience

- **Community credibility is the goal.** Improve Qdrant's image within the search engineering community through useful research. The goal is not to sell Qdrant.
- **Aim for a paper people find worth reading.** The research preview has attracted some search-engineering attention. The paper should have a clear purpose, an interesting finding, and a reason for that audience to care.
- **Write for a seasoned search engineer, not a project insider.** Do not assume familiarity with every embedding model, technique, or prior paper. Define models such as Stella at first use, including in the standalone abstract, and explain their roles before comparing numbers.
- **Readability matters even in a whitepaper.** The owner finds the current narration hard to read and jargon heavy. A narration pass must improve the flow and explanations, not merely expand acronyms or add a glossary.

## Editorial freedom and research scope

- **Existing content is expendable.** The owner is not attached to the current thesis, sections, or findings. Remove material with little relevance or importance; challenge assumptions and ideas.
- **Review the whole project before settling on an angle.** The owner challenges an emphasis on one aspect that hides the rest of Constella. Its interchangeable query paths and minimal-compute option should be considered for their actual relevance to the paper.
- **Independent research is authorized.** The owner gives broad discretion to improve the paper and permits cheaper subagents for research.
- **Additional work is welcome when justified.** There is no major publication rush. Some compute is acceptable if it materially strengthens a finding or makes the paper more interesting.
- **Do not retrain the whole models.** Additional analysis, inference, or bounded experiments should respect this constraint.

## Scientific expectations

- **Challenge rather than flatter.** The owner explicitly welcomes disagreement about angles, assumptions, importance, and the evidence.
- **Explain what the paper contributes.** The owner was unsure of the earlier paper's goal and was unimpressed by its state. A careful inventory of measurements is insufficient without a clear question and useful conclusions.
- **Findings must withstand a reasonable skeptic.** Distinguish controlled and repeated findings from anecdotal examples or narrow diagnostics. State how far a result can generalize and which units were actually replicated. The owner does not require convincing every possible skeptic.
- **Keep claims proportional to evidence.** Do not present an illustrative workload, recipe, or model family as a universal law. Broadening a claim may require an additional independent model space or workload, rather than more queries from the same setup.

## Tentative ideas, not mandatory theses

- **Training cost and approachability.** The owner suggests that the cost of training small models, and whether others can do it, might be interesting. This is a possible angle, not an instruction to make the entire paper about cost or promise turnkey reproduction.
- **Teacher differences.** The owner points to evidence that teachers are not interchangeable in their suitability for these small encoders. Its significance and scope should be evaluated, rather than assumed.
- **Constella capabilities.** Hot swapping and what the owner calls “zero latency options” might belong in the paper. The intent is to assess their fit and value, not to claim literal zero retrieval time or instantaneous model loading.
- **External comparators.** The owner questions whether bge-small and LEAF, selected as project targets, give a balanced paper comparison. A full model resweep may cost too much; whether and how to include these references remains a question. This is not an instruction to remove unfavorable results or to call these two models universally best in class.
- **Quantization.** The owner proposes it tentatively and worries about fit and fair comparisons. This authorizes investigating a useful controlled question, not adding a broad quantization sweep or treating the suggestion as a required headline.

## Auditability and iteration

- **Commit and push often.** Coherent changes, methods, results, and dispositions should remain auditable in the repository.
- **Preserve the owner's preferences across sessions.** The owner explicitly requests a persistent decision record so repeated iterations do not repeat earlier mistakes. Append new direction or mark superseded direction here; keep the detailed work history in `LOG.md`.
- **Do not turn agent judgments into owner decisions.** For example, calling v13 a good applied study or recommending native precision validation is an agent assessment. It is not an owner endorsement of the draft or approval of a final thesis.

## Checks before the next draft

1. Can an engineer explain what Stella, Zero, Nano, and Constella are after the opening, without consulting another file?
2. Does each experiment explain what stays fixed, what changes, what is measured, and what decision the result supports?
3. Are changes in workload, quality target, or timing protocol explicit before comparing numbers?
4. Are exact retrieval quality, candidate coverage, encoding time, and full search cost kept distinct?
5. Does each main conclusion state its reach: capability, repeated observation within this family, recipe-specific result, or exploratory diagnostic?
6. Does the main narrative lead with a useful question and findings, with implementation inventories and project identifiers subordinate to the explanation?

These checks operationalize the owner's readability and evidence concerns; they do not add an approval gate or authorize new experiments by themselves.

## Dated changes

### 2026-10-01

- Owner-directed independent review and rewrite, permission to remove low-value content, community credibility over selling Qdrant, frequent commits/pushes.
- Cost/approachability and teacher differences proposed as possible angles; owner questions whether the paper has a clear goal.
- Owner asks whether Constella's switching and minimal-compute capabilities fit, and challenges overfocus on one part of the project.
- Justified additional research/compute welcomed; no full-model retraining and no major deadline pressure.
- Quantization proposed with uncertainty about relevance and comparability.
- Owner identifies undefined Stella, asks for a narration pass, and clarifies the audience's prior knowledge.
- Owner says the paper is generally hard to read and jargon heavy, then challenges anecdotal evidence and asks for some generalizability under skeptical reading.
- Owner questions the relevance and balance of the selected bge-small/LEAF comparators, and the feasibility of rerunning a much broader roster.
- Owner requests this persistent record to prevent repeated mistakes.
