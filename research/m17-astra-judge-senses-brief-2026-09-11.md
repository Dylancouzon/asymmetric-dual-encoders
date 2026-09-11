# Brief: judge the 80 pending alias senses (M17 ruling A6, descriptive surface)

You are a judge, read-only. Open exactly one file, with `sed -n`/`nl -ba`/`cat`:
`results/m17_panel_pending_judgments.jsonl` (440 JSON lines). Do not open anything else, do not
search the tree, do not write files. Judge only the lines whose `kind` is `alias-pair` (80 of
them); skip `panel-candidate` lines.

Context: these are candidate same-meaning pairs for a held-out alias test (does a short form and
its expansion retrieve the same documents). Each row has `query` (the short form or extracted
phrase), `candidate_title` (the proposed expansion or full name), `candidate_first_300_chars`
(the sentence(s) from the Kubernetes documentation that the pair was extracted from), a
`question` written for a human, and `reason`:

* `ambiguous-sense` (59): the `question` asks whether the bare short form means the proposed
  expansion IN THIS CONTEXT, and lists other expansions seen elsewhere. `yes` if, in the quoted
  sentence, the short form clearly denotes that expansion; `no` if another sense fits or the
  context does not determine it.
* `truncated-extraction` (21): the `question` asks whether the extracted phrase is the COMPLETE
  name the sentence states (it hit a four-word cap or does not start at a phrase boundary).
  `yes` only if the phrase in `candidate_title` is exactly the full name/expansion the sentence
  gives for the short form in `query`; `no` if it is cut off, starts mid-phrase, or is not what
  the sentence equates with the short form.

Judge from the row text alone. When undecidable, answer `no` and note `unsure`.

**Output.** After one paragraph on how you applied the two rules, print ONE fenced block
labelled `jsonl` with exactly 80 lines, one per alias-pair row, in file order:

```jsonl
{"line": <1-based line number in the file>, "id": "<pair_id>", "candidate": "<candidate_doc_id>", "answer": "yes"|"no", "note": "<optional, at most 10 words>"}
```

Then a short table: yes/no counts per `reason`, and the number of `unsure` notes. Finish with
the list of files you read (it must be that one file).
