# Brief: judge the 360 Kubernetes panel rows (M17 ruling A6, descriptive surface)

You are a relevance judge, read-only. Open exactly one file, with `sed -n`/`nl -ba`/`cat`:
`results/m17_panel_pending_judgments.jsonl` (440 JSON lines). Do not open anything else, do not
search the tree, do not write files. Judge only the lines whose `kind` is `panel-candidate`
(360 of them); skip `alias-pair` lines.

Context: the panel is a descriptive evaluation surface for a query encoder over Kubernetes
documentation. Each row has `query` (a documentation page title or heading used as a search
query), `candidate_title` and `candidate_first_300_chars` (one candidate page), and
`candidate_doc_id` (its path). You judge from the row text alone.

**Rule.** `yes` if a Kubernetes user searching that query would consider the candidate page a
relevant result: the candidate is the page the heading belongs to, or substantively covers the
topic named by the query. `no` if it only shares words or mentions the topic in passing, or is
about something else. When the 300 characters do not settle it, use the title and path; when
still undecidable, answer `no` and note `unsure`.

**Output.** After a one-paragraph statement of how you applied the rule, print ONE fenced block
labelled `jsonl` with exactly 360 lines, one per panel-candidate row, in file order:

```jsonl
{"line": <1-based line number in the file>, "id": "<query_id>", "candidate": "<candidate_doc_id>", "answer": "yes"|"no", "note": "<optional, at most 10 words>"}
```

No other text inside the block. Then a short table: yes/no counts per `partition`
(`selection`, `audit`), and the number of `unsure` notes. Finish with the list of files you read
(it must be that one file).
