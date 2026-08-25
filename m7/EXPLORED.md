# M7 avenues explored and closed

Check this before starting anything new.

| avenue | why killed | evidence |
|---|---|---|
| avenue | why killed | evidence |
|---|---|---|
| MIRACL (English) as a training source | Recovering positives for its 2,863 English train queries needs the 32.9M-passage `miracl/miracl-corpus` (no parquet mirror; the HF loader is a removed dataset script). Wikipedia coverage is already supplied by HotpotQA (85K), FEVER (110K), SQuAD (88K) and NQ-open (88K). Cost/benefit fails by two orders of magnitude. | `m7src/trainmix.py` docstring; `datasets` 5.0 refuses `miracl.py` |
| Climate-FEVER in UNTOUCHED-FINAL | No affirmative license at any primary source (climatefever.ai, arXiv:2012.00614 incl. appendices, the GitHub repo has no LICENSE and a silent README). Only HF mirrors claim CC-BY-SA-4.0, and a wrapper tag is not evidence here. | `m7/LEDGER.md` partition ledger; Sonnet primary-source sweep 2026-08-25 |
| fp32 teacher encodes on the dev and training corpora | fp16 is 2.4x faster and indistinguishable: cosine 1.000000 on 10K FiQA docs, \|Δ nDCG@10\| ≤ 3e-4 on both CQADupStack dev components. fp32 is kept for the six-set and untouched-final final run, where the cost is 0.2 h. | `results/m7_throughput.json`, `m7src/dtype_check.py` |
| bare (unprefixed) teacher query vectors as the distillation target | The bge query prefix is worth +1.85 dev macro to the teacher itself (0.5722 prefixed vs 0.5537 bare on the three fast dev components), so the prefixed vector is the better target and the higher retention ceiling. The table's own two prefix variants remain a separate mandated ablation. | `work/devres/refs.json` (`bge-base-symmetric` vs `-nopfx`) |
