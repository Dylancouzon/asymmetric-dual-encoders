# Published-model smoke test

Tested on 2026-09-16 with the public Hugging Face artifacts and FastEmbed preview commit
`47a50907415a90f26d4b49d7a01efe210848813e`.

The environment used FastEmbed 0.8.0, Qdrant Client 1.19.0, ONNX Runtime 1.29.0, and
Hugging Face Hub 1.28.0.

| Artifact | Tested Hugging Face revision |
|---|---|
| `DylanCouzon/stella-en-400M-v5-doc-onnx` | `44c3ba46827965e9f712f84ece996698fffafcb4` |
| `DylanCouzon/constella-nano` | `cd8e70646a1242e12b0c9cffe1dced73803d08f5` |
| `DylanCouzon/constella-zero` | `54453d612eb5a173aa327be0c4fd991c18f97861` |

Both query encoders returned the expected top destination for all six queries in
`space_travel.json`. The test used one in-memory Qdrant collection: its document vectors were
created once and were not rebuilt when switching query encoders.

Observed top-result cosine scores:

| Query | Nano | Zero |
|---|---:|---:|
| underwater hotel | 0.5164 | 0.4103 |
| train through a red desert | 0.6666 | 0.4884 |
| flowers above the clouds | 0.6410 | 0.5235 |
| beginner kayaking near Saturn | 0.7314 | 0.5536 |
| unusual ice cream for children | 0.4841 | 0.3085 |
| objects from early Moon trips | 0.5917 | 0.3969 |

Scores are included only as a record of this run. The smoke-test contract is the expected top
destination, not exact floating-point equality.

The published-card check also executed every Python block from the three current card sources:

- the document encoder usage block;
- the Nano + Qdrant usage block;
- the Zero + Qdrant usage block; and
- Zero's standalone NumPy reference block.

All four completed successfully against the public artifacts above. Nano ranked the vaccine
passage first with cosine similarity 0.7418; Zero ranked it first with cosine similarity 0.6423.
