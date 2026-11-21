# Reproduce the frozen-index paper inputs

Verify inputs by SHA-256 before using them. The commands here start from the repository root and do not run any evaluation or load encoder checkpoints.

## Million-passage MS MARCO sample

The reconstruction reads the passage-ID column from the pinned public corpus and the public development judgments. It uses the original seed and JSON layout from `m15src/e2_encode.py`. The script verifies the recorded hash before writing a new file.

Populate the Hugging Face cache with these two public inputs. The optional `--hub-cache` argument points to a cache at a different location.

- `BeIR/msmarco` at `a918e0d11a77ed33f42f29d98340b655593b96ad`, file `corpus/corpus-00000-of-00001.parquet`.
- `BeIR/msmarco-qrels` at `253fbf8a3f8d4a0932b63882b5162bedc84779f5`, file `dev.tsv`.

```sh
.venv/bin/python m15/rebuild_msmarco_ids.py /tmp/msmarco1m-rebuilt.json
shasum -a 256 /tmp/msmarco1m-rebuilt.json
```

The expected SHA-256 is `4a09d3d055f13ae8cfd3f3a9202574af0f084954cd51f482918ce31aaee181a3`, from `results/m15_e2_ann_msmarco1m.json`. The reconstruction matched the original JSON byte for byte. If you have a copy of the original archive at `artifacts/reproduction/msmarco1m_ids.json.gz`, verify its decompressed bytes.

```sh
gzip -dc artifacts/reproduction/msmarco1m_ids.json.gz | shasum -a 256
```

## Cleaned fitting queries

The original `work/m8_trainq_texts.json` is absent from the local workspace and Git. `results/m8_trainq_manifest.json` records 337,981 queries, 21,327,603 bytes, and SHA-256 `da0f208ef29bae3833ffead070eb34b61f390f9d72ed96d448d45ae6b52072c2`. Check a recovered copy with `shasum -a 256` before using it. The original cleaning step used protected evaluation inputs, so this reproduction workflow requires recovery of the already-cleaned file. The related [provenance audit](PROVENANCE_AUDIT.md) records the storage boundary.
