# M18/M19 portable handoff

The branch records the M18 and M19 code, protocol, compact results and review trail. The large
runtime bytes remain under gitignored `work/`; committing roughly 484 MiB of binary/model/corpus
data to Git would make the repository harder to reproduce, not easier. Instead,
`m19/portable-artifacts-v1.json` is an explicit content-addressed allowlist and
`m19src/portable_handoff.py` verifies or exports exactly those files.

There are two payloads:

- `system` (443,837,325 bytes before gzip) contains the M18 corpus/index and released Zero-v1
  bytes needed by the system selected at the end of M19.
- `m19-candidates` (63,769,976 bytes) adds the unselected deterministic V0-compose and T0-teacher
  bundles and their twelve cached teacher vectors. It is useful for inspecting M19, but it is not
  required to run the selected system.

Neither payload contains query text, qrels, judgment labels or packets, confirmation material,
raw GitHub acquisition, or any historical evaluation surface. Do not add those to a handoff.

## On the source machine

Check the minimum runnable payload. The release bundle currently lives in the original checkout,
so name it explicitly:

```bash
/home/dylan/asymetric-dual-encoders/.venv/bin/python m19src/portable_handoff.py verify \
  --profile system \
  --release-root /home/dylan/asymetric-dual-encoders/work/release/zero-v1
```

Create one internal-transfer archive with the runnable system and optional M19 candidates:

```bash
/home/dylan/asymetric-dual-encoders/.venv/bin/python m19src/portable_handoff.py export \
  --profile all \
  --release-root /home/dylan/asymetric-dual-encoders/work/release/zero-v1 \
  --output /SAFE/INTERNAL/PATH/m18-m19-work-v1.tar.gz
```

The exporter refuses a missing, changed or incorrectly sized input, an unsafe archive path, and an
existing output. It reads only the files named in the manifest and publishes the archive only after
the complete write succeeds.

## On the receiving machine

Clone and check out `m18-qdrant-project-memory`, then extract the archive at that checkout's root:

```bash
tar -xzf /SAFE/INTERNAL/PATH/m18-m19-work-v1.tar.gz
python3.12 -m venv .venv
.venv/bin/pip install -r m7/requirements.lock.txt
.venv/bin/python m19src/portable_handoff.py verify --profile all
```

Run the selected system with paths relative to the new checkout:

```bash
.venv/bin/python m18src/system.py query "k8s readiness probes" \
  --index work/m18/derived/index \
  --bundle work/release/zero-v1 \
  --limit 10
```

The first query loads a roughly 162 MiB memory-mapped document matrix and the BM25 index. Exact
ranking can vary in latency across machines; the manifest verifies the actual model, corpus and
index bytes rather than treating timing as portable.

## What can and cannot be reproduced

The archive fully replays the final selected local system and preserves the exact unselected M19
candidate artifacts when `all` is chosen. The tracked locks and result manifests provide the hashes,
counts, decisions and adversarial review paper trail for both milestones.

M18's one-shot confirmation cannot be rerun: it is consumed and its raw queries/qrels are a
protected surface. M19 confirmation was never accessed, and M19 stopped before qrels or quality
metrics because 60 independently audited items remained unjudgeable. The private development query
and judgment payloads are deliberately absent. Accordingly, another machine can verify and replay
the system and candidate bytes, but must not claim to have independently regenerated the closed
M18 confirmation or M19 judgment outcome from this handoff.

For a fresh deterministic M19 candidate rebuild rather than byte replay, download
`NovaSearch/stella_en_400M_v5` at revision
`ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20` and verify every file against
`m19/teacher-snapshot-lock-v1.json`; then follow `m19/CODEMAP.md` and the build receipts. That model
snapshot is intentionally not duplicated in the transfer archive.
