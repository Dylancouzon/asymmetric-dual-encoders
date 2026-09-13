# M19 code and artifact map

| Path | Purpose |
|---|---|
| `m19/registry.json` | prospective constants, gates, versions and execution state |
| `m19/inheritance-lock.json` | generated binding of admitted M18 manifests and local inherited bytes |
| `m19/STATUS.md`, `PLANNING.md`, `LEDGER.md`, `FINDINGS.md` | compact state and durable record |
| `m19src/common.py` | M19 paths, strict read/write admission, hashes and atomic writes |
| `m19src/inherit.py` | verify and publish the M18/Zero-v1 inheritance lock |
| `m19src/test_common.py`, `test_inherit.py` | synthetic path-guard and lock tests |
| `work/m19/` | gitignored raw/derived/run state owned by M19 |
| `results/m19_*` | tracked compact evidence; never raw private query text |

M19 starts from M18 commit `bfa7257fe369a41ce048e150f861703a662a08f0`. The first fork is
`m18src/common.py`: M19 replaces its historical substring denylist with a narrow admitted-input
boundary, adds M18 confirmation refusal, and limits writes to M19-owned paths.
