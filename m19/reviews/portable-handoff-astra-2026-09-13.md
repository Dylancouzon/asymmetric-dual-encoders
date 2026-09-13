# M18/M19 portable handoff — Astra review

Date: 2026-09-13

Reviewer: `portable_handoff_astra` (`gpt-6-astra`, high reasoning)

Final disposition: **GO**

## Scope

The read-only review covered the README handoff link,
`M18_M19_REPRODUCIBILITY.md`, `m19/portable-artifacts-v1.json`,
`m19src/portable_handoff.py`, `m19src/test_portable_handoff.py`, and the M19 inheritance/final
system manifests for identity comparison. The brief prohibited recursive `work/` or `results/`
searches and every protected, spent, raw-query, qrels, judgment and confirmation surface.

## Findings and disposition

No P0/P1 finding was reported. Astra verified all 24 allowlisted files (507,607,301 bytes), found no
protected filename in the payload, matched the system and Zero-v1 identities to the frozen evidence,
and judged the implementation appropriately small.

Two P2 export races were found:

1. Inputs were verified before being reread into the archive, allowing a concurrent mutation to
   produce unverified archived bytes.
2. `os.replace` could overwrite a destination created after the initial existence check.

The fixes hash the exact bytes consumed by `tarfile.addfile`, refuse publication on any mismatch,
and atomically publish with a no-clobber hard link. The added regression test covers changed-input
rejection. Astra re-reviewed the fixes, confirmed both findings closed and returned GO.

The parent separately exercised the real payload: export, fresh extraction, hash verification and
an actual selected-system query all passed. That smoke used only the explicit system allowlist.

## Access declaration

The reviewer read only the assigned tracked files and ran the permitted synthetic tests and
allowlist verifier. The verifier hashed only the 24 explicitly named runtime/candidate files. No
protected queries, judgments, confirmation content or historical evaluation surfaces were read;
no recursive search, edit or commit was performed.
