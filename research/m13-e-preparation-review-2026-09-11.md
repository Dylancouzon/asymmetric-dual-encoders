# E preparation and launch reviews — 2026-09-11

The dependency preflight reads only admitted COV/training metadata, validates the live registry
and F verdict, checks the 60k warm-start configuration, requires absent registered E output,
and verifies exact COV cache identities, chunk hashes/shapes and stitched bytes. No score or
encoder is invoked. The throughput benchmark times pinned stella fp32 at 512 tokens with
32,768 padded tokens/batch on 1,000 then 10,000 admitted SQuAD training passages. It records
length distributions and treats the 10M projection as a surrogate with twofold headroom, not
as a protected-corpus measurement. The reserved executor remains a separate readiness item.

Independent cross-reviews by encode_benchmark and e_launch_audit cleared both new scripts;
synthetic checks covered sampling/allocation arithmetic/refusals and good/corrupted cache
validation. Parent inspected the scripts and teacher cache/encode interfaces. No cloud or
protected content was opened during reviews; only named source, permitted result metadata
and scratch fixtures were accessed. No recursive searches or credential reads by reviewers.

Both reviewers cleared the bounded preparation launcher after signal/ownership cleanup fixes:
start only the existing stopped Pod at the quoted price; restore bootstrap under remote timeout;
deploy exact committed HEAD via Git bundle; run dependency preflight and benchmark under separate
1,200-second remote deadlines; copy successful/failure receipts before bounded STOP retries;
raise if shutdown is unconfirmed. No registered E training is launched by preparation.

## Registered E controller

Both independent reviewers returned GO for `scripts/m13_cloud_e.py`, conditional on committed
passing preflight/encode receipts and a reviewed measured allocation below $1,000. Fixed findings:
receipt hash/PASS/registry binding; live Pod price/state; zero-exit terminal FAILED arm outcomes
reported explicitly while both registered arms remain scheduled unchanged; partial running
backups need no nonexistent published terminal twin; exact owned process marker parsing.
Synthetic verification covered good/bad/stale allocations, complete/corrupt/partial backups,
and actual rendered cleanup against owned and unowned sleeping local processes. No cloud or
protected access in these checks. The controller is fresh-launch only, with no automatic rerun.
It copies entire E output trees and published records, verifies remote manifest plus checkpoint
and COV references locally, and stops compute on completion/failure. WSL must remain awake.
