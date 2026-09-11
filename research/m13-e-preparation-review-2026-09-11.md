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

## Launcher and measured allocation

Both reviewers cleared the outer E launcher after replacing subprocess.run with retained Popen:
interruptions send the controller SIGTERM and allow 1,800 seconds for its bounded evidence/STOP
cleanup before forced kill and outer STOP. The startup wrapper refuses tracked prep receipts
before archiving the two untracked originals, deploys exact HEAD, and bootstraps under timeout.
Operational wrapper copy: `work/m13cloud-launchers/e.py` (no embedded credentials).

The cloud preflight passed all ten admitted COV units and five teacher caches. The encode
benchmark passed both sizes with no allocator retries. Parent validation: all 55 build-controller
tests passed; build13 printed the measured allocation; E controller verified receipt/hash gates.
The numerical review recomputed the base $719.09, E runtime topup $16.70 and $4.74 paid, yielding
$740.53 and $259.47 ceiling headroom. Allowances use explicit conservative surrogates and do not
authorize build or protected access. No recipe/schedule/protocol fields changed.

Both e_launch_audit and encode_benchmark independently recomputed the final allocation and
returned GO for E only. Both confirmed all six artifact bindings and three benchmark code
hashes, the 199 ex/s safety rate, reserved55.2h/LoTTE15h allowances, $740.53 total and $259.47
headroom. Their access audits covered only named source/receipt/config/plan metadata; no cloud,
credentials or protected payloads.

## Local E follow-on

Both e_launch_audit and encode_benchmark returned GO for `scripts/m13_after_e.py`,
conditional on the actual dependency preflight passing. It passed all six DEV-6 components
(`results/m13_dev6_preflight.json`) without scoring or teacher encoding. The four text caches'
current shard and combined bytes match their recorded hashes. Their historical trust-on-first-use
labels remain intact; this is not an upgrade to authenticated provenance. An initial stricter
check refused those legacy labels, as expected; both attempt logs are retained locally.

The one-shot follow-on requires successful cloud completion, confirmed STOP, and verified
backups bound to the cloud receipt before restoring independent local copies. It publishes
the original records, fills descriptive DEV-6 only if the local GPU is idle, then computes
E1 and applies the registered selection. It refuses existing evidence, source/registry drift,
unrelated tracked edits and incomplete contrast results. Failures preserve outputs without
automatic retries. The scope ends at E1; no build or protected evaluation is launched.
Author synthetic checks covered success, busy GPU, prior computed E1 and an uncomputed result
despite zero CLI exit. Parent inspected the final script and actual passing dependency receipt;
independent reviewers inspected named source files and parsed the scripts, without cloud,
credential or protected-content access. Operational preflight copy is retained in
`work/m13cloud-launchers/dev6_preflight.py`.
