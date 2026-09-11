# Cloud readiness — 2026-09-11

Dylan reports cloud budget approval. The registered ceiling remains $1,000. Ready for provider
setup and the benchmark; not yet ready to start the 200M build. The ordered execution gates
remain in `EXECUTION.md`. No rental or protected evaluation was performed in this readiness check.

M13 now has its own worktree at `/home/dylan/asymetric-dual-encoders/work/m13cloud`, on
`m13-stage1-execution-prep`, starting at reviewed commit `3006eab`. M17 keeps the original checkout.
Only the existing Python environment is linked into the new worktree; training caches and outputs
are not shared into it. A synthetic corpus-identity test needed its own holdout fixture to run
without the original checkout's gitignored artifacts; production behavior is unchanged.

Validation: `run_checks.sh` passed M13 (170 tests), M9 (16), and M12. M10 had 473 passes,
one optional cache skip and the holdout-fixture failure above. After fixing that test, all 79
corpus-loader tests passed. The full runner was not repeated after the fixture-only fix.
No cloud GPU smoke or throughput measurement has been performed in this session.

Account connection verified on 2026-09-11: Runpod reports $505 credit, no Pods and no network
volumes. Dylan confirms that this is initial funding and can be refilled; the $1,000 ceiling
is unchanged. The authenticated Secure Cloud quote for one A100 SXM 80 GB is $1.59/hour,
stock Low. The data-center query reports Low A100 SXM stock in US-MD-1; the other listed
A100 SXM locations returned null stock, which is not a capacity confirmation. Recheck matching
network-volume support, full SSH and live capacity before creating storage or compute.
Credentials are kept outside Git in the user's WSL configuration directory and are never shipped.

## Selected deployment — 2026-09-11

Dylan authorized starting cloud setup. Live matching A100 SXM/network-volume capacity was absent
in US-CA-2, EU-RO-1 and EUR-IS-1; the Maryland A100 SXM is available but its data center reports
`storageSupport=false`. The initial deployment therefore uses a **Pod volume disk**, 500 GB,
mounted at `/home/dylan`, with one Secure Cloud on-demand A100 SXM 80 GB in US-MD-1.
Exact request: `runpod_pod_config.json`; PUBLIC_KEY is injected from the local SSH public key.
The official Runpod image is pinned by digest; bootstrap installs the separately pinned Python
3.12 environment on the persistent mount. Scoped GitHub deploy access is configured separately.

**STOP preserves this disk; TERMINATE DELETES it.** Do not terminate until required artifacts
are backed up and verified locally. GPU capacity on restart is not guaranteed. No network volume
is created. Keep every unique checkpoint on the local box as well as the Pod disk.

Published disk rates are $50/month while running, $100/month while stopped, plus up to $3/month
for the 30 GB container disk while running. Reserve $103 for a conservative 30-day storage
window in the allocation; data transfer has no provider fee. Measure and record actual charges.
GPU quote is $1.59/hour, so the initial running configuration is about $1.664/hour including
storage at a 720-hour monthly conversion. Stop compute between stages and monitor balance/spend.
Source: [Runpod Pod pricing](https://docs.runpod.io/pods/pricing).

## Initial validation job

`scripts/m13_cloud_stage0.py` runs on WSL after provisioning/bootstrap. It uploads only the
audited list, verifies every SHA-256, runs active checks and the plan, smokes E-bs32/E-bs128 at
512 tokens, and invokes `scripts/m13_cloud_smoke.py` for checkpoint interruption/resume.
It copies and verifies smoke checkpoints/records locally and requests STOP, with bounded retries
and confirmation of the provider's desired `EXITED` state. Six-hour overall limit; four-hour upload
limit. It never launches registered arms or protected evaluation. Receipt:
`results/m13_cloud_stage0.json`; local logs: `logs/m13-cloud-*.log`; downloaded evidence:
`work/m13cloud-evidence/`. Keep WSL and Windows awake while this local upload/controller runs.

Two independent reviews cleared this job after fixes to signal/process cleanup, exact remote
commit checks, complete backup verification, and STOP retries. The transfer reviewer checked
the 266 allowlisted paths and matching checksum entries. The smoke reviewer checked the real
600-step checkpoint interruption/resume caller; evaluations are the runner's smoke stubs.

Cloud bootstrap passed on the actual A100: Python 3.12.14, torch 2.8.0+cu126, CUDA 12.6, driver
580.126.16, A100-SXM4-80GB (81920 MiB in nvidia-smi), one GPU, 250 GB advertised host RAM and
16 vCPUs. Captured all 95 resolved package pins, including dependency extras. Initial CPU checks
exposed that noninteractive SSH does not inherit the Pod's HF cache environment; the controller
now exports the persistent HF paths explicitly. The initial failing log is retained beside the
subsequent corrected run; no protected evaluation was performed.

## Original supplier comparison (network-volume preference)

Recommend Runpod Secure Cloud, on-demand, one A100 SXM 80 GB, a 500 GB standard network volume
in a data center with matching GPU capacity, and full SSH via public IP. Provider choice is still
open; advertised prices are not an account-specific quote or a capacity guarantee.

- [Runpod pricing](https://www.runpod.io/pricing): A100 SXM $1.59/hour, H100 PCIe $2.89/hour,
  H100 SXM $3.49/hour when checked today. Start with A100; an H100 requires a measured lower
  cost per example under the existing ruling.
- [Network volumes](https://docs.runpod.io/storage/network-volumes): standard storage
  $0.07/GB/month for the first TB, so 500 GB costs $35/month. Attach at deployment; GPU choices
  depend on the volume's data center. Keep checkpoint backups on the original box too.
- [SSH](https://docs.runpod.io/pods/configuration/use-ssh): select full SSH with public IP for
  file transfers. The basic SSH proxy lacks SCP/SFTP support.
- [Pod lifecycle](https://docs.runpod.io/pods/manage-pods): container storage is cleared on stop;
  network-volume data survives stop/termination. Put the repository, caches and checkpoints on
  the persistent mount while preserving the required absolute repository path. Verify the mount
  before uploading. Reacquiring GPU capacity after stopping is not guaranteed.
- [Lambda](https://lambda.ai/instances) is the fallback if matching Runpod capacity is unavailable.
  Its single-GPU list shows H100 PCIe 80 GB at $3.29/hour and H100 SXM at $4.29/hour; single A100
  offerings listed today are 40 GB, below our registered 80 GB spec. Verify persistent storage
  and availability before choosing it.

## Remaining launch work

1. Provider/account access, actual GPU quote, persistent mount, SSH and scoped Git push credentials.
2. Restore the approximately 35 GB training/COV ship set plus required model weights; validate
   identities and environment on the instance. Include the locally cached
   `hf-internal-testing/tiny-random-BertModel` configuration required by `run_checks.sh`; the ship
   list's weights rows omit it. Use an explicit training/COV allowlist rather than the example
   rsync's optional DEV/final paths. The repo lacks a validated fresh-install bootstrap.
3. Run instance checks, real CUDA smoke and resume checks; measure both batch rates and the
   reserved-encode allowance using permitted inputs. Replace the stale $25 storage allowance in
   `build_config.json` with the actual planned storage/transfer cost, include applicable charges,
   and validate the complete allocation under $1,000 before the E runs.
4. Both E arms, returned-checkpoint DEV-6, E1, pushed records, LoTTE manifest and same-day pin,
   then the registered gate and final recipe/review requirements before the fixed 200M build.
5. Final scoring still owes the executor pin/R6 flip and M9's amendment and executor wiring.
   Final evaluation and comparable serving costs remain milestone work.

## Restart dependency correction

STOP/restart rebuilds container disk. Run `scripts/m13_cloud_bootstrap.sh` after each restart
to restore rsync and revalidate the persistent pinned environment/CUDA allocation. GitHub's
scoped deploy key, when needed, must also be restored to `/root/.ssh` with mode 600; never put
a private key on the Pod volume, which ignores restrictive modes. An offline Git bundle can
deploy committed code without a remote private key. Stage-0 preflight now requires rsync.
The first repair retry stopped before transfer/training because rsync was missing; receipt
`results/m13_cloud_stage0_attempt2.json`. Original GPU failure evidence remains preserved.

## Completed validation

The fresh attempt on `ca9ccdc` passed all 266 SHA-256 checks, active CPU suites, both 512-token
shape smokes and both 600-step interrupted/resumed E smokes. Local backups verified; STOP
confirmed at 2026-09-11T19:12:35Z. The restart fingerprint repair excludes only loader clock
metadata; identity changes still refuse. Passing records live in `results/m13_cloud_stage0.json`
and `results/m13_cloud_resume_smoke.json`; prior failed receipts remain alongside them.
No registered E run, protected evaluation or full build has started. Measure remaining mandatory
encode allowances and the budget allocation before progressing through `EXECUTION.md`.

## Prepared local handoff and cheaper-path measurement

The isolated M13 checkout now reuses exact DEV-6/source/pool files and encoded binaries from
the original checkout through 131 individual file links, with nine independent metadata copies.
No broad directory links or M17 edits. Manifest: `work/m13cloud-dev6-links.json`. Parent verified
all destinations, sizes and link targets; metadata copies agree. No DEV-6 scoring has run.

`scripts/m13_encode_benchmark.py --dtype fp16` measures the actual LoTTE encoding kernel on the
same 1k/10k admitted training passages and writes `results/m13_encode_benchmark_fp16.json`.
Default fp32 and original receipts remain preserved. Independent source review and synthetic
argument/routing/refusal checks passed. This small addition stays local until the E run ends;
there is no concurrent GPU benchmark and no change to the active recipe. Its generic 10M
projection must be rescaled for the 2.7M LoTTE allowance, not read as LoTTE hours directly.
