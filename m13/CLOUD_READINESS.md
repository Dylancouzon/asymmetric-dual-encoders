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

## Supplier and deployment

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
   identities and environment on the instance. The repo lacks a validated fresh-install bootstrap.
3. Run instance checks, real CUDA smoke and resume checks; measure both batch rates and the
   reserved-encode allowance using permitted inputs. Replace the stale $25 storage allowance in
   `build_config.json` with the actual planned storage/transfer cost, include applicable charges,
   and validate the complete allocation under $1,000 before the E runs.
4. Both E arms, returned-checkpoint DEV-6, E1, pushed records, LoTTE manifest and same-day pin,
   then the registered gate and final recipe/review requirements before the fixed 200M build.
5. Final scoring still owes the executor pin/R6 flip and M9's amendment and executor wiring.
   Final evaluation and comparable serving costs remain milestone work.
