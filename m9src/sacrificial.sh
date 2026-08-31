#!/usr/bin/env bash
# Run a helper command with the maximum OOM penalty, so that if the box is ever short of memory
# the kernel kills THIS and not the 7-day trainer.
#
# The trainer's oom_score is ~1073 because oom_badness counts its 18.5 GB of memory-mapped
# corpus pages, even though those are reclaimable page cache and its real anonymous footprint is
# only ~2.1 GB. Lowering the trainer's score needs CAP_SYS_RESOURCE (no root here); RAISING a
# process's own score needs nothing. So helpers opt in to losing.
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
exec "$@"
