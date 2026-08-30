#!/usr/bin/env bash
# One-line M9 chain status. Reads the LAST chain entry, never greps the log's history --
# run_m9_stage.sh appends, so a `grep -q "START m9s2"` matches a previous run's marker and a
# monitor built on it reports a stage that has not started.
cd /home/dylan/asymetric-dual-encoders
echo "now $(date +%H:%M:%S)   chain: $(tail -1 logs/m9_chain.log | cut -c1-120)"
cur=$(tail -1 logs/m9_chain.log | grep -oE '(START|GATE) [a-z0-9_:]+' | awk '{print $2}')
[ -n "${cur:-}" ] && [ -f "logs/m9_arm_${cur}.log" ] && echo "  arm: $(tail -1 logs/m9_arm_${cur}.log | cut -c1-120)"
echo "  artifacts: $(ls results/m9_screen_m9s*.json 2>/dev/null | wc -l) arms written"
