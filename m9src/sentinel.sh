#!/usr/bin/env bash
# Stall sentinel for the M9.3 build.
#
# The build monitor tails logs, so it only speaks when the run WRITES something. The failure
# that costs three days is the opposite one: everything stops and nothing is written ever
# again. Silence is indistinguishable from health.
#
# This inverts that. It polls state and prints a line ONLY when something is wrong or the run
# has ended. Healthy ticks are silent, so every line it emits is actionable.
set -u
cd /home/dylan/asymetric-dual-encoders || exit 1
RUN=work/m9long
HB=$RUN/heartbeat.json
PERIOD=300           # 5 min
STALL_S=1500         # 25 min with no step advance == wedged (ckpt interval is ~22 min)
HB_S=1500            # 25 min of heartbeat silence
EVAL_S=21600         # 6 h with no new eval (schedule is 1.8 h nominal, 3.6 h at the floor)
MIN_GB=50

py() { .venv/bin/python -c "$1" 2>/dev/null; }
last_step=""; last_step_at=$(date +%s); last_evals=""; last_evals_at=$(date +%s)

while true; do
  sleep $PERIOD
  now=$(date +%s)

  # The run ending is itself an event worth waking someone for -- including a clean finish.
  if [ -f "$RUN/terminal.json" ]; then
    echo "RUN ENDED: terminal.json present -- $(tr -d '\n' < "$RUN/terminal.json" | cut -c1-300)"
    exit 0
  fi

  trainer=$(pgrep -c -f "\.venv/bin/python m9src/longrun\.py" || true)
  watchdog=$(pgrep -c -f "\.venv/bin/python m9src/watchdog\.py" || true)
  guardian=$(pgrep -c -f "bash m9src/guardian\.sh" || true)

  # A dead trainer with no terminal marker is the silent-death case. The watchdog should
  # restart it; say so anyway, because "the watchdog should" is the assumption under test.
  [ "$trainer" -eq 0 ] && echo "ALERT: no trainer process and no terminal.json (watchdog should restart; verify)"
  [ "$trainer" -gt 1 ] && echo "ALERT: $trainer trainer processes -- possible double writer"
  [ "$watchdog" -eq 0 ] && echo "ALERT: no watchdog (guardian should relaunch within 60s; verify)"
  [ "$watchdog" -gt 1 ] && echo "ALERT: $watchdog watchdogs running -- must never happen"
  [ "$guardian" -eq 0 ] && echo "ALERT: guardian is gone -- watchdog is now unsupervised"

  [ -f "$HB" ] || { echo "ALERT: heartbeat.json missing"; continue; }
  hb_age=$(( now - $(stat -c %Y "$HB") ))
  [ "$hb_age" -gt "$HB_S" ] && echo "ALERT: heartbeat ${hb_age}s old (>${HB_S}s) -- trainer wedged?"

  step=$(py "import json;print(json.load(open('$HB')).get('step',''))")
  evals=$(py "import json;print(json.load(open('$HB')).get('evals',''))")
  rate=$(py "import json;d=json.load(open('$HB'));print(int(d.get('tok_per_s') or 0))")
  floor=$(py "import json;d=json.load(open('$HB'));print(int(d.get('floor') or 0))")
  state=$(py "import json;print(json.load(open('$HB')).get('state',''))")

  if [ -n "$step" ] && [ "$step" != "$last_step" ]; then last_step=$step; last_step_at=$now; fi
  if [ -n "$step" ] && [ $(( now - last_step_at )) -gt "$STALL_S" ] && [ "$state" != "eval" ]; then
    echo "ALERT: step stuck at $step for $(( (now-last_step_at)/60 )) min (state=$state)"
    last_step_at=$now
  fi

  if [ -n "$evals" ] && [ "$evals" != "$last_evals" ]; then last_evals=$evals; last_evals_at=$now; fi
  if [ -n "$evals" ] && [ $(( now - last_evals_at )) -gt "$EVAL_S" ]; then
    echo "ALERT: no new eval for $(( (now-last_evals_at)/3600 ))h (count stuck at $evals)"
    last_evals_at=$now
  fi

  if [ "${floor:-0}" -gt 0 ] && [ "${rate:-0}" -gt 0 ] && [ "$rate" -lt "$floor" ]; then
    echo "ALERT: throughput $rate tok/s below floor $floor -- registered collapse stop is imminent"
  fi

  free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  [ "${free_gb:-999}" -lt "$MIN_GB" ] && echo "ALERT: only ${free_gb}GB free (<${MIN_GB}GB)"
done
