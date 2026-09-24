#!/bin/bash
# Container entry point for the nick_p12_bot HF Space.
#
# 1. Start the tiny health server on port 7860 (HF Spaces requirement) right
#    away, so the Space is reachable while the model downloads.
# 2. Pull the latest fine-tuned checkpoint from the HF model repo.
# 3. Start lichess-bot, which invokes maia_engine.py per UCI move.
#
# Both long-running processes run under a supervisor loop. HF Spaces does NOT
# restart a container whose entrypoint exits: it just flips the Space to
# RUNTIME_ERROR ("Exit code: 0") and the bot stays offline until someone
# restarts it by hand (this is what took the bot down on 2026-05-15).
# lichess-bot only retries network errors internally; any other exception
# escaping its main loop is logged and it returns cleanly (exit 0), see
# lib/lichess_bot.py::start_program. So this script must be the thing that
# never exits.

RESTART_DELAY_S="${RESTART_DELAY_S:-15}"          # first restart delay (integer seconds)
RESTART_DELAY_MAX_S="${RESTART_DELAY_MAX_S:-300}" # backoff cap for crash loops
STABLE_RUN_S="${STABLE_RUN_S:-120}"               # a child alive this long resets the backoff
SHUTDOWN_GRACE_S="${SHUTDOWN_GRACE_S:-8}"         # docker's default stop timeout is 10s
PID_DIR="${PID_DIR:-/tmp}"

# Commands to supervise; main() sets the real ones, tests override them.
HEALTH_CMD=()
BOT_CMD=()
HEALTH_LOOP_PID=""
BOT_LOOP_PID=""
PREP_PID=""

log() { echo "=== $(date -Iseconds) $* ==="; }

# _alive PID... : true if any of the given PIDs still exists.
_alive() {
  local p
  for p in "$@"; do
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then return 0; fi
  done
  return 1
}

# run_forever NAME CMD... : run CMD and restart it whenever it exits (any code),
# backing off exponentially while it keeps dying quickly. The child's PID is
# written to PID_DIR/NAME.pid so shutdown() can signal the real process.
run_forever() {
  local name="$1"; shift
  local pid rc started elapsed delay="$RESTART_DELAY_S"
  while true; do
    log "[$name] starting"
    started=$(date +%s)
    # bash starts async children with SIGINT ignored, and python inherits that
    # until lichess-bot installs its own handler (after its slow imports).
    # Reset it so a forwarded SIGINT always lands; exec keeps $! == real PID.
    ( trap - INT; exec "$@" ) &
    pid=$!
    echo "$pid" > "${PID_DIR}/${name}.pid" || log "[$name] pidfile write failed (pid $pid)"
    rc=0
    wait "$pid" || rc=$?
    : > "${PID_DIR}/${name}.pid" 2>/dev/null || true
    elapsed=$(( $(date +%s) - started ))
    if [ "$elapsed" -ge "$STABLE_RUN_S" ]; then delay="$RESTART_DELAY_S"; fi
    log "[$name] exited rc=$rc after ${elapsed}s; restarting in ${delay}s"
    sleep "$delay" || true
    if [ "$elapsed" -lt "$STABLE_RUN_S" ]; then
      delay=$(( delay * 2 > RESTART_DELAY_MAX_S ? RESTART_DELAY_MAX_S : delay * 2 ))
    fi
  done
}

# _kill_orphan NAME SIGNAL : if a loop died but its child is still running,
# stop the child so the respawned loop does not start a second copy (two
# lichess-bots on one token would fight over the event stream).
_kill_orphan() {
  local pid i
  pid=$(cat "${PID_DIR}/$1.pid" 2>/dev/null || true)
  if ! _alive "$pid"; then return 0; fi
  log "[$1] loop died; stopping orphaned child $pid"
  kill "-$2" "$pid" 2>/dev/null || true
  for i in 1 2 3 4 5; do
    if ! _alive "$pid"; then return 0; fi
    sleep 1
  done
  kill -KILL "$pid" 2>/dev/null || true
}

# ensure_loops : (re)spawn each supervisor loop that is not running.
ensure_loops() {
  if [ "${#HEALTH_CMD[@]}" -gt 0 ] && ! _alive "$HEALTH_LOOP_PID"; then
    if [ -n "$HEALTH_LOOP_PID" ]; then _kill_orphan health_server TERM; log "health loop respawning"; fi
    run_forever health_server "${HEALTH_CMD[@]}" &
    HEALTH_LOOP_PID=$!
  fi
  if [ "${#BOT_CMD[@]}" -gt 0 ] && ! _alive "$BOT_LOOP_PID"; then
    if [ -n "$BOT_LOOP_PID" ]; then _kill_orphan lichess-bot INT; log "bot loop respawning"; fi
    run_forever lichess-bot "${BOT_CMD[@]}" &
    BOT_LOOP_PID=$!
  fi
}

# Forward a container stop (SIGTERM from HF on restart/sleep/redeploy) to the
# children: lichess-bot only handles SIGINT, so it gets that; everything else
# gets SIGTERM. Wait (bounded) for them, then exit 0.
shutdown() {
  trap - TERM INT
  log "stop signal received; shutting down"
  local p bot_pid health_pid waited=0
  for p in "$HEALTH_LOOP_PID" "$BOT_LOOP_PID" "$PREP_PID"; do
    if [ -n "$p" ]; then kill -TERM "$p" 2>/dev/null || true; fi
  done
  bot_pid=$(cat "${PID_DIR}/lichess-bot.pid" 2>/dev/null || true)
  health_pid=$(cat "${PID_DIR}/health_server.pid" 2>/dev/null || true)
  if [ -n "$bot_pid" ]; then kill -INT "$bot_pid" 2>/dev/null || true; fi
  if [ -n "$health_pid" ]; then kill -TERM "$health_pid" 2>/dev/null || true; fi
  while [ "$waited" -lt "$SHUTDOWN_GRACE_S" ] && _alive "$bot_pid" "$health_pid"; do
    sleep 1; waited=$((waited + 1))
  done
  for p in "$bot_pid" "$health_pid"; do
    if [ -n "$p" ]; then kill -KILL "$p" 2>/dev/null || true; fi
  done
  log "shutdown complete"
  exit 0
}

# supervise : keep both loops alive forever. `wait -n` returns when a loop
# dies or a trapped signal arrives (the trap then runs), and returns at once
# when there is nothing to wait for, hence the nap.
supervise() {
  while true; do
    ensure_loops
    wait -n 2>/dev/null || true
    sleep 1
  done
}

prepare() {
  # --- Required env vars (set in HF Space "Settings -> Variables and secrets") ---
  : "${LICHESS_BOT_TOKEN:?LICHESS_BOT_TOKEN env var must be set (Lichess BOT account token)}"
  : "${MAIA_HF_REPO:?MAIA_HF_REPO env var must be set, e.g. nikhileshp12/nick-p12-bot}"

  # --- Pull the fine-tuned checkpoint + time-mimic LightGBM models ---
  # Retry forever with capped backoff: a cold start after a Space sleep must
  # not turn one flaky HF fetch into a dead Space (the health loop is already
  # up, so the Space stays reachable meanwhile). The pull runs in the
  # background so a stop signal can interrupt the wait.
  local attempt=1 delay=30 rc
  while true; do
    log "pulling checkpoint + time-mimic models from huggingface://${MAIA_HF_REPO} (attempt $attempt)"
    python /app/pull_model.py &
    PREP_PID=$!
    rc=0
    wait "$PREP_PID" || rc=$?
    PREP_PID=""
    if [ "$rc" -eq 0 ]; then break; fi
    log "model pull failed rc=$rc; retrying in ${delay}s"
    sleep "$delay" || true
    attempt=$((attempt + 1))
    delay=$(( delay * 2 > 300 ? 300 : delay * 2 ))
  done

  # --- Patch the lichess-bot config with the token (config.yml is checked in
  #     with token: "" so we never commit secrets) ---
  sed -i "s|token: \"\"|token: \"${LICHESS_BOT_TOKEN}\"|" /app/engine/config.yml
}

main() {
  set -euo pipefail
  # We are PID 1: a signal with no handler installed is simply discarded, so
  # install the stop handler before anything slow runs.
  trap shutdown TERM INT
  log "nick_p12_bot starting"

  # Engine resolves these as relative paths by default. lichess-bot cd's into
  # its own dir before spawning us, so we must hard-pin absolute paths here.
  # Exported before any child starts so every process inherits them.
  export MAIA_CHECKPOINT_PATH=/app/model/nick_p12.pt
  export MAIA_BOOK_DIR=/app/engine/book
  export MAIA_TIME_MODEL_DIR=/app/weights/time_model

  # Health server first: HF needs :7860 listening, and the keepalive pings
  # must keep succeeding while the model downloads.
  HEALTH_CMD=(python /app/health_server.py)
  ensure_loops

  prepare

  BOT_CMD=(python lichess-bot.py --config /app/engine/config.yml)
  cd "${LICHESS_BOT_DIR}"
  supervise
}

# Only run when executed, not when sourced (tests source this file).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
