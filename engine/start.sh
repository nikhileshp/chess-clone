#!/bin/bash
# Container entry point for the nick_p12_bot HF Space.
#
# 1. Pull the latest fine-tuned checkpoint from the HF model repo.
# 2. Start the tiny health server on port 7860 (HF Spaces requirement).
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

RESTART_DELAY_S="${RESTART_DELAY_S:-15}"
SHUTDOWN_GRACE_S="${SHUTDOWN_GRACE_S:-20}"
PID_DIR="${PID_DIR:-/tmp}"

# Commands to supervise; main() sets the real ones, tests override them.
HEALTH_CMD=()
BOT_CMD=()

log() { echo "=== $(date -Iseconds) $* ==="; }

# run_forever NAME CMD... : run CMD, and whenever it exits (any code) restart
# it after RESTART_DELAY_S. The child's PID is written to PID_DIR/NAME.pid so
# shutdown() can signal the real process, not just this loop.
run_forever() {
  local name="$1"; shift
  local pid rc
  while true; do
    log "[$name] starting"
    "$@" &
    pid=$!
    echo "$pid" > "${PID_DIR}/${name}.pid"
    rc=0
    wait "$pid" || rc=$?
    log "[$name] exited rc=$rc; restarting in ${RESTART_DELAY_S}s"
    sleep "$RESTART_DELAY_S"
  done
}

# Forward a container stop (SIGTERM from HF on restart/sleep/redeploy) to the
# children. lichess-bot only handles SIGINT, so it gets that; the health server
# gets SIGTERM. Then wait (bounded) for them to go away and exit 0.
shutdown() {
  trap - TERM INT
  log "stop signal received; shutting down"
  kill -TERM "$HEALTH_LOOP_PID" "$BOT_LOOP_PID" 2>/dev/null || true
  local bot_pid health_pid
  bot_pid=$(cat "${PID_DIR}/lichess-bot.pid" 2>/dev/null || true)
  health_pid=$(cat "${PID_DIR}/health_server.pid" 2>/dev/null || true)
  [ -n "$bot_pid" ] && kill -INT "$bot_pid" 2>/dev/null || true
  [ -n "$health_pid" ] && kill -TERM "$health_pid" 2>/dev/null || true
  local waited=0
  while [ -n "$bot_pid" ] && kill -0 "$bot_pid" 2>/dev/null && [ "$waited" -lt "$SHUTDOWN_GRACE_S" ]; do
    sleep 1; waited=$((waited + 1))
  done
  [ -n "$bot_pid" ] && kill -KILL "$bot_pid" 2>/dev/null || true
  [ -n "$health_pid" ] && kill -KILL "$health_pid" 2>/dev/null || true
  log "shutdown complete"
  exit 0
}

# supervise: start both loops, install the stop handler, and block forever.
supervise() {
  run_forever health_server "${HEALTH_CMD[@]}" &
  HEALTH_LOOP_PID=$!
  run_forever lichess-bot "${BOT_CMD[@]}" &
  BOT_LOOP_PID=$!
  trap shutdown TERM INT
  # `wait` returns early when a trapped signal arrives (the trap then runs), and
  # returns 0 at once if there is nothing left to wait for, so loop with a nap.
  while true; do
    wait || true
    if ! kill -0 "$HEALTH_LOOP_PID" 2>/dev/null && ! kill -0 "$BOT_LOOP_PID" 2>/dev/null; then
      log "both supervisor loops are gone; exiting"
      exit 1
    fi
    sleep 1
  done
}

prepare() {
  log "nick_p12_bot starting"

  # --- Required env vars (set in HF Space "Settings -> Variables and secrets") ---
  : "${LICHESS_BOT_TOKEN:?LICHESS_BOT_TOKEN env var must be set (Lichess BOT account token)}"
  : "${MAIA_HF_REPO:?MAIA_HF_REPO env var must be set, e.g. nikhileshp12/nick-p12-bot}"

  # Engine resolves these as relative paths by default. lichess-bot cd's into
  # its own dir before spawning us, so we must hard-pin absolute paths here.
  export MAIA_CHECKPOINT_PATH=/app/model/nick_p12.pt
  export MAIA_BOOK_DIR=/app/engine/book
  export MAIA_TIME_MODEL_DIR=/app/weights/time_model

  # --- Pull the fine-tuned checkpoint + time-mimic LightGBM models ---
  log "pulling checkpoint + time-mimic models from huggingface://${MAIA_HF_REPO}"
  # Retry: a cold start after a Space sleep must not die on one flaky HF fetch.
  local attempt
  for attempt in 1 2 3 4 5; do
    if python /app/pull_model.py; then break; fi
    if [ "$attempt" -eq 5 ]; then log "model pull failed 5 times; giving up"; exit 1; fi
    log "model pull failed (attempt $attempt); retrying in 30s"
    sleep 30
  done

  # --- Patch the lichess-bot config with the token (config.yml is checked in
  #     with token: "" so we never commit secrets) ---
  sed -i "s|token: \"\"|token: \"${LICHESS_BOT_TOKEN}\"|" /app/engine/config.yml
}

main() {
  set -euo pipefail
  prepare
  HEALTH_CMD=(python /app/health_server.py)
  BOT_CMD=(python lichess-bot.py --config /app/engine/config.yml)
  cd "${LICHESS_BOT_DIR}"
  supervise
}

# Only run when executed, not when sourced (tests source this file).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
