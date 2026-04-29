#!/bin/bash
# Container entry point for the nick_p12_bot HF Space.
#
# 1. Pull the latest fine-tuned checkpoint from the HF model repo.
# 2. Start the tiny health server on port 7860 (HF Spaces requirement).
# 3. Start lichess-bot, which invokes maia_engine.py per UCI move.
set -euo pipefail

echo "=== nick_p12_bot starting at $(date -Iseconds) ==="

# --- Required env vars (set in HF Space "Settings -> Variables and secrets") ---
: "${LICHESS_BOT_TOKEN:?LICHESS_BOT_TOKEN env var must be set (Lichess BOT account token)}"
: "${MAIA_HF_REPO:?MAIA_HF_REPO env var must be set, e.g. nikhileshp12/nick-p12-bot}"

# Engine resolves these as relative paths by default. lichess-bot cd's into
# its own dir before spawning us, so we must hard-pin absolute paths here.
export MAIA_CHECKPOINT_PATH=/app/model/nick_p12.pt
export MAIA_BOOK_DIR=/app/engine/book

# --- Pull the fine-tuned checkpoint ---
echo "Pulling checkpoint from huggingface://${MAIA_HF_REPO}..."
python /app/pull_model.py

# --- Patch the lichess-bot config with the token (config.yml is checked in
#     with token: "" so we never commit secrets) ---
sed -i "s|token: \"\"|token: \"${LICHESS_BOT_TOKEN}\"|" /app/engine/config.yml

# --- Health server in background (HF Spaces needs port 7860 listening) ---
echo "Starting health server on :7860..."
python /app/health_server.py &

# --- lichess-bot polls the Lichess API and runs games ---
echo "Starting lichess-bot..."
cd "${LICHESS_BOT_DIR}"
exec python lichess-bot.py --config /app/engine/config.yml
