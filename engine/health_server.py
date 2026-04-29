"""Tiny HTTP server so HF Spaces marks the container as healthy.

Lichess-bot is a long-running poller; HF Spaces expects something listening
on the configured PORT (default 7860).  We satisfy that with a 3-route
FastAPI app that doubles as a quick liveness/info page for visitors.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

START_TIME = time.time()
BOT_NAME = os.environ.get("BOT_NAME", "nick_p12_bot")
PORT = int(os.environ.get("PORT", "7860"))

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    uptime = int(time.time() - START_TIME)
    return f"""
<!doctype html>
<html><head><title>{BOT_NAME}</title></head>
<body style="font-family:sans-serif;max-width:640px;margin:40px auto;padding:0 20px">
  <h1>{BOT_NAME}</h1>
  <p>A Maia2 model fine-tuned on nick_p12's blitz games.
     Plays nick's openings (heavy Caro-Kann, 1.d4 systems) at roughly his strength.</p>
  <p>Challenge me on Lichess:
     <a href="https://lichess.org/?user={BOT_NAME}#friend">lichess.org/@/{BOT_NAME}</a></p>
  <hr>
  <p><small>Container uptime: {uptime}s &middot; port {PORT}</small></p>
</body></html>
"""


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "uptime_s": int(time.time() - START_TIME)})


@app.get("/info")
def info() -> JSONResponse:
    ckpt = Path(os.environ.get("MAIA_CHECKPOINT_PATH", "/app/model/nick_p12.pt"))
    return JSONResponse(
        {
            "bot": BOT_NAME,
            "checkpoint_present": ckpt.exists(),
            "checkpoint_size_mb": (ckpt.stat().st_size / 1e6) if ckpt.exists() else None,
            "uptime_s": int(time.time() - START_TIME),
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
