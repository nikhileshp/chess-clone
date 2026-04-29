"""HTTP server for the nick_p12_bot HF Space.

Two roles:
  1. Health/info routes so HF Spaces marks the container as healthy and
     visitors get a useful landing page on port 7860.
  2. /predict — inference endpoint used by the docs "Play in browser"
     page. Loads MaiaEngine lazily on first request so container start
     stays fast and idle Spaces don't keep the model resident.

Lichess-bot is a long-running poller; HF Spaces expects something
listening on the configured PORT (default 7860). Same FastAPI app
covers both.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import chess
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

START_TIME = time.time()
BOT_NAME = os.environ.get("BOT_NAME", "nick_p12_bot")
PORT = int(os.environ.get("PORT", "7860"))

# Comma-separated list of allowed origins for /predict (CORS).
# Default permits the GitHub Pages docs site and local dev.
DEFAULT_ORIGINS = "https://nikhileshp.github.io,http://localhost:8000,http://127.0.0.1:8000"
CORS_ORIGINS = [o.strip() for o in os.environ.get("PREDICT_CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()]

logger = logging.getLogger("health_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Lazy MaiaEngine singleton ------------------------------------------------
# Loading the model is slow (~5-10s on CPU). We do it on first /predict call
# so /health stays cheap and unrelated container traffic doesn't pay for it.
_engine = None  # type: Optional[object]
_engine_lock = threading.Lock()
_engine_load_error: Optional[str] = None


def _get_engine():
    """Lazy-load and return a MaiaEngine instance. Thread-safe singleton."""
    global _engine, _engine_load_error
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        try:
            # Import here so /health works even if maia_engine import fails
            from maia_engine import MaiaEngine  # type: ignore

            logger.info("Loading MaiaEngine for /predict (first request)...")
            t0 = time.time()
            _engine = MaiaEngine()
            logger.info("MaiaEngine ready (%.1fs)", time.time() - t0)
            return _engine
        except Exception as e:  # noqa: BLE001
            _engine_load_error = f"{type(e).__name__}: {e}"
            logger.exception("Failed to load MaiaEngine")
            raise


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
  <p>Or play in browser (no account):
     <a href="https://nikhileshp.github.io/chess-clone/play.html">play.html</a> &mdash;
     posts to <code>POST /predict</code> below.</p>
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
            "engine_loaded": _engine is not None,
            "engine_load_error": _engine_load_error,
            "cors_origins": CORS_ORIGINS,
        }
    )


# --- /predict -----------------------------------------------------------------

class PredictRequest(BaseModel):
    fen: str = Field(..., description="Position FEN to move from.")
    opponent_elo: int = Field(2000, ge=600, le=3200, description="Opponent rating bucket hint.")


class PredictResponse(BaseModel):
    move: str
    san: str
    fen_after: str
    is_book: bool
    elapsed_ms: int


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Return the bot's next move for a given FEN.

    Stateless: each call sets MaiaEngine._board to the requested FEN,
    runs choose_move(), and returns the UCI + SAN. Safe against concurrent
    callers because we hold the engine lock for the duration of one request.
    """
    try:
        board = chess.Board(req.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Bad FEN: {e}")

    if board.is_game_over():
        raise HTTPException(status_code=400, detail="Position is already terminal.")

    engine = _get_engine()
    t0 = time.time()
    # MaiaEngine isn't designed for concurrent callers — it mutates ._board.
    # Serialize requests; HF Space free tier is single-replica anyway.
    with _engine_lock:
        engine._board = board  # type: ignore[attr-defined]
        engine._elo_oppo = engine._elo_to_bucket(req.opponent_elo)  # type: ignore[attr-defined]
        # Detect book hit by peeking — simpler than threading a return flag through.
        is_book = engine._book_pick() is not None and board.ply() < int(os.environ.get("MAIA_BOOK_MAX_PLIES", "30"))  # type: ignore[attr-defined]
        # _book_pick() doesn't mutate the board, but reset turn-state explicitly.
        engine._board = board  # type: ignore[attr-defined]
        move = engine.choose_move()  # type: ignore[attr-defined]

    if move == chess.Move.null():
        raise HTTPException(status_code=500, detail="Engine returned a null move.")

    san = board.san(move)
    board.push(move)
    return PredictResponse(
        move=move.uci(),
        san=san,
        fen_after=board.fen(),
        is_book=is_book,
        elapsed_ms=int((time.time() - t0) * 1000),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
