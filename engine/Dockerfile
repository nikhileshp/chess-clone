# HF Space deployment image for nick_p12_bot.
#
# At runtime the container does three things:
#   1. Pull the fine-tuned Maia2 checkpoint from a HF model repo
#   2. Start a tiny FastAPI server on port 7860 (HF Spaces requires a
#      health endpoint; lichess-bot itself doesn't need one)
#   3. Start lichess-bot, which polls the Lichess API and invokes our
#      maia_engine.py via UCI for every move
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    LICHESS_BOT_DIR=/opt/lichess-bot

# System deps (git for cloning lichess-bot)
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates stockfish \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# lichess-bot framework (cloned from its repo since not on PyPI)
RUN git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git ${LICHESS_BOT_DIR} \
 && pip install -r ${LICHESS_BOT_DIR}/requirements.txt

# Engine code + opening books
COPY maia_engine.py /app/engine/maia_engine.py
COPY time_mimic.py /app/engine/time_mimic.py
COPY config.yml /app/engine/config.yml
COPY start.sh /app/start.sh
COPY health_server.py /app/health_server.py
COPY pull_model.py /app/pull_model.py

# Opening books are tiny (<1MB total); ship them in the image
COPY book/ /app/engine/book/

# Time-mimic models are pulled from HF at container start (see pull_model.py).
# Pre-create the destination dir so pull_model.py's hf_hub_download can land
# files there without permission errors on read-only HF Spaces filesystems.
RUN mkdir -p /app/weights/time_model

RUN chmod +x /app/start.sh /app/engine/maia_engine.py

# Pre-create writable dirs HF Spaces expects
RUN mkdir -p /app/model /app/.cache/huggingface

EXPOSE 7860

CMD ["/app/start.sh"]
