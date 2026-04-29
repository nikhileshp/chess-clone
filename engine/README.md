# engine/ — Lichess bot for `nick_p12_bot`

A UCI engine wrapping the fine-tuned Maia2 model + Polyglot opening books,
deployed on a Hugging Face Space and connected to Lichess via the
`lichess-bot` framework.

```
Lichess players  ──challenges──▶  Lichess API
                                       │
                                       ▼
                                  HF Space (this code)
                                       │
                                       ├── lichess-bot (long poller)
                                       └── maia_engine.py (UCI adapter)
                                                │
                                                ├── opening book lookup (book/*.bin)
                                                └── Maia2 fine-tuned (./model/nick_p12.pt)
```

## One-time setup

You need three accounts: Lichess (bot), Hugging Face (model + Space), and
GitHub (already have).

### 1. Download your trained checkpoint from Colab

In Colab, after the fine-tune finished:

```python
from google.colab import files
files.download('/content/repo/colab_outputs/maia2_finetuned_best.pt')
```

This pulls a ~280 MB `.pt` file to your laptop.

### 2. Push the checkpoint to a HF model repo (one-shot)

Locally:

```bash
pip install huggingface_hub
huggingface-cli login   # paste a write-scope token from huggingface.co/settings/tokens

# from your downloaded file location:
python /home/nick/Projects/chess_clone/engine/upload_model.py \
    ~/Downloads/maia2_finetuned_best.pt \
    --repo nikhileshp12/nick-p12-bot
```

The repo is created automatically if it doesn't exist. Check it at
`https://huggingface.co/nikhileshp12/nick-p12-bot`.

You can re-run this command any time you have a new checkpoint — it
just overwrites `nick_p12.pt` in the repo.

### 3. Create the Lichess BOT account

1. Go to [lichess.org/signup](https://lichess.org/signup) and register a
   **brand new** account named `nick_p12_bot`.
   **Do not play any games** with it as a human — Lichess won't let an
   account become a bot once it has rated games.

2. Generate a Lichess API token for the bot account at
   [lichess.org/account/oauth/token](https://lichess.org/account/oauth/token/create).

   Token name: anything. **Required scopes**: `Play games with the bot API`
   (`bot:play`).

   **Save this token** — you'll paste it into HF Space secrets in step 5.

3. Upgrade the account to BOT class via the Lichess API
   (one-time, cannot be undone):

   ```bash
   curl -d '' https://lichess.org/api/bot/account/upgrade \
       -H "Authorization: Bearer YOUR_BOT_TOKEN"
   ```

   Response should be `{"ok":true}`. The account now has the BOT label and
   can play unlimited games via the API but cannot enter human-only events.

### 4. Smoke-test the engine locally (recommended)

Before deploying to HF, verify the engine works on your laptop:

```bash
cd /home/nick/Projects/chess_clone

# Copy your downloaded checkpoint into the right place
mkdir -p engine/model
cp ~/Downloads/maia2_finetuned_best.pt engine/model/nick_p12.pt

# Copy the Polyglot books built by scripts/build_book.py
mkdir -p engine/book
cp data/books/white.bin data/books/black.bin engine/book/

# Run the smoke test
uv run python engine/smoke_test.py --checkpoint engine/model/nick_p12.pt
```

Expected output: three `bestmove ...` lines plus `SMOKE TEST PASSED`.
First call takes ~10 sec (model load); subsequent calls are <1 sec.

### 5. Deploy to Hugging Face Space

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space):
   - Owner: `nikhileshp12`
   - Space name: `nick-p12-bot`
   - SDK: **Docker**
   - Hardware: **CPU basic** (free)
   - Visibility: Public (or Private — both work)

2. Add Space secrets at
   `https://huggingface.co/spaces/nikhileshp12/nick-p12-bot/settings`:
   - `LICHESS_BOT_TOKEN`: the bot token from step 3.2
   - `MAIA_HF_REPO`: `nikhileshp12/nick-p12-bot` (where the model was uploaded)

3. Push the engine/ directory (plus books) to the Space's git repo:

   ```bash
   cd /home/nick/Projects/chess_clone
   # Set up the Space remote — use HTTPS + your HF token for auth
   git clone https://huggingface.co/spaces/nikhileshp12/nick-p12-bot \
       /tmp/nick-p12-space
   # Copy engine files plus books into the Space repo
   cp -r engine/* /tmp/nick-p12-space/
   cp data/books/white.bin data/books/black.bin /tmp/nick-p12-space/book/
   cd /tmp/nick-p12-space
   git add -A
   git commit -m "Initial bot deploy"
   git push
   ```

   The push triggers an automatic Docker build on HF (~3–5 min).
   Watch the build logs in the Space "Logs" tab.

4. Once the Space says "Running", visit `https://nikhileshp12-nick-p12-bot.hf.space/`
   to see the health page, then check Lichess: your bot at
   `https://lichess.org/@/nick_p12_bot` should now accept challenges.

### 6. Test by challenging your bot

From any other Lichess account: `https://lichess.org/?user=nick_p12_bot#friend`
→ pick blitz 5+0 → "Challenge". Bot should accept within 1–2 sec
(or ~30 sec if Space was sleeping).

## Updating the bot later

| Want to... | Do this |
|---|---|
| Push a re-trained model | `python engine/upload_model.py /path/to/new.pt --repo nikhileshp12/nick-p12-bot`, then "Restart Space" in HF UI |
| Tweak time-control filters / temperature | Edit `engine/config.yml` → push to Space repo |
| Stop the bot | "Pause Space" in HF UI |
| Resume | "Restart Space" |
| Watch live games | Lichess profile page or `https://lichess.org/@/nick_p12_bot/tv` |

## Troubleshooting

- **HF build fails with `torch==2.4.0` not found**: HF builders may not support
  the exact version; widen the constraint in `requirements.txt` to `torch>=2.4,<2.6`.
- **Bot stays offline despite container running**: confirm `LICHESS_BOT_TOKEN`
  in Space secrets has `bot:play` scope. Re-issue if not.
- **Smoke test fails locally with module-not-found**: run from project root
  with the project's uv env activated (`uv run python engine/smoke_test.py`).
- **Cold-start too slow on free tier**: upgrade Space to "CPU upgrade" for $9/mo
  to keep the container always warm, or move to Fly.io for ~$5/mo.

## What's next

Once the bot has played ~30+ rated games on Lichess, its rating will stabilize
and we can:
1. Pull its loss-game PGNs via Lichess API
2. Identify positions where it blundered
3. Fine-tune a v2 weighted toward those positions (the bot learning from its own losses)
4. Re-upload checkpoint, restart Space → improved bot

Then: embed it in your github.io site (separate doc once we have a working bot).
