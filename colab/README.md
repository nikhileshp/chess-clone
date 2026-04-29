# Colab fine-tuning of Maia2 on your games

This directory holds everything needed to fine-tune the Maia2 base "blitz"
checkpoint on **nick_p12**'s PGN history, producing a model that plays
your openings and your style.

Repo: **`nikhileshp/chess-clone`** (public).

## One-time setup (local)

1. **Run the data pipeline locally** so the cleaned `.pgn.zst` lands in the repo:
   ```bash
   uv run python scripts/download_lichess.py
   uv run python scripts/download_chesscom.py
   uv run python scripts/merge_filter.py
   uv run python scripts/to_pgn_zst.py    # produces data/clean/all.pgn.zst
   ```

2. **Commit and push to GitHub.** The compressed PGN (~15–25 MB) goes into
   git — `.gitignore` is configured to allow exactly that file:
   ```bash
   git add data/clean/all.pgn.zst colab/ scripts/ pyproject.toml uv.lock README.md
   git commit -m "Initial pipeline + cleaned game data"
   git remote add origin git@github.com:nikhileshp/chess-clone.git   # one-time
   git push -u origin main
   ```

That's it. No Drive, no manual upload. Colab just `git clone`s the repo at
the start of every session.

## Launching the notebook

```
https://colab.research.google.com/github/nikhileshp/chess-clone/blob/main/colab/train_maia2.ipynb
```

Once Colab opens it:

1. **Runtime → Change runtime type → GPU → A100 (or L4 / V100)**.
   Colab Pro should let you select premium GPUs.
2. **Runtime → Run all**.
3. The first cells install deps and mount your Drive (auth popup).
4. Subsequent cells fine-tune; total wall-clock is **~1–3 hours** on an A100.
5. The final cell saves outputs to `/content/repo/colab_outputs/`:
   - `maia2_finetuned_<timestamp>.pt` — model checkpoint
   - `metrics.json` — per-epoch training/val metrics
   - `eval_sample.txt` — sanity check on held-out positions

   To pull these back to your laptop, the notebook offers a one-click
   `files.download(...)` cell. Or run a follow-up cell that pushes them
   to a `weights` branch of the repo (instructions inline in the notebook).

## Two-plan structure

The notebook has two strategies in sequence; use whichever works:

- **Plan A — `maia2.train.run()` with fine-tune YAML.** The package ships its
  own training loop. We load a pretrained blitz checkpoint and continue
  training on your data with a low LR. Cleanest if their YAML supports the
  knobs we need.
- **Plan B — Custom PyTorch fine-tune loop.** If Plan A doesn't expose
  resume-from-checkpoint or per-layer LR control, we drop to a hand-written
  training loop on top of `model.from_pretrained()`. More code, more control.

Plan A is run first. If it fails or is missing knobs, Plan B is uncommented.

## Outputs you'll pull back

- `checkpoints/maia2_finetuned_<ts>.pt` — fine-tuned PyTorch state dict
- `metrics.json` — training/validation move-prediction accuracy per epoch
- `eval_sample.txt` — model's top-3 predictions on 50 random positions from
  your held-out games (sanity-check that it actually picks your moves)

## After you have the checkpoint

You can either:
- Use `maia2.inference` directly with your fine-tuned weights for analysis
- Wrap it as a UCI engine (small adapter, ~100 lines of Python) and play
  against it in any chess GUI
- Combine with the Polyglot opening books from this repo for opening fidelity

## Troubleshooting

- **"No GPU available"**: Pro account but no premium GPU offered → reload
  Colab and try again, or pick L4 if A100 is gated by usage limits.
- **OOM on A100**: shouldn't happen with 40 GB + Maia2's modest size; if it
  does, halve the batch size in the YAML.
- **`pip install maia2` fails**: the package may pin an older NumPy; the
  notebook handles this by force-reinstalling NumPy 2.1.3 first.
- **`train.run(cfg)` errors**: switch to Plan B.
