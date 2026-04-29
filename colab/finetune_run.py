"""Self-contained Maia2 fine-tune runner.

Designed to be exec()'d from a Colab cell after the notebook has already:
  * pip-installed maia2 + chess + pyzstd
  * loaded the pretrained model into a global `m`
  * set `device`, `cfg` (FineTuneConfig)

Usage from any Colab cell:
    exec(open('/content/repo/colab/finetune_run.py').read())

This avoids the "notebook on disk vs browser cache" confusion entirely.
"""
import io
import json
import random
import time
from pathlib import Path

import chess
import chess.pgn
import pyzstd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from maia2.utils import board_to_tensor, get_all_possible_moves

# Globals expected from the notebook session
m  # noqa: F821 — pretrained MAIA2Model on GPU
cfg  # noqa: F821 — FineTuneConfig
device  # noqa: F821 — torch.device('cuda')

ALL_MOVES_DICT = {mv: i for i, mv in enumerate(get_all_possible_moves())}
N_MOVES = len(ALL_MOVES_DICT)
print(f"Action space: {N_MOVES} moves")


def elo_to_bucket(elo: int) -> int:
    """Match Maia2's `map_to_category` from maia2/utils.py.

    11 buckets (indices 0..10):
      0  <1100
      1  1100-1199
      ...
      9  1900-1999
      10 >=2000
    """
    if elo < 1100:
        return 0
    if elo >= 2000:
        return 10
    return ((elo - 1100) // 100) + 1


# Sanity-check against the model's actual embedding size
_n_buckets = m.elo_embedding.num_embeddings
print(f"Model elo_embedding has {_n_buckets} buckets; our mapping produces 0..10.")
assert _n_buckets == 11, f"Unexpected bucket count {_n_buckets}; bucketing may have changed upstream."

USER = cfg.user.lower()


def parse_pgn_zst(path: str, max_games: int | None = None):
    samples = []
    with pyzstd.open(path, "rb") as fz:
        text = fz.read().decode("utf-8", errors="replace")
    buf = io.StringIO(text)
    n_games = 0
    pbar = tqdm(desc="parsing games", unit="g")
    while True:
        game = chess.pgn.read_game(buf)
        if game is None:
            break
        n_games += 1
        pbar.update(1)
        if n_games % 500 == 0:
            pbar.set_postfix(samples=len(samples))
        if max_games and n_games > max_games:
            break
        h = game.headers
        white = (h.get("White") or "").lower()
        black = (h.get("Black") or "").lower()
        user_is_white = white == USER
        if not user_is_white and black != USER:
            continue
        try:
            elo_self = int(h.get("WhiteElo" if user_is_white else "BlackElo", "?"))
            elo_oppo = int(h.get("BlackElo" if user_is_white else "WhiteElo", "?"))
        except ValueError:
            continue
        result = h.get("Result", "*")
        if result == "1-0":
            user_score = 1 if user_is_white else 0
        elif result == "0-1":
            user_score = 0 if user_is_white else 1
        elif result == "1/2-1/2":
            user_score = 0.5
        else:
            continue

        board = game.board()
        node = game
        while node.variations:
            node = node.variation(0)
            move = node.move
            user_to_move = (board.turn == chess.WHITE) == user_is_white
            if user_to_move:
                uci = move.uci()
                if uci in ALL_MOVES_DICT:
                    aw = 1 if user_score == 1 else (0 if user_score == 0.5 else -1)
                    samples.append(
                        (board.fen(), uci, elo_to_bucket(elo_self), elo_to_bucket(elo_oppo), aw)
                    )
            board.push(move)
    pbar.close()
    return samples, n_games


print("Parsing PGN...")
_t0 = time.time()
_samples, _n_games = parse_pgn_zst(cfg.pgn_zst_path)
print(f"  {_n_games} games | {len(_samples)} user-move samples | {time.time()-_t0:.1f}s")


class UserMoveDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        fen, uci, elo_s, elo_o, aw = self.rows[i]
        board = chess.Board(fen)
        x = board_to_tensor(board)
        y = ALL_MOVES_DICT[uci]
        return x, y, elo_s, elo_o, aw


random.Random(cfg.seed).shuffle(_samples)
_n_val = int(len(_samples) * cfg.val_fraction)
_train_rows = _samples[_n_val:]
_val_rows = _samples[:_n_val]
print(f"  train={len(_train_rows)} val={len(_val_rows)}")

train_dl = DataLoader(
    UserMoveDataset(_train_rows),
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
val_dl = DataLoader(
    UserMoveDataset(_val_rows),
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

m.train()
opt = torch.optim.AdamW(m.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=cfg.warmup_steps)


def run_eval(loader, label):
    m.eval()
    correct1, correct3, total = 0, 0, 0
    with torch.no_grad():
        for x, y, es, eo, _ in tqdm(loader, desc=f"eval {label}", leave=False):
            x = x.to(device)
            y = y.to(device)
            es = es.to(device)
            eo = eo.to(device)
            out = m(x, es, eo)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            top3 = logits.topk(3, dim=-1).indices
            correct1 += (top3[:, 0] == y).sum().item()
            correct3 += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)
    print(f"  {label}: top-1 {correct1/total:.4f} | top-3 {correct3/total:.4f} | n={total}")
    m.train()
    return correct1 / total


print("Baseline (pretrained Maia2-blitz, no fine-tune):")
base_top1 = run_eval(val_dl, "val")

best_val = 0.0
metrics = {"epochs": []}
Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

for epoch in range(cfg.epochs):
    print(f"\n=== Epoch {epoch+1}/{cfg.epochs} ===")
    _t0 = time.time()
    running, steps = 0.0, 0
    pbar = tqdm(train_dl, desc=f"epoch {epoch+1}")
    for x, y, es, eo, _ in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        es = es.to(device, non_blocking=True)
        eo = eo.to(device, non_blocking=True)
        out = m(x, es, eo)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()
        steps += 1
        if steps % 20 == 0:
            pbar.set_postfix(loss=f"{running/steps:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
    train_loss = running / max(steps, 1)
    val_top1 = run_eval(val_dl, "val")
    metrics["epochs"].append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_top1": val_top1,
            "seconds": time.time() - _t0,
        }
    )
    if val_top1 > best_val:
        best_val = val_top1
        ckpt = f"{cfg.output_dir}/maia2_finetuned_best.pt"
        torch.save(
            {
                "state_dict": m.state_dict(),
                "val_top1": val_top1,
                "baseline_top1": base_top1,
                "config": {k: v for k, v in vars(cfg).items()},
                "epoch": epoch + 1,
            },
            ckpt,
        )
        print(f"  saved best -> {ckpt}")

ts = int(time.time())
final_path = f"{cfg.output_dir}/maia2_finetuned_{ts}.pt"
torch.save(
    {
        "state_dict": m.state_dict(),
        "baseline_top1": base_top1,
        "best_val_top1": best_val,
        "config": {k: v for k, v in vars(cfg).items()},
    },
    final_path,
)
print(f"\nFinal checkpoint: {final_path}")
print(f"Baseline val top-1: {base_top1:.4f}")
print(f"Best fine-tuned val top-1: {best_val:.4f}")
print(f"Improvement: +{(best_val - base_top1)*100:.2f} pp")

with open(f"{cfg.output_dir}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
