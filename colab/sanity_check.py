"""Side-by-side sanity check: base Maia2 vs fine-tuned, on 50 held-out positions.

Runs after finetune_run.py has executed, reusing its `val_rows`, `m`, `device`,
`cfg`, `ALL_MOVES_DICT`, `board_to_tensor`. Loads a fresh base model for comparison.

Usage from a Colab cell:
    exec(open('/content/repo/colab/sanity_check.py').read())
"""
import random

import chess
import torch

# Validate required globals exist
for _name in ("val_rows", "m", "device", "cfg", "ALL_MOVES_DICT", "board_to_tensor"):
    if _name not in dir():
        raise RuntimeError(
            f"Required global '{_name}' is missing. Run finetune_run.py first."
        )

from maia2 import model as _model_mod  # noqa: E402

# Load a fresh BASE Maia2-blitz so we can compare apples-to-apples
print("Loading fresh base Maia2-blitz for comparison...")
m_base = _model_mod.from_pretrained(type=cfg.pretrained_type, device="gpu")
m_base.eval()
m.eval()

# Build inverse move dict for printing UCI strings
INV_MOVES = {v: k for k, v in ALL_MOVES_DICT.items()}

# Pick 50 random held-out samples
rng = random.Random(0)  # fixed seed so results are reproducible across reruns
N_SAMPLES = 50
picks = rng.sample(val_rows, min(N_SAMPLES, len(val_rows)))


def predict_top3(model, board_tensor, elo_self, elo_oppo):
    with torch.no_grad():
        x = board_tensor.unsqueeze(0).to(device)
        es = torch.tensor([elo_self], device=device)
        eo = torch.tensor([elo_oppo], device=device)
        out = model(x, es, eo)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        top3 = logits[0].topk(3).indices.tolist()
    return [INV_MOVES[i] for i in top3]


print(f"\n{'='*100}")
print(f"50-position sanity check  ({cfg.user})")
print(f"{'='*100}")
print(f"{'#':>3} {'actual':<7} | {'base top-3':<23} | {'finetuned top-3':<23} | base | ft")
print("-" * 100)

base_t1 = base_t3 = ft_t1 = ft_t3 = 0

for i, (fen, uci, elo_s, elo_o, _aw) in enumerate(picks):
    board = chess.Board(fen)
    x = board_to_tensor(board)

    base_top3 = predict_top3(m_base, x, elo_s, elo_o)
    ft_top3 = predict_top3(m, x, elo_s, elo_o)

    b_t1 = uci == base_top3[0]
    b_t3 = uci in base_top3
    f_t1 = uci == ft_top3[0]
    f_t3 = uci in ft_top3
    base_t1 += b_t1
    base_t3 += b_t3
    ft_t1 += f_t1
    ft_t3 += f_t3

    base_str = " ".join(base_top3)
    ft_str = " ".join(ft_top3)
    base_mark = "✓1" if b_t1 else ("✓3" if b_t3 else "✗ ")
    ft_mark = "✓1" if f_t1 else ("✓3" if f_t3 else "✗ ")
    print(f"{i+1:>3} {uci:<7} | {base_str:<23} | {ft_str:<23} | {base_mark:<4} | {ft_mark}")

n = len(picks)
print("-" * 100)
print(f"\n{'':<10}{'top-1':>10}{'top-3':>10}{'delta-1':>10}{'delta-3':>10}")
print(
    f"{'base':<10}{base_t1/n:>10.4f}{base_t3/n:>10.4f}{'':>10}{'':>10}"
)
print(
    f"{'finetuned':<10}{ft_t1/n:>10.4f}{ft_t3/n:>10.4f}"
    f"{(ft_t1-base_t1)/n*100:>+9.1f}pp"
    f"{(ft_t3-base_t3)/n*100:>+9.1f}pp"
)

m.train()
print("\nDone. Free up the base model with `del m_base; torch.cuda.empty_cache()` if memory tight.")
