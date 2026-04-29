"""Helpers used by colab/train_maia2.ipynb.

Kept as a plain .py file so logic is reviewable in git rather than buried
in notebook JSON. The notebook imports from this module after cloning the
repo into Colab's /content/.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FineTuneConfig:
    """Knobs for the Maia2 personal fine-tune.

    Defaults are conservative — full fine-tune at low LR — since we have
    only ~15-25k games (small relative to Maia2's pretraining set).
    """

    user: str = "nick_p12"
    pgn_zst_path: str = "/content/repo/data/clean/all.pgn.zst"
    output_dir: str = "/content/repo/colab_outputs"
    pretrained_type: str = "blitz"  # or "rapid"
    epochs: int = 3
    learning_rate: float = 1e-5  # 100x lower than typical pretraining
    batch_size: int = 256
    val_fraction: float = 0.05
    seed: int = 42
    freeze_encoder: bool = False
    weight_decay: float = 1e-4
    warmup_steps: int = 200


def set_seed(seed: int = 42) -> None:
    import os

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_environment() -> dict:
    import platform

    import torch

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
    }
    logger.info("Environment: %s", info)
    return info


def write_yaml_config(cfg: FineTuneConfig, yaml_path: Path) -> Path:
    """Plan A: write a YAML config for maia2.train.run().

    The exact schema is read from the repo on first run. This writes a
    best-effort config; if maia2.train.run rejects keys, the notebook
    falls through to Plan B.
    """
    import yaml  # PyYAML, present in Colab by default

    payload = {
        "data": {
            "pgn_zst": cfg.pgn_zst_path,
            "val_fraction": cfg.val_fraction,
        },
        "model": {
            "from_pretrained": cfg.pretrained_type,
            "freeze_encoder": cfg.freeze_encoder,
        },
        "training": {
            "epochs": cfg.epochs,
            "lr": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "weight_decay": cfg.weight_decay,
            "warmup_steps": cfg.warmup_steps,
            "seed": cfg.seed,
        },
        "output": {
            "dir": cfg.output_dir,
        },
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    logger.info("Wrote training config to %s", yaml_path)
    return yaml_path


def write_run_metadata(cfg: FineTuneConfig, env: dict, out_dir: Path) -> Path:
    """Save a JSON record of the training run alongside the checkpoint.

    Captures config + env + timestamp so the resulting checkpoint is
    reproducible and self-describing.
    """
    meta = {
        "config": asdict(cfg),
        "environment": env,
        "started_unix": int(time.time()),
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"run_meta_{meta['started_unix']}.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    logger.info("Wrote run metadata to %s", path)
    return path


def custom_finetune_step(model, batch, optimizer, device):
    """Plan B: one fine-tune step, written by hand on top of `from_pretrained`.

    Used if maia2.train.run's YAML doesn't expose per-checkpoint resume.
    Implementation defers to the notebook because it depends on the
    actual model.forward() signature, which we'll inspect at runtime.
    """
    raise NotImplementedError(
        "Plan B step is wired in the notebook after we've inspected "
        "model.forward() and the dataset format at runtime. See the "
        "'Plan B' section of train_maia2.ipynb."
    )
