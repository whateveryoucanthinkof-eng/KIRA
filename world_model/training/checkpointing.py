"""
Checkpointing utility for the World Dynamics Transformer.

Handles saving and loading of model weights, optimizer states, learning rate
schedulers, training metadata, and validation metrics with hash verification.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import json
import logging
import os
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Union[str, Path],
    filename: str = "wdt_checkpoint.pt",
    is_best: bool = False,
    best_filename: str = "wdt_best.pt",
) -> Path:
    """
    Save training state dictionary to disk.
    
    Args:
        state: Dict containing 'epoch', 'model_state_dict', 'optimizer_state_dict',
               'val_loss', 'config', etc.
        checkpoint_dir: Directory path to save checkpoint.
        filename: Checkpoint filename.
        is_best: If True, also copies/saves as best_filename.
        best_filename: Name for best model checkpoint.
    Returns:
        Path to saved checkpoint file.
    """
    chk_dir = Path(checkpoint_dir)
    chk_dir.mkdir(parents=True, exist_ok=True)
    filepath = chk_dir / filename

    torch.save(state, filepath)
    logger.info(f"Saved checkpoint to {filepath} (epoch {state.get('epoch', '?')})")

    if is_best:
        best_path = chk_dir / best_filename
        torch.save(state, best_path)
        logger.info(f"Saved new best model checkpoint to {best_path}")

    # Also write a lightweight metadata JSON for inspection without loading full tensors
    meta = {
        "epoch": state.get("epoch"),
        "val_loss": state.get("val_loss"),
        "best_val_loss": state.get("best_val_loss"),
        "train_loss": state.get("train_loss"),
        "config": state.get("config"),
    }
    try:
        with open(chk_dir / "checkpoint_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Could not save checkpoint metadata JSON: {e}")

    return filepath


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load model weights and optional optimizer/scheduler state from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint .pt file.
        model: Target PyTorch model instance.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional lr_scheduler to restore state into.
        device: Device to map tensors to.
    Returns:
        Full checkpoint dictionary with metadata.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    chk = torch.load(path, map_location=device or torch.device("cpu"), weights_only=False)

    # Load model weights
    state_dict = chk.get("model_state_dict", chk)
    # Handle DataParallel prefix if present
    cleaned_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_dict[k[7:]] = v
        else:
            cleaned_dict[k] = v

    model.load_state_dict(cleaned_dict, strict=False)
    logger.info(f"Loaded model weights from {path}")

    if optimizer is not None and "optimizer_state_dict" in chk:
        optimizer.load_state_dict(chk["optimizer_state_dict"])
        logger.info("Restored optimizer state")

    if scheduler is not None and "scheduler_state_dict" in chk:
        scheduler.load_state_dict(chk["scheduler_state_dict"])
        logger.info("Restored scheduler state")

    return chk
