"""
Downstream Decoders Training and Evaluation Pipeline.

Trains and evaluates task-specific interpretation heads on top of the
frozen/fine-tuned World Dynamics Transformer:
1. StateDecoder: Reconstructs interpretable traffic & graph topology metrics.
2. AttackStageDecoder: Classifies future MITRE ATT&CK stages and techniques.
3. RiskHead: Predicts per-window infiltration hazard and cumulative campaign risk.
"""

from pathlib import Path
from typing import Dict, Tuple
import argparse
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from world_model.data.feature_schema import ATTACK_STAGE_NAMES, D_STATS, D_Z, N_ATTACK_STAGES
from world_model.data.latent_dataset import LatentSequenceDataset
from world_model.data.splits import chronological_split
from world_model.evaluation.calibration import compute_calibration_metrics
from world_model.evaluation.detection_metrics import compute_attack_stage_metrics, compute_detection_metrics
from world_model.models.attack_decoder import AttackStageDecoder
from world_model.models.readout import AttentiveGraphReadout
from world_model.models.risk_head import RiskHead
from world_model.models.state_decoder import StateDecoder
from world_model.models.world_dynamics_transformer import WorldDynamicsTransformer
from world_model.training.checkpointing import load_checkpoint, save_checkpoint
from world_model.training.losses import AttackStageFocalLoss, RiskInfiltrationLoss, StateReconstructionLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MultiHeadWorldModel(nn.Module):
    """Integrates WDT with StateDecoder, AttackStageDecoder, and RiskHead."""

    def __init__(self, wdt: WorldDynamicsTransformer):
        super().__init__()
        self.wdt = wdt
        self.d_z = wdt.d_z
        self.state_decoder = StateDecoder(d_z=self.d_z)
        self.attack_decoder = AttackStageDecoder(d_z=self.d_z, n_stages=N_ATTACK_STAGES)
        self.risk_head = RiskHead(d_z=self.d_z, default_horizon=wdt.default_horizon)

    def forward(self, z_seq: torch.Tensor, timestamps: torch.Tensor) -> Dict[str, torch.Tensor]:
        wdt_out = self.wdt(z_seq, timestamps)
        z_pred = wdt_out["z_pred"]

        state_recon = self.state_decoder(z_pred)
        attack_out = self.attack_decoder(z_pred)
        risk_out = self.risk_head(z_pred)

        return {
            "wdt_out": wdt_out,
            "z_pred": z_pred,
            "state_recon": state_recon,
            "attack_out": attack_out,
            "risk_out": risk_out,
        }


def train_downstream_heads(
    config_path: str = "world_model/config/default_config.yaml",
    wdt_checkpoint: str = "saved_models/world_model/wdt_best.pt",
    epochs: int = 5,
    save_dir: str = "saved_models/world_model",
):
    """Executes downstream heads training."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training downstream heads on {device}...")

    # Load data
    cache_path = Path(cfg["data"]["latent_cache_dir"]) / "ctu13_latent_dataset.pt"
    data_dict = torch.load(cache_path, map_location="cpu", weights_only=False)

    latent_states = data_dict["latent_states"]
    timestamps = data_dict["timestamps"]
    stats = data_dict["statistics"]
    labels = data_dict["labels"]
    mal_fractions = data_dict.get("malicious_fractions", torch.zeros_like(timestamps))

    train_mask, val_mask, test_mask = chronological_split(
        timestamps,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
    )

    H = cfg["data"]["history_len"]
    K = cfg["data"]["horizon_k"]

    train_ds = LatentSequenceDataset(
        latent_states[train_mask], timestamps[train_mask], stats[train_mask],
        labels[train_mask], mal_fractions[train_mask], history_len=H, horizon_k=K,
    )
    val_ds = LatentSequenceDataset(
        latent_states[val_mask], timestamps[val_mask], stats[val_mask],
        labels[val_mask], mal_fractions[val_mask], history_len=H, horizon_k=K,
    )
    test_ds = LatentSequenceDataset(
        latent_states[test_mask], timestamps[test_mask], stats[test_mask],
        labels[test_mask], mal_fractions[test_mask], history_len=H, horizon_k=K,
    )

    batch_size = cfg["data"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Load WDT backbone
    m_cfg = cfg["model"]
    wdt = WorldDynamicsTransformer(
        d_z=m_cfg["d_z"],
        d_model=m_cfg["d_model"],
        n_heads=m_cfg["n_heads"],
        n_layers=m_cfg["n_layers"],
        d_ff=m_cfg["d_ff"],
        d_time=m_cfg["d_time"],
        max_len=m_cfg["max_len"],
        dropout=0.1,
        default_horizon=m_cfg["default_horizon"],
    ).to(device)

    load_checkpoint(wdt_checkpoint, model=wdt, device=device)
    # Freeze WDT backbone for downstream heads training
    for p in wdt.parameters():
        p.requires_grad = False

    model = MultiHeadWorldModel(wdt).to(device)

    # Losses
    state_loss_fn = StateReconstructionLoss()
    attack_loss_fn = AttackStageFocalLoss()
    risk_loss_fn = RiskInfiltrationLoss()

    # Optimize only decoder heads
    head_params = [
        p for name, p in model.named_parameters() if not name.startswith("wdt.") and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)

    logger.info(f"Starting downstream heads training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            z_hist = batch["z_history"].to(device)
            ts_hist = batch["ts_history"].to(device)
            fut_stats = batch["future_stats"].to(device)
            fut_stages = batch["future_stages"].to(device)
            fut_risk_step = batch["future_risk_step"].to(device)
            fut_risk_cumul = batch["future_risk_cumulative"].to(device)

            optimizer.zero_grad()

            # Rollout K steps
            rollout = wdt.rollout(z_hist, ts_hist, K=K, use_kv_cache=True)
            z_fut = rollout["z_future"]

            state_preds = model.state_decoder(z_fut)
            attack_preds = model.attack_decoder(z_fut)
            risk_preds = model.risk_head(z_fut)

            l_state = state_loss_fn(state_preds, fut_stats)
            l_attack = attack_loss_fn(attack_preds["stage_logits"], fut_stages[:, 0])
            l_risk = risk_loss_fn(
                risk_preds["per_step_risk"],
                fut_risk_step,
                risk_preds.get("cumulative_risk"),
                fut_risk_cumul,
            )

            loss = 0.2 * l_state + 0.5 * l_attack + 0.5 * l_risk
            loss.backward()
            nn.utils.clip_grad_norm_(head_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f}")

    # Evaluate on Holdout Test Set
    logger.info("Evaluating downstream heads on holdout test set...")
    model.eval()
    all_stage_trues, all_stage_preds_logits = [], []
    all_risk_trues, all_risk_preds = [], []

    with torch.no_grad():
        for batch in test_loader:
            z_hist = batch["z_history"].to(device)
            ts_hist = batch["ts_history"].to(device)
            fut_stages = batch["future_stages"].to(device)
            fut_risk_cumul = batch["future_risk_cumulative"].to(device)

            rollout = wdt.rollout(z_hist, ts_hist, K=K, use_kv_cache=True)
            z_fut = rollout["z_future"]

            attack_preds = model.attack_decoder(z_fut)
            risk_preds = model.risk_head(z_fut)

            all_stage_trues.extend(fut_stages[:, 0].cpu().numpy())
            all_stage_preds_logits.append(attack_preds["stage_logits"].cpu().numpy())

            all_risk_trues.extend(fut_risk_cumul.cpu().numpy())
            all_risk_preds.extend(risk_preds["cumulative_risk"].cpu().numpy())

    stage_logits = np.concatenate(all_stage_preds_logits, axis=0)
    stage_metrics = compute_attack_stage_metrics(np.array(all_stage_trues), stage_logits)
    risk_metrics = compute_detection_metrics(np.array(all_risk_trues), np.array(all_risk_preds))
    calib_metrics = compute_calibration_metrics(np.array(all_risk_preds), np.array(all_risk_trues))

    logger.info("==========================================================")
    logger.info("DOWNSTREAM DECODER TEST RESULTS:")
    logger.info(f"ATT&CK Stage Accuracy: {stage_metrics['stage_accuracy']*100:.2f}%")
    logger.info(f"ATT&CK Stage Macro F1: {stage_metrics['stage_macro_f1']:.4f}")
    logger.info(f"Per-Class F1:          {stage_metrics['per_class_f1']}")
    logger.info(f"Infiltration AUROC:    {risk_metrics['auroc']:.4f}")
    logger.info(f"Infiltration AUPRC:    {risk_metrics['auprc']:.4f}")
    logger.info(f"Infiltration F1:       {risk_metrics['f1']:.4f}")
    logger.info(f"Expected Calib Error:  {calib_metrics['ece']:.4f}")
    logger.info(f"Brier Score:           {calib_metrics['brier_score']:.4f}")
    logger.info("==========================================================")

    # Save checkpoint
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chk_file = out_dir / "downstream_heads.pt"
    torch.save(
        {
            "state_decoder": model.state_decoder.state_dict(),
            "attack_decoder": model.attack_decoder.state_dict(),
            "risk_head": model.risk_head.state_dict(),
            "stage_metrics": stage_metrics,
            "risk_metrics": risk_metrics,
            "calibration": calib_metrics,
        },
        chk_file,
    )
    logger.info(f"Saved downstream heads checkpoint to {chk_file}")

    return {
        "stage_metrics": stage_metrics,
        "risk_metrics": risk_metrics,
        "calib_metrics": calib_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train_downstream_heads(epochs=args.epochs)
