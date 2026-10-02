"""
Training pipeline for Branch A (MultiTaskLSTM Attack-Sequence Model).
Trains on unified host trajectories extracted from CIC-IDS2017, CTU-13, and Warden.
"""

import os
import sys
import argparse
from typing import Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add paths
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("bita"))

from data_unification.cic2017_adapter import CIC2017Adapter
from data_unification.ctu13_adapter import CTU13Adapter
from data_unification.warden_adapter import WardenAdapter
from data_unification.flow_to_temporal_event import FlowToTemporalEventAdapter
from data_unification.multi_dataset_stream import HostTrajectoryExtractor
from model.extentedtgn import ExtendedTGN
from utils.utils import NeighborFinder
from branch_a_gnn_lstm.sequence_dataset import (
    HostSequenceDataset,
    create_host_sequence_samples,
    TECHNIQUE_VOCAB,
)
from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
from cyberworld_v4.config import get_contract
from data_unification.tgne_features import SCHEMA_VERSION


def load_sample_multi_dataset_records(max_per_source: int = 500):
    """Loads a balanced sample of flow records across all available datasets."""
    records = []

    # 1. CIC-IDS2017 PortScan and DDoS
    cic_adapter = CIC2017Adapter()
    for f in [
        "cic2017csv/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "cic2017csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ]:
        if os.path.exists(f):
            records.extend(list(cic_adapter.parse_file(f, max_rows=max_per_source)))

    # 2. CTU-13 Parquet
    ctu_adapter = CTU13Adapter()
    ctu_path = "data/ctu13/test_ctu13_states_2s_pcap.parquet"
    if os.path.exists(ctu_path):
        records.extend(list(ctu_adapter.parse_parquet(ctu_path, max_rows=max_per_source)))

    # 3. Warden
    warden_adapter = WardenAdapter()
    warden_path = "bita/Dataset/11March_e.csv"
    if os.path.exists(warden_path):
        records.extend(list(warden_adapter.parse_file(warden_path, max_rows=max_per_source)))

    return records


class StaleEncoderArchitecture(RuntimeError):
    """A TGNE checkpoint whose architecture predates the current model.

    Distinct from a contract mismatch (right shapes, wrong temporal
    granularity): here the weights cannot be loaded at all.
    """


# The encoder retrained 2026-09-21 on CIC-2017 + CIC-2018 + CTU-13 at full
# density: inductive test AUC 0.9626, transductive 0.9972, macro F1 0.8405.
# Paths are resolved relative to the repo root so a different cwd still finds
# them.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_TGNE_CHECKPOINT = os.path.join(
    _REPO_ROOT, "saved_models", "bita_bigru_transformer-unified_final.pth")
LEGACY_TGNE_CHECKPOINT = os.path.join(
    _REPO_ROOT, "bita", "saved_models", "bita_bigru_transformer-warden_alerts.pth")


def build_or_load_tgne_ta(
    config_path: str = "bita/saved_models/bita_config.json",
    checkpoint_path: Optional[str] = None,
):
    """
    Builds and loads pre-trained TGNE-TA model adhering strictly to serialized configuration.
    Fails loudly if checkpoint or configuration differs.
    """
    checkpoint_path = checkpoint_path or os.environ.get("TGNE_CHECKPOINT_PATH")
    if not checkpoint_path:
        # Resolution order: explicit argument, then TGNE_CHECKPOINT_PATH, then
        # the canonical retrained encoder, then the legacy Warden checkpoint.
        #
        # The canonical entry is not cosmetic. Six call sites -- including
        # control_backend/model_adapter.py, the live serving path -- call this
        # with no argument, and every one of them used to land on the Warden
        # checkpoint. That checkpoint predates the edge-aware category head:
        # its head was a single Linear over summed node embeddings, so it could
        # not see the flow it was classifying and collapsed to one class. Branch
        # A is trained against the retrained encoder, so leaving the default on
        # Warden means training and serving disagree about the encoder.
        #
        # The legacy path is kept last so an install without the retrained file
        # still reports the familiar StaleEncoderArchitecture rather than a
        # bare FileNotFoundError.
        for _cand in (CANONICAL_TGNE_CHECKPOINT, LEGACY_TGNE_CHECKPOINT):
            if os.path.exists(_cand):
                checkpoint_path = _cand
                break
    ckpt_path = checkpoint_path or LEGACY_TGNE_CHECKPOINT
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"TGNE-TA checkpoint not found at: {ckpt_path}")

    config = {
        "n_layers": 1,
        "n_heads": 2,
        "dropout": 0.0,
        "use_memory": False,
        "message_dimension": 12,
        "memory_dimension": 12,
        "embedding_module_type": "graph_attention",
        "message_function": "identity",
        "aggregator_type": "bigru_transformer",
        "memory_updater_type": "gru",
        "num_categories": 4,
        "edge_feat_dim": 12,
        "node_feat_dim": 12,
        # bita/train.py defaults, which every encoder predating these config
        # keys was trained with.
        "n_neighbors": 10,
        "neighbor_sampling": "most_recent",
    }
    if checkpoint_path:
        config_path = os.path.splitext(checkpoint_path)[0] + "_config.json"
    if os.path.exists(config_path):
        import json
        with open(config_path, "r") as f:
            loaded_cfg = json.load(f)
            config.update({k: v for k, v in loaded_cfg.items() if k in config})
            _want = SCHEMA_VERSION
            _got = loaded_cfg.get("feature_schema_version")
            if _got != _want:
                # Not cosmetic. A schema bump means the VALUES changed, so the
                # encoder would receive an input distribution it never saw --
                # with matching shapes and no error anywhere. Refusing here is
                # the only place that can catch it.
                raise ValueError(
                    f"TGNE checkpoint {ckpt_path} was trained under feature "
                    f"schema {_got!r}; this tree is {_want!r}.\n\n"
                    f"What changed in 2.0.0: dst_port is log-scaled instead of "
                    f"divided by 65535, and the unique_peers / unique_dst_ports "
                    f"host attributes no longer saturate at 147. Every stored "
                    f"feature value is different, so the encoder must be "
                    f"retrained -- there is no conversion.\n\n"
                    f"Retrain:  ./train.sh (see README.md, 'Train the models')"
                )
            if loaded_cfg.get("edge_feat_dim") != 12 or loaded_cfg.get("node_feat_dim") != 12:
                raise ValueError("TGNE checkpoint dimensions do not match the canonical 12-D contract")
            # Edge-feature ablation (CYBERWORLD_ABLATE_EDGE_FEATURES) zeroes
            # features at extraction time, for training AND serving. An encoder
            # trained with dst_port zeroed and served with it present (or the
            # reverse) sees an input distribution it never saw, with matching
            # shapes and no error. Configs that predate the field were trained
            # with nothing ablated.
            from data_unification.tgne_features import ablated_edge_features
            _trained = sorted(loaded_cfg.get("ablated_edge_features") or [])
            _serving = sorted(ablated_edge_features())
            if _trained != _serving:
                raise ValueError(
                    f"TGNE checkpoint {ckpt_path} was trained with edge features "
                    f"{_trained or 'none'} ablated, but this process ablates "
                    f"{_serving or 'none'} (CYBERWORLD_ABLATE_EDGE_FEATURES). "
                    f"Set the variable to match the checkpoint.")
            # And the IP node features (CYBERWORLD_ABLATE_NODE_FEATURES).
            from data_unification.ip_features import ablated_node_features
            _trained_n = sorted(loaded_cfg.get("ablated_node_features") or [])
            _serving_n = sorted(ablated_node_features())
            if _trained_n != _serving_n:
                raise ValueError(
                    f"TGNE checkpoint {ckpt_path} was trained with node features "
                    f"{_trained_n or 'none'} ablated, but this process ablates "
                    f"{_serving_n or 'none'} (CYBERWORLD_ABLATE_NODE_FEATURES). "
                    f"Set the variable to match the checkpoint.")

    n_nodes = 5000
    adj_list = [[] for _ in range(n_nodes)]
    ngh_finder = NeighborFinder(adj_list, uniform=True)

    # Placeholder only: HostTrajectoryExtractor.extract_trajectories replaces
    # node_raw_features with real intrinsic IP features for the hosts actually
    # present (data_unification/ip_features.py). Left zero here because no IP
    # map exists yet at construction time -- but if a caller ever runs the
    # encoder WITHOUT going through the extractor, it would be feeding zeros to
    # a model trained on IP features, so this stays a documented placeholder
    # rather than a silent default.
    node_feats = np.zeros((n_nodes, config["node_feat_dim"]), dtype=np.float32)
    edge_feats = np.zeros((10000, config["edge_feat_dim"]), dtype=np.float32)

    tgn = ExtendedTGN(
        neighbor_finder=ngh_finder,
        node_features=node_feats,
        edge_features=edge_feats,
        device="cpu",
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        dropout=config["dropout"],
        use_memory=config["use_memory"],
        message_dimension=config["message_dimension"],
        memory_dimension=config["memory_dimension"],
        embedding_module_type=config["embedding_module_type"],
        message_function=config["message_function"],
        aggregator_type=config["aggregator_type"],
        memory_updater_type=config["memory_updater_type"],
        num_categories=config["num_categories"],
    )

    state_dict = torch.load(ckpt_path, map_location="cpu")
    # A memory-enabled encoder saves its per-node memory rows, sized to the
    # training graph's node count. They are runtime state, not weights:
    # extraction and serving reset memory per capture/session and grow the
    # table as hosts appear. Keep this model's own (empty) rows so the strict
    # load checks every real weight.
    _own = tgn.state_dict()
    for _k in list(state_dict):
        if _k.endswith("memory.memory") or _k.endswith("memory.last_update"):
            state_dict[_k] = _own[_k]
    try:
        tgn.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        # The category head changed shape: it used to be a single Linear over
        # `src_emb + dst_emb` (12-D, direction-blind, edge-blind) and is now an
        # MLP over [src ; dst ; edge_features]. The old head could not see the
        # flow at all, so it collapsed to predicting one class for everything.
        #
        # Say that plainly instead of surfacing a raw state_dict diff.
        msg = str(exc)

        # Distinguish two very different failures that both mention
        # category_predictor:
        #
        #   a) SHAPE of the final layer differs -> num_categories mismatch,
        #      which means the config JSON is missing or wrong. The fix is to
        #      write one (scripts/write_encoder_config.py), NOT to retrain.
        #   b) the layer NAMES differ (category_predictor.weight vs
        #      category_predictor.0.weight) -> the checkpoint predates the
        #      edge-aware head and genuinely needs a retrain.
        #
        # The original message said "predates the edge-aware category head"
        # for both, which sends someone to retrain an encoder that only needed
        # a 400-byte JSON file beside it.
        if "size mismatch for category_predictor" in msg:
            import re as _re
            want = _re.search(r"shape torch\.Size\(\[(\d+)\]\) from checkpoint", msg)
            got = _re.search(r"current model is torch\.Size\(\[(\d+)\]\)", msg)
            raise StaleEncoderArchitecture(
                f"TGNE checkpoint {ckpt_path} was trained with "
                f"{want.group(1) if want else '?'} categories but this loader built "
                f"{got.group(1) if got else '?'}.\n\n"
                f"The architecture is fine -- the config JSON is missing or stale. "
                f"build_or_load_tgne_ta reads '<checkpoint>_config.json' and falls "
                f"back to num_categories=4 when it is absent.\n\n"
                f"Fix:  python scripts/write_encoder_config.py {ckpt_path}\n\n"
                f"Underlying: {exc}"
            ) from exc

        if "category_predictor" in msg:
            raise StaleEncoderArchitecture(
                f"TGNE checkpoint {ckpt_path} predates the edge-aware category "
                f"head. The old head was a single Linear on summed node "
                f"embeddings, which could not distinguish two flows between the "
                f"same pair of hosts and collapsed to a constant prediction. "
                f"Retrain the encoder: ./train.sh, then ./train.sh promote (README.md, "
                f"'Train the models').\n\nUnderlying: {exc}"
            ) from exc
        raise
    tgn.eval()
    _attach_neighbor_sampling(tgn, config)
    return tgn


def _attach_neighbor_sampling(tgn, config: Dict[str, Any]) -> None:
    """Carry the training-time neighbour sampling onto the model for serving.

    HostTrajectoryExtractor used a literal n_neighbors=10 with most-recent
    sampling, which matched bita/train.py's defaults by coincidence rather than
    by construction. It now reads these attributes, so an encoder trained with
    another --n_degree or --uniform is served the way it was trained.
    """
    sampling = config.get("neighbor_sampling", "most_recent")
    if sampling not in ("most_recent", "uniform"):
        raise ValueError(f"unknown neighbor_sampling {sampling!r} in the encoder config")
    n = int(config.get("n_neighbors", 10))
    if n < 1:
        raise ValueError(f"n_neighbors must be >= 1, got {n}")
    tgn.serving_n_neighbors = n
    tgn.serving_neighbor_uniform = sampling == "uniform"



from data_unification.split_manager import get_split_manager


def train_branch_a(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str = "saved_models/branch_a/branch_a_lstm.pt",
) -> Dict[str, Any]:
    print("Loading scientific disjoint train & val multi-dataset flow records...")
    sm = get_split_manager()
    train_records = sm.get_train_records(max_per_source=1000)
    val_records = sm.get_val_records(max_per_source=500)
    print(f"Total loaded train records: {len(train_records)}, val records: {len(val_records)}")

    # Extract dynamic graph & host trajectories for train partition
    tgn = build_or_load_tgne_ta()
    # Contract-bound. This was hardcoded to 60.0 while the shipped checkpoints
    # and live inference ran at 2.0s, so this trainer could not reproduce them.
    # The window now comes from the single source of truth.
    extractor = HostTrajectoryExtractor(
        tgne_ta_model=tgn, window_size_sec=get_contract().window_seconds
    )
    print("Extracting per-host trajectories with TGNE-TA embeddings for train partition...")
    train_trajectories = extractor.extract_trajectories(train_records)
    print(f"Active train hosts tracked: {len(train_trajectories)}")

    train_samples = create_host_sequence_samples(train_trajectories, seq_len=get_contract().history_steps, min_trajectory_len=1)
    print(f"Total train sequence samples: {len(train_samples)}")
    if len(train_samples) < 10:
        print("Warning: Few train samples created. Duplicating for robust mini-batch training.")
        train_samples = train_samples * 5

    # Extract dynamic graph & host trajectories for disjoint validation partition
    print("Extracting per-host trajectories with TGNE-TA embeddings for val partition...")
    val_trajectories = extractor.extract_trajectories(val_records)
    print(f"Active val hosts tracked: {len(val_trajectories)}")

    val_samples = create_host_sequence_samples(val_trajectories, seq_len=get_contract().history_steps, min_trajectory_len=1)
    print(f"Total val sequence samples: {len(val_samples)}")
    if len(val_samples) < 5:
        val_samples = val_samples * 5

    train_set = HostSequenceDataset(train_samples, seq_len=get_contract().history_steps)
    val_set = HostSequenceDataset(val_samples, seq_len=get_contract().history_steps)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskLSTM(
        input_dim=27,
        hidden_dim=64,
        num_layers=2,
        num_techniques=len(TECHNIQUE_VOCAB),
        num_gradations=4,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best_val_loss = float("inf")
    metrics_history = []

    print(f"Starting Branch A training on {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["features"].to(device)
            b_targets = {
                "risk": batch["risk"].to(device),
                "technique": batch["technique"].to(device),
                "gradation": batch["gradation"].to(device),
            }

            optimizer.zero_grad()
            preds = model(x)
            loss, m = model.compute_loss(preds, b_targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluation phase
        model.eval()
        val_losses = []
        correct_tech = 0
        correct_grad = 0
        total_eval = 0
        risk_errors = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                b_targets = {
                    "risk": batch["risk"].to(device),
                    "technique": batch["technique"].to(device),
                    "gradation": batch["gradation"].to(device),
                }
                preds = model(x)
                loss, _ = model.compute_loss(preds, b_targets)
                val_losses.append(loss.item())

                pred_tech = preds["technique_logits"].argmax(dim=-1)
                pred_grad = preds["gradation_logits"].argmax(dim=-1)
                correct_tech += (pred_tech == b_targets["technique"]).sum().item()
                correct_grad += (pred_grad == b_targets["gradation"]).sum().item()
                total_eval += len(b_targets["technique"])
                risk_errors.extend(
                    (preds["risk_score"] - b_targets["risk"]).abs().cpu().numpy()
                )

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        tech_acc = correct_tech / max(1, total_eval)
        grad_acc = correct_grad / max(1, total_eval)
        risk_mae = float(np.mean(risk_errors)) if risk_errors else 0.0

        epoch_stats = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "tech_acc": tech_acc,
            "grad_acc": grad_acc,
            "risk_mae": risk_mae,
        }
        metrics_history.append(epoch_stats)

        print(
            f"Epoch {epoch:02d} | Train: {epoch_stats['train_loss']:.4f} | "
            f"Val: {val_loss:.4f} | Tech Acc: {tech_acc*100:.1f}% | "
            f"Grad Acc: {grad_acc*100:.1f}% | Risk MAE: {risk_mae:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": epoch_stats,
                },
                save_path,
            )

    print(f"Model saved to {save_path}")
    return {
        "best_val_loss": best_val_loss,
        "history": metrics_history,
        "save_path": save_path,
    }


if __name__ == "__main__":
    train_branch_a(epochs=3)
