"""
Stage 0: Latent Dataset Generator for CTU-13.

Transforms the CTU-13 2-second time-windowed dataset into cached latent states z_t
conforming to the unified schema:
    z_t = LayerNorm(Concat(z_structural, z_statistical)) in R^{128}
    z_structural in R^{114}
    z_statistical in R^{14} (Φ_t)

Saves the processed latent sequence to data/latent_cache/ctu13_latent_dataset.pt
for high-speed training of the World Dynamics Transformer.
"""

from pathlib import Path
import argparse
import logging
import torch
import torch.nn as nn

from world_model.data.ctu13_adapter import normalize_features, prepare_ctu13_dataset
from world_model.data.feature_schema import D_STATS, D_STRUCTURAL, D_WINDOW, D_Z
from world_model.models.readout import AttentiveGraphReadout

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class WindowTelemetryEncoder(nn.Module):
    """
    Encoder for window-level flow + PCAP telemetry features (36-dim)
    into the 114-dim structural representation component of z_t.
    """

    def __init__(self, in_dim: int = D_WINDOW, out_dim: int = D_STRUCTURAL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )
        self.final_ln = nn.LayerNorm(D_Z)

    def forward(self, window_feats: torch.Tensor, stats_feats: torch.Tensor) -> torch.Tensor:
        z_struct = self.net(window_feats)
        z_raw = torch.cat([z_struct, stats_feats], dim=-1)
        z_t = self.final_ln(z_raw)
        return z_t


def generate_and_cache_ctu13_latents(
    parquet_path: str = r"c:\Users\vyomk\OneDrive\Desktop\gnn\data\ctu13\test_ctu13_states_2s_pcap.parquet",
    output_dir: str = r"c:\Users\vyomk\OneDrive\Desktop\gnn\data\latent_cache",
    device: str = "cpu",
) -> Path:
    """
    Processes CTU-13 dataset and caches the latent sequences.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ctu13_latent_dataset.pt"

    logger.info(f"Loading and preparing CTU-13 dataset from {parquet_path}...")
    data_dict = prepare_ctu13_dataset(parquet_path)

    raw_features = data_dict["features"]      # [N, 36]
    stats_features = data_dict["statistics"]  # [N, 14]
    timestamps = data_dict["timestamps"]      # [N]
    labels = data_dict["labels"]              # [N]
    is_attack = data_dict["is_attack"]        # [N]
    scenario_ids = data_dict["scenario_ids"]

    logger.info("Normalizing features...")
    norm_features, mean_feat, std_feat = normalize_features(raw_features)
    norm_stats, mean_stats, std_stats = normalize_features(stats_features)

    # Encode into latent states z_t
    logger.info(f"Encoding {len(raw_features):,} windows into {D_Z}-dim latent states z_t...")
    torch.manual_seed(42)
    encoder = WindowTelemetryEncoder(in_dim=D_WINDOW, out_dim=D_STRUCTURAL)
    encoder.eval()

    with torch.no_grad():
        # Process in batches to keep memory bounded
        batch_size = 4096
        z_list = []
        for i in range(0, len(norm_features), batch_size):
            b_feat = norm_features[i : i + batch_size]
            b_stat = norm_stats[i : i + batch_size]
            b_z = encoder(b_feat, b_stat)
            z_list.append(b_z)
        z_tensor = torch.cat(z_list, dim=0)

    # Extract malicious edge fractions from stats column 11
    mal_fractions = stats_features[:, 11].clone()

    cache_payload = {
        "latent_states": z_tensor,             # [N, 128]
        "timestamps": timestamps,              # [N]
        "statistics": stats_features,          # [N, 14]
        "labels": labels,                      # [N] 5-class attack stage
        "is_attack": is_attack,                # [N] binary
        "malicious_fractions": mal_fractions,  # [N]
        "scenario_ids": scenario_ids,
        "feature_means": {"features": mean_feat, "stats": mean_stats},
        "feature_stds": {"features": std_feat, "stats": std_stats},
    }

    logger.info(f"Saving latent dataset to {out_file} ({z_tensor.shape[0]:,} states, {D_Z} dims)...")
    torch.save(cache_payload, out_file)
    logger.info("Latent dataset generation and caching complete!")

    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and cache CTU-13 latent dataset")
    parser.add_argument("--parquet_path", type=str, default=r"c:\Users\vyomk\OneDrive\Desktop\gnn\data\ctu13\test_ctu13_states_2s_pcap.parquet")
    parser.add_argument("--output_dir", type=str, default=r"c:\Users\vyomk\OneDrive\Desktop\gnn\data\latent_cache")
    args = parser.parse_args()

    generate_and_cache_ctu13_latents(args.parquet_path, args.output_dir)
