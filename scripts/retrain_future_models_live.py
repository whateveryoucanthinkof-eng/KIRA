"""Retrain Branch B and DeepOP on the canonical live TGNE latent space."""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
from branch_b_world_model.infiltration_head import InfiltrationRiskHead
from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
from branch_b_world_model.train_branch_b import HostRolloutDataset, create_rollout_samples
from data_unification.cic2018_adapter import CIC2018Adapter
from data_unification.ctu13_adapter import CTU13Adapter
from data_unification.multi_dataset_stream import HostTrajectoryExtractor
from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
from deepop_decoder.joint_vocab import get_joint_vocab
from deepop_decoder.train_cwa_decoder import CWASequenceDataset, create_cwa_training_samples


def load_records(cic_dir, ctu_dir, rows_per_file):
    records = []
    cic = CIC2018Adapter()
    ctu = CTU13Adapter()
    files = sorted(cic_dir.glob("*.csv")) + sorted(ctu_dir.glob("*/*.binetflow"))
    split = max(1, int(len(files) * 0.8))
    for path in files[:split]:
        records.extend(cic.parse_file(str(path), max_rows=rows_per_file) if path.suffix == ".csv" else ctu.parse_netflow_csv(str(path), max_rows=rows_per_file))
    train_files = records
    records = []
    for path in files[split:]:
        records.extend(cic.parse_file(str(path), max_rows=rows_per_file) if path.suffix == ".csv" else ctu.parse_netflow_csv(str(path), max_rows=rows_per_file))
    return train_files, records


def train_branch_b_live(train_traj, val_traj, output, epochs, device):
    train_samples = create_rollout_samples(train_traj, T=5, K=8)
    val_samples = create_rollout_samples(val_traj, T=5, K=8)
    train_loader = DataLoader(HostRolloutDataset(train_samples), batch_size=128, shuffle=True)
    val_loader = DataLoader(HostRolloutDataset(val_samples), batch_size=128)
    wdt = HostWorldDynamicsTransformer(d_latent=12, d_model=64, n_heads=4, n_layers=3).to(device)
    risk = InfiltrationRiskHead(d_latent=12, hidden_dim=32).to(device)
    optimizer = torch.optim.Adam(list(wdt.parameters()) + list(risk.parameters()), lr=1e-3, weight_decay=1e-4)
    best = float("inf")
    for epoch in range(epochs):
        wdt.train(); risk.train()
        for batch in train_loader:
            h = batch["h_history"].to(device); target = batch["h_future"].to(device); target_risk = batch["risk_future"].to(device)
            optimizer.zero_grad()
            pred = wdt.rollout(h, K=8)
            pred_risk, _ = risk.forward_trajectory(pred)
            loss = sum((0.9 ** k) * F.mse_loss(pred[:, k], target[:, k]) for k in range(8))
            loss = loss + F.binary_cross_entropy(pred_risk, target_risk)
            loss.backward(); optimizer.step()
        wdt.eval(); risk.eval(); losses=[]
        with torch.no_grad():
            for batch in val_loader:
                h=batch["h_history"].to(device); target=batch["h_future"].to(device); target_risk=batch["risk_future"].to(device)
                pred=wdt.rollout(h,K=8); pred_risk,_=risk.forward_trajectory(pred)
                losses.append((F.mse_loss(pred,target)+F.binary_cross_entropy(pred_risk,target_risk)).item())
        score=float(np.mean(losses)); print(f"Branch B epoch={epoch+1} val_loss={score:.4f}")
        if score < best:
            best=score; output.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"wdt_state_dict":wdt.state_dict(),"risk_head_state_dict":risk.state_dict(),"epoch":epoch+1,"history_steps":5,"forecast_steps":8,"window_seconds":2.0},output)


def train_deepop_live(train_traj, val_traj, output, epochs, device):
    vocab=get_joint_vocab(network_observable_only=True)
    train_samples=create_cwa_training_samples(train_traj,vocab,K=8)
    val_samples=create_cwa_training_samples(val_traj,vocab,K=8)
    train_loader=DataLoader(CWASequenceDataset(train_samples),batch_size=64,shuffle=True)
    val_loader=DataLoader(CWASequenceDataset(val_samples),batch_size=64)
    decoder=DeepOPForecastDecoder(d_latent=12,d_model=72,vocab_size=vocab.vocab_size,n_heads=6,num_layers=2,window_sizes=[2,4,8],dim_feedforward=144).to(device)
    optimizer=torch.optim.AdamW(decoder.parameters(),lr=5e-4,weight_decay=1e-4)
    best=float("inf")
    for epoch in range(epochs):
        decoder.train(); train_losses=[]
        for batch in train_loader:
            h=batch["h_future"].to(device); inp=batch["input_tokens"].to(device); tgt=batch["target_tokens"].to(device)
            optimizer.zero_grad(); logits=decoder(h,inp); loss=F.cross_entropy(logits.reshape(-1,vocab.vocab_size),tgt.reshape(-1),label_smoothing=0.04); loss.backward(); torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.0); optimizer.step(); train_losses.append(loss.item())
        decoder.eval(); val_losses=[]
        with torch.no_grad():
            for batch in val_loader:
                logits=decoder(batch["h_future"].to(device),batch["input_tokens"].to(device)); val_losses.append(F.cross_entropy(logits.reshape(-1,vocab.vocab_size),batch["target_tokens"].to(device).reshape(-1)).item())
        score=float(np.mean(val_losses)); print(f"DeepOP epoch={epoch+1} train_loss={np.mean(train_losses):.4f} val_loss={score:.4f}")
        if score < best:
            best=score; output.parent.mkdir(parents=True,exist_ok=True); torch.save({"decoder_state_dict":decoder.state_dict(),"epoch":epoch+1,"forecast_steps":8,"window_seconds":2.0,"vocab_size":vocab.vocab_size},output)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cic-dir",type=Path,required=True); parser.add_argument("--ctu-dir",type=Path,required=True)
    parser.add_argument("--tgne",type=Path,required=True); parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--rows-per-file",type=int,default=1000); parser.add_argument("--epochs",type=int,default=3)
    args=parser.parse_args(); random.seed(42); np.random.seed(42); torch.manual_seed(42)
    train_records,val_records=load_records(args.cic_dir,args.ctu_dir,args.rows_per_file)
    tgn=build_or_load_tgne_ta(checkpoint_path=str(args.tgne)); extractor=HostTrajectoryExtractor(tgne_ta_model=tgn,window_size_sec=2.0)
    train_traj=extractor.extract_trajectories(train_records); val_traj=extractor.extract_trajectories(val_records)
    device="cuda" if torch.cuda.is_available() else "cpu"
    train_branch_b_live(train_traj,val_traj,args.out_dir/"host_wdt.canonical-tgne.pt",args.epochs,device)
    train_deepop_live(train_traj,val_traj,args.out_dir/"cwa_forecast_decoder.canonical-tgne.pt",args.epochs,device)


if __name__ == "__main__":
    main()
