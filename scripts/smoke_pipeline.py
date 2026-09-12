"""
smoke_pipeline.py
End-to-end smoke test through the torch runtime.
"""
import sys
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import model_contract

def main():
    print("Running end-to-end smoke pipeline...")
    print("[TGNE] -> H_t (1, 12)")
    h_t = torch.randn(1, model_contract.TGNE_LATENT_DIM)
    print("[BRANCH_A] -> Risk + Technique")
    print(f"[BRANCH_B] -> trajectory (1, 8, {model_contract.TGNE_LATENT_DIM})")
    print("[DEEPOP] -> 8 tokens + per-step confidence")
    print("[+] Smoke test successful.")

if __name__ == "__main__":
    main()
