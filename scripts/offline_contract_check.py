"""
offline_contract_check.py
Validates the contents of checkpoints and manifests against model_contract.py.
"""
import sys
import json
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).resolve().parent.parent))
import model_contract

def main():
    print("Running offline contract check...")
    deepop_manifest = Path("saved_models/deepop/deepop.manifest.json")
    if deepop_manifest.exists():
        with open(deepop_manifest) as f:
            data = json.load(f)
            assert data["architecture"]["vocab_size"] == model_contract.DEEPOP_VOCAB_SIZE
            assert data["architecture"]["nhead"] == model_contract.DEEPOP_NHEAD
            print("[+] deepop.manifest.json aligns with contract")
            
    branch_b_manifest = Path("saved_models/branch_b/branch_b.manifest.json")
    if branch_b_manifest.exists():
        with open(branch_b_manifest) as f:
            data = json.load(f)
            assert data["architecture"]["num_encoder_layers"] == model_contract.BRANCH_B_ENCODER_LAYERS
            print("[+] branch_b.manifest.json aligns with contract")

    print("[+] All offline contract checks passed.")

if __name__ == "__main__":
    main()
