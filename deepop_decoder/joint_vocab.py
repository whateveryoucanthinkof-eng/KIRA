"""
Joint Token Vocabulary for DeepOP ATT&CK CWA Decoder.

Implements Strategy B: Consolidates fine-grained labels into the 6 primary network-observable
MITRE ATT&CK macro-techniques plus Benign baseline and special tokens (<PAD>=0, <BOS>=1, <EOS>=2).
"""

from typing import List, Dict, Tuple, Optional
import os
import pandas as pd


PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

# Canonical 6 Network-Observable MITRE ATT&CK Macro-Techniques + Benign
NETWORK_MACRO_TECHNIQUES = [
    ("Benign", "None"),
    ("C2", "T1071"),
    ("CredentialAccess", "T1110"),
    ("Exfiltration", "T1005"),
    ("Impact", "T1498"),
    ("InitialAccess", "T1190"),
    ("Recon", "T1595"),
]


def consolidate_network_technique(coarse: str, tech: Optional[str] = None) -> Tuple[str, str]:
    """
    Maps fine-grained raw label / technique pairs to the canonical 6 network-observable
    MITRE ATT&CK macro-techniques:
    1. Recon.T1595 (Active Scanning & Service Discovery, includes T1046)
    2. InitialAccess.T1190 (Exploit Public-Facing Application / Infiltration, includes T1189)
    3. CredentialAccess.T1110 (Brute Force / Password Guessing)
    4. C2.T1071 (Command and Control / Botnet Beaconing, includes T1071.001, T1568, T1204)
    5. Exfiltration.T1005 (Data Exfiltration / Disclosure, includes T1020)
    6. Impact.T1498 (Network Denial of Service / Flooding, includes T1498.001)
    Plus Benign.None.
    """
    coarse_str = str(coarse).strip() if coarse else "Benign"
    tech_str = str(tech).strip() if tech and str(tech).strip() != "None" else ""

    if coarse_str.lower() in ["benign", "normal", "background", "unknown"] and not tech_str:
        return "Benign", "None"

    # Reconnaissance: PortScan, T1046, T1595, Scanning
    if "T1595" in tech_str or "T1046" in tech_str or coarse_str == "Recon":
        return "Recon", "T1595"

    # Impact / DoS / DDoS: T1498, T1498.001
    if "T1498" in tech_str or coarse_str == "Impact" or "dos" in coarse_str.lower():
        return "Impact", "T1498"

    # Credential Access / Brute Force: T1110
    if "T1110" in tech_str or coarse_str == "CredentialAccess" or "brute" in coarse_str.lower() or "patator" in coarse_str.lower():
        return "CredentialAccess", "T1110"

    # Initial Access / Exploits: T1190, T1189
    if "T1190" in tech_str or "T1189" in tech_str or coarse_str == "InitialAccess":
        return "InitialAccess", "T1190"

    # Command & Control / Botnet: T1071, T1568, T1204
    if "T1071" in tech_str or "T1568" in tech_str or "T1204" in tech_str or coarse_str in ["C2", "Execution"]:
        return "C2", "T1071"

    # Exfiltration: T1005, T1020
    if "T1005" in tech_str or "T1020" in tech_str or coarse_str == "Exfiltration":
        return "Exfiltration", "T1005"

    # Fallback
    return "Benign", "None"


class JointAttackVocab:
    """Manages joint (coarse_category, technique_id) tokens and mappings."""

    def __init__(self, label_maps_dir: Optional[str] = None, network_observable_only: bool = True):
        self.label_maps_dir = label_maps_dir or os.path.join(
            os.path.dirname(__file__), "..", "data_unification", "label_maps"
        )
        self.network_observable_only = network_observable_only
        self.tokens: List[str] = list(SPECIAL_TOKENS)
        self.token_to_idx: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.idx_to_token: Dict[int, str] = {i: t for i, t in enumerate(self.tokens)}
        self._build_vocab()

    def _build_vocab(self):
        if self.network_observable_only:
            # Add canonical 6 network-observable techniques + Benign
            for coarse, tech in NETWORK_MACRO_TECHNIQUES:
                joint_str = f"{coarse}.{tech}"
                if joint_str not in self.token_to_idx:
                    idx = len(self.tokens)
                    self.tokens.append(joint_str)
                    self.token_to_idx[joint_str] = idx
                    self.idx_to_token[idx] = joint_str
        else:
            # Load all three label maps for full unconstrained vocabulary
            map_files = ["cic_label_map.csv", "ctu_label_map.csv", "warden_label_map.csv"]
            unique_pairs = set([("Benign", "None")])

            for mf in map_files:
                p = os.path.join(self.label_maps_dir, mf)
                if os.path.exists(p):
                    df = pd.read_csv(p)
                    for _, row in df.iterrows():
                        coarse = str(row["coarse_category"]).strip()
                        techs = str(row.get("attck_technique_ids", "")).strip()
                        if pd.isna(techs) or not techs:
                            unique_pairs.add((coarse, "None"))
                        else:
                            for t in techs.split(";"):
                                if t.strip():
                                    unique_pairs.add((coarse, t.strip()))

            sorted_pairs = sorted(list(unique_pairs))
            for coarse, tech in sorted_pairs:
                joint_str = f"{coarse}.{tech}"
                if joint_str not in self.token_to_idx:
                    idx = len(self.tokens)
                    self.tokens.append(joint_str)
                    self.token_to_idx[joint_str] = idx
                    self.idx_to_token[idx] = joint_str

    @property
    def pad_idx(self) -> int:
        return self.token_to_idx[PAD_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.token_to_idx[BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token_to_idx[EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, coarse: str, tech: Optional[str] = None) -> int:
        if self.network_observable_only:
            coarse, tech = consolidate_network_technique(coarse, tech)
        
        tech_str = tech if tech and tech != "" else "None"
        key = f"{coarse}.{tech_str}"
        if key in self.token_to_idx:
            return self.token_to_idx[key]
        
        # Consolidation fallback
        c_alt, t_alt = consolidate_network_technique(coarse, tech)
        alt_key = f"{c_alt}.{t_alt}"
        if alt_key in self.token_to_idx:
            return self.token_to_idx[alt_key]

        fallback = f"{coarse}.None"
        if fallback in self.token_to_idx:
            return self.token_to_idx[fallback]
        return self.token_to_idx["Benign.None"]

    def decode(self, idx: int) -> Tuple[str, str]:
        token = self.idx_to_token.get(idx, "Benign.None")
        if token in SPECIAL_TOKENS:
            return token, token
        parts = token.split(".", 1)
        coarse = parts[0]
        tech = parts[1] if len(parts) > 1 and parts[1] != "None" else ""
        return coarse, tech


_GLOBAL_VOCAB = None


def get_joint_vocab(network_observable_only: bool = True) -> JointAttackVocab:
    global _GLOBAL_VOCAB
    if _GLOBAL_VOCAB is None or _GLOBAL_VOCAB.network_observable_only != network_observable_only:
        _GLOBAL_VOCAB = JointAttackVocab(network_observable_only=network_observable_only)
    return _GLOBAL_VOCAB

