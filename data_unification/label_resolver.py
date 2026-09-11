"""
Label Resolver.

Loads versioned label mapping tables and resolves heterogeneous raw dataset labels
into normalized coarse categories and MITRE ATT&CK technique IDs.
"""

import os
import re
from typing import Dict, List, Tuple, Optional
import pandas as pd


class LabelResolver:
    """Thread-safe, cached resolver for dataset-specific label mappings."""

    def __init__(self, label_maps_dir: Optional[str] = None):
        if label_maps_dir is None:
            label_maps_dir = os.path.join(os.path.dirname(__file__), "label_maps")
        self.label_maps_dir = label_maps_dir

        self.maps: Dict[str, Dict[str, Tuple[str, List[str], bool]]] = {
            "CIC": self._load_map("cic_label_map.csv"),
            "CTU": self._load_map("ctu_label_map.csv"),
            "WARDEN": self._load_map("warden_label_map.csv"),
        }

    def _normalize_key(self, text: str) -> str:
        """Normalizes text by removing non-ascii dash chars, lowercasing, and stripping whitespace."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = text.replace("\x96", "-").replace("–", "-").replace("—", "-")
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text

    def _load_map(self, filename: str) -> Dict[str, Tuple[str, List[str], bool]]:
        filepath = os.path.join(self.label_maps_dir, filename)
        if not os.path.exists(filepath):
            return {}

        df = pd.read_csv(filepath)
        mapping = {}
        for _, row in df.iterrows():
            raw = str(row["raw_label"])
            norm_key = self._normalize_key(raw)
            coarse = str(row["coarse_category"]).strip()
            raw_attck = row.get("attck_technique_ids", "")
            if pd.isna(raw_attck) or not str(raw_attck).strip():
                attck_list = []
            else:
                attck_list = [t.strip() for t in str(raw_attck).split(";") if t.strip()]
            is_attack = bool(int(row.get("is_attack", 0)))
            mapping[norm_key] = (coarse, attck_list, is_attack)
        return mapping

    def resolve(
        self, raw_label: str, source: str
    ) -> Tuple[str, List[str], bool]:
        """
        Resolves raw_label under the given source ('CIC2017', 'CIC2018', 'CTU13', 'WARDEN').
        Returns: (coarse_category, attck_technique_ids, is_attack)
        """
        norm_source = source.upper()
        if "CIC" in norm_source:
            lut = self.maps.get("CIC", {})
        elif "CTU" in norm_source:
            lut = self.maps.get("CTU", {})
        elif "WARDEN" in norm_source:
            lut = self.maps.get("WARDEN", {})
        else:
            lut = {}

        norm_key = self._normalize_key(raw_label)

        # 1. Exact normalized match
        if norm_key in lut:
            return lut[norm_key]

        # 2. Prefix / substring match for CTU botnet flows (e.g. 'flow=from-botnet...')
        if "ctu" in norm_source.lower():
            if "botnet" in norm_key:
                return ("C2", ["T1071"], True)
            if "normal" in norm_key or "background" in norm_key or "benign" in norm_key:
                return ("Benign", [], False)

        # 3. Substring heuristic fallback for CIC / Warden
        if "benign" in norm_key or "normal" in norm_key:
            return ("Benign", [], False)
        if "portscan" in norm_key or "scanning" in norm_key or "recon" in norm_key:
            return ("Recon", ["T1046"], True)
        if "ddos" in norm_key or "dos" in norm_key:
            return ("Impact", ["T1498"], True)
        if "bot" in norm_key:
            return ("C2", ["T1071"], True)
        if "brute" in norm_key or "patator" in norm_key:
            return ("InitialAccess", ["T1110"], True)
        if "infilt" in norm_key:
            return ("InitialAccess", ["T1190"], True)

        # Default fallback
        return ("Unknown", [], True if norm_key else False)


_DEFAULT_RESOLVER = None


def get_default_resolver() -> LabelResolver:
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = LabelResolver()
    return _DEFAULT_RESOLVER
