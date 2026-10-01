"""
control_backend/tactics.py
Kill-chain lanes for the console, derived from the models' own vocabulary.

Branch A emits ATT&CK technique ids ("T1046", "T1071.001", "Benign"); DeepOP
emits joint "Coarse.Technique" tokens ("Recon.T1595", "Benign.None"). The
console lays both out on one kill-chain strip, so each is reduced to a lane
with DeepOP's own consolidation (deepop_decoder/joint_vocab.py) rather than a
second hand-written mapping that could drift from it.

LANES lists every lane the console draws, in kill-chain order. MODEL_LANES is
the subset our models can actually emit: the joint vocabulary has no
Execution or LateralMovement token, so those two lanes are drawn but marked
under development.
"""

from typing import Optional

from deepop_decoder.joint_vocab import consolidate_network_technique

#: Every lane the console draws, in kill-chain order. CredentialAccess sits
#: before InitialAccess: on the wire, credential brute force against an
#: exposed service is how access is obtained.
LANES = [
    "Recon",
    "CredentialAccess",
    "InitialAccess",
    "Execution",
    "C2",
    "LateralMovement",
    "Exfiltration",
    "Impact",
]

#: Lanes the joint vocabulary can produce (NETWORK_MACRO_TECHNIQUES).
MODEL_LANES = ["Recon", "CredentialAccess", "InitialAccess", "C2", "Exfiltration", "Impact"]

BENIGN = "Benign"


def _is_technique_id(s: str) -> bool:
    return len(s) > 1 and s[0] == "T" and s[1].isdigit()


def split_token(label: Optional[str]) -> tuple[str, Optional[str]]:
    """("Recon", "T1595") from a DeepOP token or a Branch A technique id."""
    s = str(label or "").strip()
    if not s or s == BENIGN or s.startswith(BENIGN + "."):
        return BENIGN, None
    if _is_technique_id(s):
        lane, tech = consolidate_network_technique("", s)
        return lane, s if lane != BENIGN else None
    coarse, _, tech = s.partition(".")
    lane, canonical = consolidate_network_technique(coarse, tech or None)
    if lane == BENIGN:
        return BENIGN, None
    return lane, (tech if tech and tech != "None" else canonical)


def lane_of(label: Optional[str]) -> str:
    """Kill-chain lane of a technique id or DeepOP token; "Benign" if none."""
    return split_token(label)[0]


def lane_rank(lane: str) -> int:
    """Position on the kill chain, -1 for Benign/unknown."""
    try:
        return LANES.index(lane)
    except ValueError:
        return -1
