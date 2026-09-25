"""Experiment manifests (spec 45, 46, 47).

The question v3 could not answer: "give me the exact command and configuration
that produced this checkpoint." Its documentation, manifests, checkpoint
metadata and training code disagreed with each other, and the named trainers
could not reproduce the shipped weights at all.

A manifest pins everything needed to rerun: git commit, dataset hash, seed,
full config, dependency versions, split assignment. It is embedded *inside* the
checkpoint, so a weight file can never drift from its provenance the way v3's
did.

Seeding covers python, numpy and torch, including cuDNN determinism — spec 47.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CyberWorldConfig, DEFAULT_CONFIG


def _run(cmd: List[str]) -> Optional[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def git_state(repo: Optional[Path] = None) -> Dict[str, Any]:
    cwd = str(repo) if repo else None
    def g(*a: str) -> Optional[str]:
        try:
            r = subprocess.run(["git", *a], capture_output=True, text=True, timeout=10, cwd=cwd)
            return r.stdout.strip() or None
        except Exception:
            return None
    dirty = g("status", "--porcelain")
    return {
        "commit": g("rev-parse", "HEAD"),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the commit does not describe the code that ran.
        "dirty": bool(dirty),
        "dirty_files": (dirty or "").splitlines()[:20],
    }


def hash_paths(paths: List[Path | str], *, sample_bytes: int = 1 << 20) -> str:
    """Stable digest over a dataset's files.

    Hashes name + size + a head/tail sample rather than full contents, so a
    560 GB corpus can be fingerprinted in seconds. Detects substitution and
    truncation; will not detect a byte flipped in the middle of a large file.
    """
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        path = Path(p)
        h.update(path.name.encode())
        if not path.is_file():
            continue
        size = path.stat().st_size
        h.update(str(size).encode())
        with open(path, "rb") as f:
            h.update(f.read(min(sample_bytes, size)))
            if size > sample_bytes:
                f.seek(-min(sample_bytes, size), os.SEEK_END)
                h.update(f.read())
    return h.hexdigest()


def dependency_versions() -> Dict[str, str]:
    out: Dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "torch", "sklearn", "pandas", "pyarrow", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "absent"
    try:
        import torch
        out["cuda"] = torch.version.cuda or "cpu"
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        pass
    return out


def set_all_seeds(seed: int, *, deterministic: bool = True) -> Dict[str, Any]:
    """Seed every source of randomness. Returns what was actually set."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    info: Dict[str, Any] = {"seed": seed, "python": True, "numpy": False, "torch": False}
    try:
        import numpy as np
        np.random.seed(seed)
        info["numpy"] = True
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        info["torch"] = True
        info["cudnn_deterministic"] = deterministic
    except Exception:
        pass
    return info


@dataclass
class ExperimentManifest:
    """Immutable record of one run."""

    experiment_id: str
    seed: int
    config: Dict[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG.to_dict())
    git: Dict[str, Any] = field(default_factory=git_state)
    dependencies: Dict[str, str] = field(default_factory=dependency_versions)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    dataset_hash: Optional[str] = None
    dataset_sources: List[str] = field(default_factory=list)
    split_assignment: Optional[Dict[str, Any]] = None
    command: List[str] = field(default_factory=lambda: list(sys.argv))
    notes: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        experiment_id: str,
        seed: int,
        config: Optional[CyberWorldConfig] = None,
        *,
        repo: Optional[Path] = None,
        **kw: Any,
    ) -> "ExperimentManifest":
        return cls(
            experiment_id=experiment_id,
            seed=seed,
            config=(config or DEFAULT_CONFIG).to_dict(),
            git=git_state(repo),
            **kw,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Digest over the fields that determine the result."""
        core = {
            "config": self.config,
            "seed": self.seed,
            "git_commit": self.git.get("commit"),
            "dataset_hash": self.dataset_hash,
        }
        return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]

    def save(self, directory: Path | str = "results/experiments") -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{self.experiment_id}.json"
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str))
        return p

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentManifest":
        d = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def attach_to_checkpoint(self, ckpt: Dict[str, Any]) -> Dict[str, Any]:
        """Embed provenance in the weights themselves."""
        ckpt["manifest"] = self.to_dict()
        ckpt["config"] = self.config
        ckpt["fingerprint"] = self.fingerprint()
        return ckpt

    def reproduction_command(self) -> str:
        c = self.git.get("commit", "<unknown>")
        warn = "  # WARNING: tree was dirty; commit does not describe the code that ran\n" if self.git.get("dirty") else ""
        return (
            f"git checkout {c}\n{warn}"
            f"CYBERWORLD_SEED={self.seed} {' '.join(self.command) or '<command not recorded>'}"
        )
