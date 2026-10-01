"""cmp_ckpt.py A.pt B.pt : exact (bitwise) comparison of two checkpoints, recursively.
Keys whose name says they are wall-clock (seconds / time / manifest timestamps) are skipped."""
import sys, math, numpy as np, torch
SKIP = ("seconds", "wall", "timestamp", "created", "manifest", "fingerprint", "elapsed", "time_s", "output", "spill", "branch_a_checkpoint")
diffs = []; n = [0]
def same(a, b, path):
    n[0] += 1
    if any(s in path.lower() for s in SKIP): return
    if torch.is_tensor(a) or torch.is_tensor(b):
        ok = torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.nan_to_num(7.7), b.nan_to_num(7.7)) and torch.equal(a.isnan(), b.isnan()) if a.is_floating_point() else (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a, b))
        if not ok:
            d = (a.double() - b.double()).abs().max().item() if torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape else "shape/type"
            diffs.append(f"{path}: tensor differs (max abs {d})")
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")):
            diffs.append(f"{path}: array differs")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b): diffs.append(f"{path}: keys {set(a) ^ set(b)}")
        for k in a:
            if k in b: same(a[k], b[k], f"{path}.{k}")
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b): diffs.append(f"{path}: len {len(a)} vs {len(b)}"); return
        for i, (x, y) in enumerate(zip(a, b)): same(x, y, f"{path}[{i}]")
        return
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b): return
    if a != b: diffs.append(f"{path}: {a!r} vs {b!r}")
A = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
B = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
same(A, B, "ckpt")
print(f"compared {n[0]} leaves; {len(diffs)} differences")
for d in diffs[:30]: print("  ", d)
sys.exit(1 if diffs else 0)
