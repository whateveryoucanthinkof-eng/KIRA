"""Scoring a model on a dataset it was not trained on (CIC-2018 -> CIC-2017).

The one thing that must not happen
----------------------------------
CIC-IDS-2017 contains attack families CIC-IDS-2018 does not: PortScan (the
only reconnaissance class, 158,930 records from one host) and Heartbleed. A
model trained on 2018 has never seen a single example of them. Folding those
classes into macro-F1 as ordinary misses makes the model look worse at what it
learned, and hides the more important fact: it cannot name what it never saw.

So every technique/token metric here is reported three ways:

  * seen      classes present in BOTH training and test: the like-for-like
              score of what the model learned;
  * unseen    classes present in test but absent from training: reported one
              by one, with what the model predicted instead;
  * all       the conventional macro over every class present in test, for
              comparison with other work.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _f1_table(cm: np.ndarray):
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / np.maximum(predicted, 1), 0.0)
        recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    return support, predicted, precision, recall, f1


def unseen_class_report(train_counts, confusion, names: Optional[Dict[int, str]] = None,
                        top_k: int = 3) -> dict:
    """Split test-set classification quality by whether training saw the class.

    train_counts: per-class count in the TRAINING split, length C.
    confusion:    C x C test confusion matrix, rows = true, cols = predicted.
    """
    cm = np.asarray(confusion, dtype=np.int64)
    train_counts = np.asarray(train_counts, dtype=np.int64)
    C = cm.shape[0]
    if train_counts.shape[0] != C:
        raise ValueError(f"train_counts has {train_counts.shape[0]} classes, confusion has {C}")
    names = names or {}
    nm = lambda c: names.get(int(c), str(int(c)))  # noqa: E731

    support, predicted, precision, recall, f1 = _f1_table(cm)
    present = support > 0
    seen = train_counts > 0
    seen_present = present & seen
    unseen_present = present & ~seen
    total = int(support.sum())

    # Accuracy restricted to samples whose true class was seen in training.
    seen_rows = cm[seen_present]
    seen_acc = float(np.trace(cm[np.ix_(seen_present, seen_present)]) / max(int(seen_rows.sum()), 1))

    unseen = []
    for c in np.flatnonzero(unseen_present):
        row = cm[c]
        order = np.argsort(-row)[:top_k]
        unseen.append({
            "class": nm(c),
            "test_support": int(support[c]),
            "recall": float(recall[c]),
            "predicted_as": [{"class": nm(k), "fraction": float(row[k] / max(support[c], 1))}
                             for k in order if row[k] > 0],
        })

    return {
        "n_test": total,
        "classes_seen_in_training": [nm(c) for c in np.flatnonzero(seen)],
        "macro_f1_seen": float(f1[seen_present].mean()) if seen_present.any() else float("nan"),
        "accuracy_seen": seen_acc,
        "macro_f1_all": float(f1[present].mean()) if present.any() else float("nan"),
        "accuracy_all": float(np.trace(cm) / max(total, 1)),
        "test_fraction_in_unseen_classes": float(support[unseen_present].sum() / max(total, 1)),
        "unseen_classes": unseen,
        "per_class": {
            nm(c): {"f1": float(f1[c]), "precision": float(precision[c]), "recall": float(recall[c]),
                    "test_support": int(support[c]), "train_support": int(train_counts[c]),
                    "seen_in_training": bool(seen[c])}
            for c in range(C) if support[c] > 0 or predicted[c] > 0
        },
    }


def format_unseen_report(rep: dict, title: str = "held-out test") -> str:
    lines = [
        f"{title.upper()} -- by whether training saw the class:",
        f"  seen classes    macro-F1 {rep['macro_f1_seen']:.4f}   accuracy {rep['accuracy_seen']:.4f}",
        f"  all classes     macro-F1 {rep['macro_f1_all']:.4f}   accuracy {rep['accuracy_all']:.4f}",
        f"  {100 * rep['test_fraction_in_unseen_classes']:.2f}% of test samples belong to classes "
        f"training never contained",
    ]
    for u in rep["unseen_classes"]:
        instead = ", ".join(f"{p['class']} {100 * p['fraction']:.1f}%" for p in u["predicted_as"])
        lines.append(f"  UNSEEN {u['class']}: {u['test_support']} test samples, predicted as: {instead}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Branch B and DeepOP on a held-out store
# ---------------------------------------------------------------------------

def branch_b_on_store(wdt, risk_head, store, device, T: int, K: int, batch_size: int = 512) -> dict:
    """World-model skill against persistence on a store it was not trained on."""
    import torch
    from torch.utils.data import DataLoader
    from branch_b_world_model.train_branch_b import LazyHostRolloutDataset

    # The same samples and inputs as training and validation: short histories
    # (edge-padded), real elapsed times, padded future steps masked. This
    # called rollout(h, K) with no times -- telling a model trained on real
    # elapsed seconds that every step was 2 s apart -- and scored padded
    # future copies, which persistence predicts perfectly.
    ds = LazyHostRolloutDataset(store, T=T, K=K, min_history_steps=1)
    if len(ds) == 0:
        return {"n": 0}
    se_m = se_p = cnt = 0.0
    r_err = r_zero = r_cnt = 0.0
    n = 0
    wdt.eval()
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=batch_size):
            h, tgt = b["h_history"].to(device), b["h_future"].to(device)
            v = b["future_valid"].to(device)
            pred = wdt.rollout(h, K=K, t_history=b["t_history"].to(device),
                               t_future=b["t_future"].to(device))
            last = h[:, -1:, :].expand_as(tgt)
            w = v.unsqueeze(-1)
            se_m += float((((pred - tgt) ** 2) * w).sum())
            se_p += float((((last - tgt) ** 2) * w).sum())
            cnt += float(w.sum()) * tgt.shape[-1]
            if risk_head is not None:
                pr, _ = risk_head.forward_trajectory(pred)
                rt = b["risk_future"].to(device)
                r_err += float(((pr - rt).abs() * v).sum())
                r_zero += float((rt.abs() * v).sum())
                r_cnt += float(v.sum())
            n += tgt.shape[0]
    mse_m, mse_p = se_m / max(cnt, 1.0), se_p / max(cnt, 1.0)
    out = {"n": n, "mse_model": mse_m, "mse_persistence": mse_p,
           "skill": (1.0 - mse_m / mse_p) if mse_p > 0 else float("nan")}
    if r_cnt:
        out.update({"risk_mae_model": r_err / r_cnt, "risk_mae_predict_zero": r_zero / r_cnt})
    return out


def deepop_on_store(decoder, wdt, store, vocab, device, train_token_counts, T: int, K: int,
                    batch_size: int = 256) -> dict:
    """DeepOP free-running token quality on a held-out store, conditioned on
    Branch B's own rollout (what serving sees), split by seen/unseen tokens."""
    import torch
    from torch.utils.data import DataLoader
    from deepop_decoder.train_cwa_decoder import LazyCWADataset

    ds = LazyCWADataset(store, vocab, K=K, T=T, oversample=False)
    if len(ds) == 0:
        return {"n": 0}
    V = vocab.vocab_size
    cm = np.zeros((V, V), dtype=np.int64)
    hit_persist = total = 0
    decoder.eval(); wdt.eval()
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=batch_size):
            hist = b["h_history"].to(device)
            h_fut = wdt.rollout(hist, K=K, t_history=b["t_history"].to(device)
                                if "t_history" in b else None,
                                t_future=b["t_future"].to(device) if "t_future" in b else None)
            obs = b["obs_token"].to(device)
            pred, _ = decoder.forecast_sequence(
                h_fut, max_steps=K, observed_token=obs,
                observed_sequence=b["obs_tokens"].to(device), continuity_bonus=0.0,
                decode_names=False)
            tgt = b["target_tokens"].to(device)
            np.add.at(cm, (tgt.reshape(-1).cpu().numpy(), pred.reshape(-1).cpu().numpy()), 1)
            hit_persist += int((obs.unsqueeze(1).expand_as(tgt) == tgt).sum())
            total += int(tgt.numel())
    names = {i: ".".join(x for x in vocab.decode(i) if x) for i in range(V)}
    rep = unseen_class_report(train_token_counts, cm, names)
    rep["acc_persistence"] = hit_persist / max(total, 1)
    rep["n"] = total
    return rep
