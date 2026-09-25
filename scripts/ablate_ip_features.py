#!/usr/bin/env python3
"""Is the host address a label proxy? — the IP-octet ablation.

## The question

`data_unification/ip_features.py` gives every host 12 intrinsic features derived
from its address alone, four of which are the raw octets scaled to [0, 1].
They were added for a good and measured reason: the TGNE encoder was built with
`node_features = np.zeros(...)`, so an unseen host carried no signal at all and
the link decoder scored it at chance. The octets took inductive validation AUC
from 0.5043 to 0.8330.

That fixes link prediction. It also creates a hazard for everything downstream,
because the octets are an *identifier*. On this corpus the audit records that
Recon is carried by a single host (172.16.0.1, all 158,930 records) and C2 by
about ten, and that nine of ten CIC-2018 days have no Src/Dst IP columns at all
so `cic2018_adapter` fabricates addresses from the row index. Under those
conditions "which address is this" can be most of the way to "which attack is
this" — offline. On a live SPAN feed the addresses are different, so whatever
the model learned from them transfers as noise.

The `branch_a_lstm.pt` checkpoint makes the concern concrete rather than
theoretical: it records `val_traj_len_median = 1.0` over 2,172,277 training
hosts. The median host appears in exactly ONE window, so with `history_steps=15`
fourteen of its fifteen input steps are zero padding and it has essentially no
behaviour to be recognised by. Identity is the only thing left.

## What this measures

Four arms. A and A2 need no encoder and no GPU and are the cheap decisive
test; B and C need the encoder and answer the deployment question.

    A   ip_only        Train a classifier on the 12 IP features ALONE against
                       the label. Whatever it scores is an upper bound on how
                       much of the label is recoverable from the address. Run
                       twice:
                         transductive  same hosts in train and test -> how much
                                       shortcut is AVAILABLE to memorise
                         inductive     disjoint hosts               -> whether
                                       address structure GENERALISES
                       The gap between them is the overfitting exposure.

    A2  subnet_only    Same, with octet3 and octet4 masked out, keeping the
                       /8 and /16 and the address-class flags. Separates
                       "this subnet is a server VLAN", which is legitimate
                       topology, from "this exact machine is the attacker",
                       which is memorisation.

    B   zeroed_octets  Rebuild host trajectories with the four octets zeroed in
                       the encoder's node features, retrain the downstream head,
                       and compare. Answers what the pipeline actually loses.

    C   permuted       Train normally, then relabel host addresses in the TEST
                       split only, through a consistent bijection. A model that
                       learned behaviour is unaffected; one that learned
                       addresses collapses. This is the closest offline proxy
                       for moving to a live network.

## Reading the result

    A inductive AUC ~ 0.5          the address carries nothing transferable
    A transductive high, inductive
      near chance                  pure memorisation is available and does not
                                   generalise -- the offline number is inflated
    A2 close to A                  the signal is subnet-level topology, which
                                   is defensible and worth keeping
    A2 much lower than A           the signal is host identity, which is not
    C  large drop                  the deployed model will not transfer

Arms A and A2 are properties of the DATA and are worth running before any
training. Arms B and C are properties of a trained pipeline.

    python scripts/ablate_ip_features.py --cic-dir ~/Documents/SIH/DATA/CSV \
                                         --ctu-dir ~/Documents/SIH/CTU-13-Dataset \
                                         --arms a --rows-per-file 200000 --stride 20
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data_unification.ip_features import IP_FEATURE_NAMES, ip_node_features  # noqa: E402

#: Indices of the four raw octets inside the 12-D IP feature vector.
OCTET_IDX = tuple(IP_FEATURE_NAMES.index(n) for n in ("octet1", "octet2", "octet3", "octet4"))
#: octet3 (/24) and octet4 (host-within-subnet) are the identity-bearing pair.
HOST_IDENTITY_IDX = tuple(IP_FEATURE_NAMES.index(n) for n in ("octet3", "octet4"))


# ---------------------------------------------------------------------------
# Arm A / A2
# ---------------------------------------------------------------------------


def ip_matrix(ips: Sequence[str], mask: Sequence[int] = ()) -> np.ndarray:
    """(N, 12) IP features, with `mask` indices zeroed."""
    X = np.asarray([ip_node_features(ip) for ip in ips], dtype=np.float32)
    if mask:
        X = X.copy()
        X[:, list(mask)] = 0.0
    return X


def _fit_score(Xtr, ytr, Xte, yte, seed: int) -> Dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import f1_score, roc_auc_score

    out: Dict[str, Any] = {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "test_positive_rate": float(np.mean(yte)) if len(yte) else float("nan"),
    }
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        out["note"] = "a split has one class; AUC undefined"
        out["roc_auc"] = float("nan")
        out["macro_f1"] = float("nan")
        return out

    clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    out["roc_auc"] = float(roc_auc_score(yte, p))
    out["macro_f1"] = float(f1_score(yte, (p >= 0.5).astype(int), average="macro"))
    # A constant predictor's AUC is exactly 0.5, so lift over chance is the
    # number to read -- not accuracy, which the base rate can carry alone.
    out["auc_lift_over_chance"] = out["roc_auc"] - 0.5

    # WHICH address feature carries it. Without this the arm says only that
    # "the address predicts the label", and the two possible causes want
    # opposite responses: `is_private` / `is_global` separating an external
    # attacker from internal victims is a property of how the capture was
    # staged, while `octet3` / `octet4` picking out one machine is
    # memorisation. Permutation importance on the TEST split, so it measures
    # reliance rather than what the tree happened to split on.
    rng = np.random.default_rng(seed)
    base = out["roc_auc"]
    imp: Dict[str, float] = {}
    for j in range(Xte.shape[1]):
        if float(Xte[:, j].std()) == 0.0:
            imp[IP_FEATURE_NAMES[j]] = 0.0
            continue
        Xp = Xte.copy()
        Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
        imp[IP_FEATURE_NAMES[j]] = float(base - roc_auc_score(yte, clf.predict_proba(Xp)[:, 1]))
    out["permutation_importance"] = {k: round(v, 4) for k, v in imp.items()}
    out["top_features"] = sorted(imp.items(), key=lambda kv: -kv[1])[:4]
    return out


def arm_ip_only(
    ips: Sequence[str],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    seed: int = 42,
    mask: Sequence[int] = (),
    name: str = "ip_only",
) -> Dict[str, Any]:
    """How much of the label is recoverable from the address alone.

    Two regimes, because they answer different questions and only reporting one
    is how this kind of result gets misread:

      transductive  random row split, so the same host appears on both sides.
                    This is memorisation capacity, and it is exactly the regime
                    an offline benchmark that splits on rows is measuring.
      inductive     hosts disjoint between train and test. This is whether the
                    address STRUCTURE generalises to a machine never seen.
    """
    rng = np.random.default_rng(seed)
    X = ip_matrix(ips, mask)
    y = np.asarray(labels, dtype=int)
    g = np.asarray(groups)

    n = len(y)
    perm = rng.permutation(n)
    cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]
    transductive = _fit_score(X[tr], y[tr], X[te], y[te], seed)

    hosts = np.asarray(sorted(set(ips)))
    rng.shuffle(hosts)
    held = set(hosts[: max(1, int(0.3 * len(hosts)))])
    m = np.array([ip in held for ip in ips])
    inductive = _fit_score(X[~m], y[~m], X[m], y[m], seed)

    return {
        "arm": name,
        "features_used": [f for i, f in enumerate(IP_FEATURE_NAMES) if i not in set(mask)],
        "masked": [IP_FEATURE_NAMES[i] for i in mask],
        "n_rows": n,
        "n_hosts": int(len(hosts)),
        "n_groups": int(len(set(g))),
        "transductive": transductive,
        "inductive": inductive,
        "memorisation_gap": (
            transductive.get("roc_auc", float("nan")) - inductive.get("roc_auc", float("nan"))
        ),
    }


def concentration_report(ips: Sequence[str], categories: Sequence[str]) -> Dict[str, Any]:
    """How few hosts carry each attack class.

    This is the mechanism behind any result above: a class carried by one host
    is perfectly predictable from that host's address, and no model can be
    stopped from noticing.
    """
    by_cat: Dict[str, Counter] = {}
    for ip, cat in zip(ips, categories):
        by_cat.setdefault(str(cat), Counter())[ip] += 1

    out = {}
    for cat, c in sorted(by_cat.items()):
        total = sum(c.values())
        top_ip, top_n = c.most_common(1)[0]
        out[cat] = {
            "records": total,
            "distinct_hosts": len(c),
            "top_host": top_ip,
            "top_host_share": round(top_n / total, 4),
            "hosts_covering_90pct": _hosts_for_share(c, 0.90),
        }
    return out


def _hosts_for_share(counter: Counter, share: float) -> int:
    total = sum(counter.values())
    acc, k = 0, 0
    for _, v in counter.most_common():
        acc += v
        k += 1
        if acc >= share * total:
            break
    return k


# ---------------------------------------------------------------------------
# Arm C: address permutation
# ---------------------------------------------------------------------------


def permute_addresses(ips: Sequence[str], *, seed: int = 42, preserve_subnet: bool = False) -> Dict[str, str]:
    """A consistent bijection over the host addresses present.

    `preserve_subnet=True` permutes only the last octet inside each /24, which
    keeps topology intact and changes only host identity -- so a drop under it
    is specifically identity memorisation rather than lost subnet structure.
    """
    rng = np.random.default_rng(seed)
    uniq = sorted(set(ips))
    if not preserve_subnet:
        shuffled = list(uniq)
        rng.shuffle(shuffled)
        return dict(zip(uniq, shuffled))

    by_subnet: Dict[str, List[str]] = {}
    for ip in uniq:
        parts = ip.split(".")
        key = ".".join(parts[:3]) if len(parts) == 4 else ip
        by_subnet.setdefault(key, []).append(ip)
    mapping: Dict[str, str] = {}
    for key, members in by_subnet.items():
        shuffled = list(members)
        rng.shuffle(shuffled)
        mapping.update(dict(zip(members, shuffled)))
    return mapping


# ---------------------------------------------------------------------------
# Corpus loading (arms A / A2 only need ip + label + group)
# ---------------------------------------------------------------------------


def load_ip_label_pairs(cic_dir: Path, ctu_dir: Path, rows: Optional[int], stride: int):
    """(ips, labels, categories, groups) from the corpora, one row per flow.

    Both endpoints are emitted, because that is what
    `HostTrajectoryExtractor` does when attributing a window label to hosts.
    """
    from data_unification.cic2018_adapter import CIC2018Adapter
    from data_unification.ctu13_adapter import CTU13Adapter
    from data_unification.label_filter import drop_unresolved
    from data_unification.split_policy import capture_name_for_path

    ips: List[str] = []
    labels: List[int] = []
    cats: List[str] = []
    groups: List[str] = []

    def strided(gen, want):
        out = []
        for i, r in enumerate(gen):
            if i % stride == 0:
                out.append(r)
                if want is not None and len(out) >= want:
                    break
        return out

    cap = None if rows is None else rows * stride

    def add(path: str, recs):
        recs, _ = drop_unresolved(recs)
        dataset, capture = capture_name_for_path(path)
        g = f"{dataset}|{capture}"
        for r in recs:
            for ip in (r.src_ip, r.dst_ip):
                ips.append(ip)
                labels.append(int(bool(r.is_attack)))
                cats.append(r.coarse_category)
                groups.append(g)
        print(f"  {capture[-34:]:<34} {len(recs):>8} flows")

    for f in sorted(glob.glob(str(cic_dir / "*.csv"))):
        add(f, strided(CIC2018Adapter().parse_file(f, max_rows=cap), rows))
    for f in sorted(glob.glob(str(ctu_dir / "*/*.binetflow"))):
        add(f, strided(CTU13Adapter().parse_netflow_csv(f, max_rows=cap), rows))

    return ips, labels, cats, groups


def print_arm(res: Dict[str, Any]) -> None:
    t, i = res["transductive"], res["inductive"]
    print(f"\n  {res['arm']}   ({res['n_rows']} rows, {res['n_hosts']} hosts, "
          f"{res['n_groups']} captures)")
    if res["masked"]:
        print(f"    masked: {', '.join(res['masked'])}")
    for label, m in (("transductive (same hosts)", t), ("inductive  (new hosts)", i)):
        if "note" in m:
            print(f"    {label:<28} {m['note']}")
            continue
        print(f"    {label:<28} ROC-AUC {m['roc_auc']:.4f}  "
              f"(lift {m['auc_lift_over_chance']:+.4f})  macro-F1 {m['macro_f1']:.4f}  "
              f"base rate {m['test_positive_rate']:.4f}")
    if np.isfinite(res["memorisation_gap"]):
        print(f"    memorisation gap             {res['memorisation_gap']:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cic-dir", type=Path, required=True)
    ap.add_argument("--ctu-dir", type=Path, required=True)
    ap.add_argument("--rows-per-file", type=int, default=200000)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", default="a",
                    help="which arms to run: 'a' (ip_only + subnet_only, no encoder needed)")
    ap.add_argument("--results", type=Path, default=REPO / "results/ip_ablation.json")
    args = ap.parse_args()

    print("IP-octet ablation")
    print(f"  octets at indices {OCTET_IDX} of {len(IP_FEATURE_NAMES)} IP features")
    print("\nLoading:")
    ips, labels, cats, groups = load_ip_label_pairs(
        args.cic_dir, args.ctu_dir, args.rows_per_file, args.stride)
    if not ips:
        print("no records loaded")
        return 1
    print(f"  {len(ips)} host-rows, {len(set(ips))} distinct addresses, "
          f"attack rate {np.mean(labels):.4f}")

    out: Dict[str, Any] = {
        "n_rows": len(ips),
        "n_hosts": len(set(ips)),
        "attack_rate": float(np.mean(labels)),
        "class_concentration": concentration_report(ips, cats),
    }

    print("\nClass concentration (how few hosts carry each class):")
    for cat, c in out["class_concentration"].items():
        print(f"  {cat:<16} {c['records']:>9} records over {c['distinct_hosts']:>7} hosts | "
              f"top host {c['top_host']} holds {c['top_host_share']:.1%} | "
              f"90% needs {c['hosts_covering_90pct']} host(s)")

    if "a" in args.arms:
        full = arm_ip_only(ips, labels, groups, seed=args.seed, name="A  ip_only (all 12)")
        subnet = arm_ip_only(ips, labels, groups, seed=args.seed,
                             mask=HOST_IDENTITY_IDX, name="A2 subnet_only (octet3/4 masked)")
        none = arm_ip_only(ips, labels, groups, seed=args.seed,
                           mask=OCTET_IDX, name="A3 no_octets (all four masked)")
        out["arms"] = {"ip_only": full, "subnet_only": subnet, "no_octets": none}
        print("\nResults:")
        for r in (full, subnet, none):
            print_arm(r)

        print("\nVerdict:")
        fi = full["inductive"].get("roc_auc", float("nan"))
        ft = full["transductive"].get("roc_auc", float("nan"))
        si = subnet["inductive"].get("roc_auc", float("nan"))
        if np.isfinite(ft) and ft > 0.7:
            print(f"  The address alone recovers the label at ROC-AUC {ft:.3f} on seen hosts.")
            print( "  Any offline split that does not hold hosts out is measuring this.")
        if np.isfinite(fi) and np.isfinite(si):
            if fi - si > 0.05:
                print(f"  Host identity carries {fi - si:+.3f} AUC beyond subnet structure:")
                print( "  that part is memorisation and will not survive a change of network.")
            else:
                print( "  Masking octet3/octet4 costs little, so the transferable signal is")
                print( "  subnet-level topology rather than host identity.")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nresults: {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
