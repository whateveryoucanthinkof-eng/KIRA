"""Synthetic scale input for bita/fast/bench.py: the real CTU-13 scenario 7
schema at the node count where training was measured slowing down.

    python bita/fast/make_scale_input.py --like ctu7.pkl --nodes 1500000 --edges 3000000 --out scale.pkl

Edge and node feature rows are resampled from the real input; endpoints are
drawn from a heavy-tailed (Zipf-like) distribution so hubs exist, as in
network traffic; timestamps increase. This measures COST at scale (the
reference's O(n_nodes) memory clone/backward and its per-group Python loops
over every pending message), not model quality.
"""
import argparse
import pickle

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True)
    ap.add_argument("--nodes", type=int, default=1_500_000)
    ap.add_argument("--edges", type=int, default=3_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    with open(a.like, "rb") as fh:
        d = pickle.load(fh)
    n, m = a.nodes, a.edges

    def endpoints():
        # rank ~ Zipf(1.2) truncated to n, mapped through a random permutation of ids 1..n
        r = rng.zipf(1.2, size=m)
        r = np.where(r > n, rng.integers(1, n + 1, size=m), r)
        perm = rng.permutation(n) + 1
        return perm[r - 1]

    u, i = endpoints(), endpoints()
    same = u == i
    i[same] = (i[same] % n) + 1
    ts = 1.3e9 + np.sort(rng.uniform(0, 86400.0, size=m))
    label = (rng.random(m) < 0.01).astype(np.int32)
    g = pd.DataFrame(dict(u=u.astype(np.int64), i=i.astype(np.int64), ts=ts, label=label,
                          idx=np.arange(1, m + 1, dtype=np.int64), source=np.zeros(m, np.int8),
                          capture=np.zeros(m, np.int16)))
    ef_src = d["edge_features"]
    edge_features = np.zeros((m + 1, ef_src.shape[1]), np.float32)
    edge_features[1:] = ef_src[rng.integers(1, len(ef_src), size=m)]
    nf_src = d["node_features"]
    node_features = nf_src[rng.integers(0, len(nf_src), size=n + 1)].astype(np.float32)
    with open(a.out, "wb") as fh:
        pickle.dump(dict(graph_df=g, edge_features=edge_features, node_features=node_features,
                         category_mapping=d["category_mapping"]), fh, protocol=pickle.HIGHEST_PROTOCOL)
    print("nodes", n, "edges", m, "distinct endpoints", len(np.unique(np.concatenate([u, i]))))


if __name__ == "__main__":
    main()
