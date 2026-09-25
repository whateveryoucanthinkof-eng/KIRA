"""Load a small real input once through bita/train.py's own loader and cache it.

    python bita/fast/prep_data.py --ctu13_dir <dir with 7/> --out <file.pkl>

The bench/equivalence harness (bita/fast/bench.py) reads the pickle, so each
A/B run does not re-ingest the capture.
"""
import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # bita/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # repo root

import train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctu13_dir", required=True)
    ap.add_argument("--split_scheme", default="cross_year_ctu")
    ap.add_argument("--train_splits", default="train")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    graph_df, edge_features, node_features, category_mapping = T.load_and_preprocess_unified_dataset(
        ctu13_dir=a.ctu13_dir, splits=tuple(a.train_splits.split(",")),
        scheme=a.split_scheme, window_seconds=T._contract_window_seconds())
    with open(a.out, "wb") as fh:
        pickle.dump(dict(graph_df=graph_df, edge_features=edge_features,
                         node_features=node_features, category_mapping=category_mapping), fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print("records", len(graph_df), "edge_features", edge_features.shape,
          "nodes", node_features.shape, "categories", category_mapping)


if __name__ == "__main__":
    main()
