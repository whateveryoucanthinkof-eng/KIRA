"""A/B harness for the fast TGN step: speed, profile and numerical equivalence.

Builds the encoder exactly as bita/train.py does (same defaults as the plan's
encoder command: --use_memory --n_degree 10, BiTA aggregator, GRU memory,
graph attention, backprop_every 8, focal loss, TrainingGuard with clip 100),
runs the same inner training loop on the first N batches of a cached real
input (bita/fast/prep_data.py), and records everything needed to compare two
runs:

    python bita/fast/bench.py --data ctu7.pkl --mode ref  --batches 800 --dump ref.pt
    python bita/fast/bench.py --data ctu7.pkl --mode fast --batches 800 --dump fast.pt
    python bita/fast/bench.py --compare ref.pt fast.pt

`--mode ref` is the unmodified code; `--mode fast` calls
bita.fast.enable_fast_tgn(model) first. Both runs start from the same seed, so
they initialise the same weights, draw the same negatives and -- as long as the
fast path issues the same random ops in the same order -- the same dropout
masks.
"""
import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # bita/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # repo root

import train as T  # noqa: E402
from model.extentedtgn import ExtendedTGN  # noqa: E402
from model.time_encoding import TimeEncode  # noqa: E402
from utils.utils import RandEdgeSampler, get_neighbor_finder  # noqa: E402
from evaluation.eval_edge_prediction_with_categories import eval_edge_prediction_with_categories  # noqa: E402
from cyberworld_v4.training_guard import TrainingGuard, default_warmup_steps  # noqa: E402


def build(args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with open(args.data, "rb") as fh:
        d = pickle.load(fh)
    graph_df, edge_features, node_features = d["graph_df"], d["edge_features"], d["node_features"]
    num_categories = len(d["category_mapping"])
    node_features, edge_features, full_data, train_data, val_data, test_data, nn_val, nn_test = \
        T.split_data(graph_df, edge_features, node_features, different_new_nodes=True)
    if args.host_edges:
        # Out-of-core edge features (v5.5o): a np.memmap makes TGN keep them on
        # the host as HostEdgeFeatures, exactly as full-scale training does.
        mm = np.memmap(args.host_edges, dtype=np.float32, mode="w+", shape=edge_features.shape)
        mm[:] = edge_features
        mm.flush()
        edge_features = np.memmap(args.host_edges, dtype=np.float32, mode="r", shape=edge_features.shape)
    train_ngh = get_neighbor_finder(train_data, uniform=False)
    full_ngh = get_neighbor_finder(full_data, uniform=False)
    node_group = None
    if "capture" in graph_df.columns:
        node_group = np.full(int(max(graph_df.u.max(), graph_df.i.max())) + 1, -1, dtype=np.int64)
        node_group[graph_df.u.values] = graph_df["capture"].values
        node_group[graph_df.i.values] = graph_df["capture"].values
    train_sampler = RandEdgeSampler(train_data.sources, train_data.destinations, node_group=node_group)
    val_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0, node_group=node_group)
    ms, ss, md, sd = T.compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)
    tgn = ExtendedTGN(
        neighbor_finder=train_ngh, node_features=node_features, edge_features=edge_features,
        device=device, n_layers=1, n_heads=2, dropout=0.1, use_memory=True,
        message_dimension=100, memory_dimension=12, memory_update_at_start=True,
        embedding_module_type="graph_attention", message_function="identity",
        aggregator_type="bigru_transformer", memory_updater_type="gru",
        n_neighbors=args.n_degree, mean_time_shift_src=ms, std_time_shift_src=ss,
        mean_time_shift_dst=md, std_time_shift_dst=sd, num_categories=num_categories,
    ).to(device)
    for m in tgn.modules():
        if isinstance(m, TimeEncode):
            m.requires_grad_(False)
    alpha = T.FocalLoss.inverse_frequency_alpha(train_data.labels, num_categories)
    return dict(tgn=tgn, train_data=train_data, val_data=val_data, full_ngh=full_ngh,
                train_ngh=train_ngh, train_sampler=train_sampler, val_sampler=val_sampler,
                edge_criterion=nn.BCELoss(), category_criterion=T.FocalLoss(alpha=alpha, gamma=2.0))


def run(args):
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    if args.deterministic:
        # Makes the REFERENCE reproducible (its index_add_ uses atomics), so a
        # bit-identity claim is testable at all.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    B = build(args, device)
    tgn, train_data = B["tgn"], B["train_data"]
    if args.mode == "fast":
        from fast import enable_fast_tgn
        enable_fast_tgn(tgn, level=args.level)
    optimizer = torch.optim.Adam([p for p in tgn.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-5)
    num_instance = len(train_data.sources)
    num_batch_epoch = math.ceil(num_instance / args.batch_size)
    guard = TrainingGuard("encoder", [tgn], optimizer, mode="max", patience=3, step_back_after=2,
                          warmup_steps=default_warmup_steps(math.ceil(num_batch_epoch / args.backprop_every)),
                          clip_norm=100.0, log=lambda *_: None)
    num_batch = min(num_batch_epoch, args.batches) if args.batches else num_batch_epoch
    ec, cc = B["edge_criterion"], B["category_criterion"]

    tgn.train()
    tgn.memory.__init_memory__()
    tgn.set_neighbor_finder(B["train_ngh"])
    rec = dict(total=[], edge=[], cat=[], outputs=[], grads_first=None, norms=[])
    prof = None
    if args.profile:
        from torch.profiler import profile, ProfilerActivity, schedule
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       schedule=schedule(wait=0, warmup=args.profile_warm, active=args.profile_steps, repeat=1),
                       record_shapes=False)
        prof.start()
    t_start, t_mark, mark_batch = time.time(), None, None
    rates = []
    for k in range(0, num_batch, args.backprop_every):
        if k >= args.warm_batches and t_mark is None:
            torch.cuda.synchronize()
            t_mark, mark_batch = time.time(), k
        loss, cat_total = 0.0, 0.0
        optimizer.zero_grad()
        for j in range(args.backprop_every):
            bi = k + j
            if bi >= num_batch:
                continue
            s, e = bi * args.batch_size, min(num_instance, (bi + 1) * args.batch_size)
            src, dst = train_data.sources[s:e], train_data.destinations[s:e]
            eidx, ts, cat = train_data.edge_idxs[s:e], train_data.timestamps[s:e], train_data.labels[s:e]
            size = len(src)
            _, neg = B["train_sampler"].sample(size, sources=src, destinations=dst)
            pos_prob, neg_prob, logits = tgn.compute_edge_probabilities_and_categories(
                src, dst, neg, ts, eidx, n_neighbors=args.n_degree)
            bel = ec(pos_prob.squeeze(-1), torch.ones(size, device=device)) + \
                ec(neg_prob.squeeze(-1), torch.zeros(size, device=device))
            bcl = cc(logits, torch.tensor(cat, dtype=torch.long, device=device))
            loss = loss + bel
            cat_total = cat_total + bcl
            if args.dump and bi < args.dump_outputs:
                rec["outputs"].append((pos_prob.detach().cpu(), neg_prob.detach().cpu(), logits.detach().cpu()))
        total = (loss + 15.0 * cat_total) / args.backprop_every
        if args.dump and rec["grads_first"] is None:
            total.backward()
            rec["grads_first"] = {n: p.grad.detach().cpu().clone() for n, p in tgn.named_parameters()
                                  if p.requires_grad and p.grad is not None}
            ok = guard.step_after_backward()
        else:
            ok = guard.backward_step(total)
        if args.dump and ok:
            rec["total"].append(float(total.item()))
            rec["edge"].append(float(loss.item()) / args.backprop_every)
            rec["cat"].append(float(cat_total.item()) / args.backprop_every)
        tgn.memory.detach_memory()
        if prof is not None:
            prof.step()
        if args.log_every and (k // args.backprop_every) % args.log_every == 0 and t_mark is not None and k > mark_batch:
            torch.cuda.synchronize()
            rates.append((k, (k - mark_batch) / (time.time() - t_mark)))
            print(f"batch {k}: {rates[-1][1]:.2f} batch/s", flush=True)
    torch.cuda.synchronize()
    t_end = time.time()
    if prof is not None:
        prof.stop()
    measured = num_batch - (mark_batch or 0)
    rate = measured / (t_end - t_mark) if t_mark else float("nan")
    print(f"mode={args.mode} level={args.level} batches={num_batch} measured={measured} "
          f"rate={rate:.2f} batch/s total_wall={t_end - t_start:.1f}s "
          f"peak_gpu_mem={torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)

    if prof is not None:
        ka = prof.key_averages()
        print(ka.table(sort_by="self_cuda_time_total", row_limit=30))
        print(ka.table(sort_by="self_cpu_time_total", row_limit=30))
        n_kernels = sum(1 for ev in prof.events() if ev.device_type == torch.autograd.DeviceType.CUDA)
        cuda_total = sum(ev.self_device_time_total for ev in ka) / 1e3
        cpu_total = sum(ev.self_cpu_time_total for ev in ka) / 1e3
        print(f"PROFILE over {args.profile_steps} optimizer steps ({args.profile_steps * args.backprop_every} batches): "
              f"device kernels={n_kernels} ({n_kernels / (args.profile_steps * args.backprop_every):.0f}/batch), "
              f"self device time={cuda_total:.1f} ms, self cpu time={cpu_total:.1f} ms")
        if args.trace:
            prof.export_chrome_trace(args.trace)

    if args.dump:
        val = None
        if args.val:
            tgn.eval()
            tgn.set_neighbor_finder(B["full_ngh"])
            vd = B["val_data"]
            if args.val_limit:
                vd = T.Data(vd.sources[:args.val_limit], vd.destinations[:args.val_limit],
                            vd.timestamps[:args.val_limit], vd.edge_idxs[:args.val_limit], vd.labels[:args.val_limit])
            val = eval_edge_prediction_with_categories(model=tgn, negative_edge_sampler=B["val_sampler"], data=vd,
                                                       n_neighbors=args.n_degree, edge_criterion=ec,
                                                       category_criterion=cc)
        torch.save(dict(rec=rec, rate=rate, rates=rates,
                        params={n: p.detach().cpu().clone() for n, p in tgn.named_parameters()},
                        memory=tgn.memory.memory.detach().cpu().clone(),
                        last_update=tgn.memory.last_update.detach().cpu().clone(),
                        val=val), args.dump)
        print("dumped", args.dump)


def compare(a_path, b_path):
    a, b = torch.load(a_path, weights_only=False), torch.load(b_path, weights_only=False)

    def diff(x, y):
        x, y = torch.as_tensor(x).double(), torch.as_tensor(y).double()
        ad = (x - y).abs()
        return float(ad.max()), float((ad / y.abs().clamp_min(1e-12)).max()), bool(torch.equal(x, y))

    ra, rb = a["rec"], b["rec"]
    n = min(len(ra["total"]), len(rb["total"]))
    print(f"steps: {len(ra['total'])} vs {len(rb['total'])}")
    for key in ("total", "edge", "cat"):
        x, y = np.array(ra[key][:n]), np.array(rb[key][:n])
        d = np.abs(x - y)
        eq = bool((x == y).all())
        q = max(1, n // 4)
        print(f"loss[{key}]: bit-identical={eq} max|d|={d.max():.3e} max rel={np.max(d / np.maximum(np.abs(y), 1e-12)):.3e} "
              f"| per quarter max|d|: " + " ".join(f"{d[i:i + q].max():.2e}" for i in range(0, n, q)))
    no = min(len(ra["outputs"]), len(rb["outputs"]))
    if no:
        mx = [0.0, 0.0, 0.0]
        eqs = True
        for (pa, na, la), (pb, nb, lb) in zip(ra["outputs"][:no], rb["outputs"][:no]):
            for i, (u, v) in enumerate(((pa, pb), (na, nb), (la, lb))):
                m, _, e = diff(u, v)
                mx[i] = max(mx[i], m)
                eqs &= e
        print(f"per-batch outputs over first {no} batches (pos_prob, neg_prob, category_logits): "
              f"max|d|={mx[0]:.3e} {mx[1]:.3e} {mx[2]:.3e} bit-identical={eqs}")
    ga, gb = ra["grads_first"], rb["grads_first"]
    if ga and gb:
        worst = max(((diff(ga[k], gb[k]), k) for k in ga), key=lambda t: t[0][0])
        alleq = all(torch.equal(ga[k], gb[k]) for k in ga) and set(ga) == set(gb)
        gnorm = max(float(ga[k].abs().max()) for k in ga)
        print(f"grads after the first optimizer step's backward: bit-identical={alleq} "
              f"max|d|={worst[0][0]:.3e} (in {worst[1]}; largest |grad| {gnorm:.3e})")
    pa, pb = a["params"], b["params"]
    worst = max(((diff(pa[k], pb[k]), k) for k in pa), key=lambda t: t[0][0])
    alleq = all(torch.equal(pa[k], pb[k]) for k in pa)
    print(f"params after {n} steps: bit-identical={alleq} max|d|={worst[0][0]:.3e} (in {worst[1]})")
    m = diff(a["memory"], b["memory"])
    print(f"memory table: bit-identical={m[2]} max|d|={m[0]:.3e}; last_update bit-identical="
          f"{torch.equal(a['last_update'], b['last_update'])}")
    if a.get("val") and b.get("val"):
        names = {0: "AP", 1: "AUC", 2: "MRR", 4: "CatAcc", 5: "CatLoss", 8: "MacroF1"}
        print("val: " + "  ".join(f"{nm} {a['val'][i]:.6f}/{b['val'][i]:.6f}" for i, nm in names.items()))
    print(f"rate: {a['rate']:.2f} vs {b['rate']:.2f} batch/s ({b['rate'] / a['rate']:.2f}x)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--mode", choices=["ref", "fast"], default="ref")
    ap.add_argument("--level", type=int, default=99, help="fast path: highest optimisation level to enable")
    ap.add_argument("--batches", type=int, default=400)
    ap.add_argument("--warm_batches", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--backprop_every", type=int, default=8)
    ap.add_argument("--n_degree", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump")
    ap.add_argument("--dump_outputs", type=int, default=64)
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--host_edges", help="path for a memmap copy of the edge features (HostEdgeFeatures path)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile_warm", type=int, default=2)
    ap.add_argument("--profile_steps", type=int, default=4)
    ap.add_argument("--trace")
    ap.add_argument("--log_every", type=int, default=0)
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
