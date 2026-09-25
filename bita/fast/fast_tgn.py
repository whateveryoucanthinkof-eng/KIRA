"""The fast TGN training step (opt-in; see bita/fast/__init__.py).

Same model, same parameters, same data, same batch, same random stream: this
re-expresses TGN.compute_temporal_embeddings / BiTAAggregator.aggregate /
GraphAttentionEmbedding for the plan's configuration without per-message
Python work, without device->host synchronisation inside the step, and
without O(n_nodes) autograd nodes. Every module is called with the same
inputs, in the same order and at the same shapes as the reference, so dropout
draws the same masks; bita/fast/bench.py measures what remains.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from modules.memory import Memory
from fast.store import MemoryOverlay, MessagePool, upload


# ------------------------------------------------------------------ memory
class FastMemory(Memory):
    """Memory whose pending messages live in a MessagePool and whose rows
    written since the last detach are tracked by a MemoryOverlay. The two
    Parameters (memory, last_update) keep their names, so state_dict,
    checkpoints and serving are unchanged; they simply never carry autograd
    history."""

    def _fast_init_store(self, raw_width: int):
        self._raw_width = raw_width
        self._pool = MessagePool(self.n_nodes, raw_width, self.memory.device)
        self._mo = MemoryOverlay(self.n_nodes, self.memory.device)
        self._lu = self.last_update.detach().cpu().numpy().astype(np.float64).copy()

    def __init_memory__(self):
        super().__init_memory__()
        if hasattr(self, "_raw_width"):
            self._fast_init_store(self._raw_width)

    def ensure_capacity(self, n_nodes):
        super().ensure_capacity(n_nodes)
        if hasattr(self, "_pool"):
            self._pool.ensure_nodes(self.n_nodes)
            self._mo.ensure_nodes(self.n_nodes)
            if len(self._lu) < self.n_nodes:
                self._lu = np.concatenate([self._lu, np.zeros(self.n_nodes - len(self._lu))])

    def detach_memory(self):
        # The tables hold detached values already; only the overlays hold history.
        self._pool.detach()
        self._mo.detach()

    def backup_memory(self):
        self.detach_memory()
        return ("fast", self.memory.data.clone(), self.last_update.data.clone(),
                self._pool.snapshot(), self._lu.copy())

    def restore_memory(self, memory_backup):
        assert memory_backup[0] == "fast", "backup was not taken by the fast path"
        self.memory.data, self.last_update.data = memory_backup[1].clone(), memory_backup[2].clone()
        self._mo.detach()
        self._pool.restore(memory_backup[3])
        self._lu = memory_backup[4].copy()

    def clear_messages(self, nodes):
        self._pool.clear(np.asarray(nodes, dtype=np.int64))

    def store_raw_messages(self, nodes, node_id_to_messages):
        raise RuntimeError("the fast TGN path stores messages itself (bita/fast)")

    # device reads/writes with the overlay
    def fast_read(self, nodes_np, nodes_gpu):
        return self._mo.read(self.memory, nodes_np, nodes_gpu)

    def fast_write(self, nodes_np, nodes_gpu, values):
        self._mo.write(self.memory, nodes_np, nodes_gpu, values)


class _MemView:
    """`memory[nodes, :]` for embedding modules the fast path does not
    re-express (anything but 1-layer graph attention)."""

    def __init__(self, mem: FastMemory):
        self.mem = mem

    def __getitem__(self, key):
        nodes = key[0] if isinstance(key, tuple) else key
        nodes_np = np.asarray(nodes, dtype=np.int64)
        flat = nodes_np.reshape(-1)
        out = self.mem.fast_read(flat, upload(flat, self.mem.memory.device, torch.int64))
        return out.reshape(*nodes_np.shape, out.shape[-1])


class _Batch:
    """Per-batch host arrays, device copies and memory-independent tensors,
    shared by the two compute_temporal_embeddings calls of one batch."""

    def __init__(self, model, src, dst, neg, ts, eidx):
        dev = model.device
        self.B = B = len(src)
        self.src = np.asarray(src, dtype=np.int64)
        self.dst = np.asarray(dst, dtype=np.int64)
        self.neg = np.asarray(neg, dtype=np.int64)
        self.ts = np.asarray(ts, dtype=np.float64)
        self.eidx = np.asarray(eidx, dtype=np.int64)
        self.nodes = np.concatenate([self.src, self.dst, self.neg])
        self.ts3 = np.concatenate([self.ts, self.ts, self.ts])
        ints = upload(np.concatenate([self.nodes, self.eidx]), dev, torch.int64)
        self.nodes_g = ints[: 3 * B]
        self.src_g, self.dst_g = ints[:B], ints[B: 2 * B]
        self.eidx_g = ints[3 * B:]
        self.ef = model.edge_raw_features[self.eidx_g] if model.edge_raw_features is not None \
            else torch.zeros((B, model.n_edge_features), device=dev)
        self.cache = {}


# ------------------------------------------------- stable-shape regions
# Pure functions of their inputs, so level 2 can hand them to torch.compile.
def _attention_core(layer, src_mem, nf_src, nbr_mem, nf_nbr, src_time, ef_n, edge_time_emb, mask, invalid):
    """GraphAttentionEmbedding (1 layer) + TemporalAttentionLayer.forward,
    minus the neighbour lookup: [src_mem+nf ; t] attends over
    [nbr_mem+nf ; edge feats ; t(dt)] with the padding mask already fixed up
    (all-padding rows unmask slot 0) and those rows zeroed afterwards."""
    n, k = mask.shape
    src_feat = src_mem + nf_src
    nbr_emb = (nbr_mem + nf_nbr).view(n, k, -1)
    query = torch.cat([src_feat.unsqueeze(1), src_time], dim=2).permute(1, 0, 2)
    key = torch.cat([nbr_emb, ef_n, edge_time_emb], dim=2).permute(1, 0, 2)
    attn, _ = layer.multi_head_target(query=query, key=key, value=key, key_padding_mask=mask)
    attn = attn.squeeze(0).masked_fill(invalid, 0)
    return layer.merger(attn, src_feat)


def _transformer_core(transformer, grouped, pad):
    return transformer(grouped, src_key_padding_mask=pad)


# -------------------------------------------------------------------- model
class FastTGNMixin:
    """Mixed in front of TGN/ExtendedTGN by enable_fast_tgn()."""

    # ------------------------------------------------------------- BiTA
    def _fast_bita(self, nodes_np):
        """BiTAAggregator.aggregate for the nodes in `nodes_np` with pending
        messages: host-side grouping, one device gather, then the reference's
        own module calls. Returns (to_update, h_bar, node_timestamps) or None.
        """
        agg = self.message_aggregator
        mem: FastMemory = self.memory
        pool = mem._pool
        dev = self.device
        cand = np.unique(nodes_np)
        to_update = cand[pool.cnt[cand] > 0]
        n_nodes = len(to_update)
        if n_nodes == 0:
            return None
        cnts = pool.cnt[to_update]
        R = int(cnts.sum())
        seg0 = np.cumsum(cnts) - cnts
        node_pos = np.repeat(np.arange(n_nodes), cnts)
        ins = np.arange(R) - np.repeat(seg0, cnts)
        rows = np.repeat(pool.start[to_update], cnts) + ins          # insertion order
        peer = pool.row_peer[rows]
        t = pool.row_t[rows]
        # Latest message time per node, over ALL its pending messages.
        node_ts = np.maximum.reduceat(t, seg0)

        # Edges in the reference's order: node ascending (np.unique), then
        # peers in order of first appearance in the node's list (dict order).
        o = np.lexsort((ins, peer, node_pos))
        new_grp = np.ones(R, dtype=bool)
        new_grp[1:] = (node_pos[o][1:] != node_pos[o][:-1]) | (peer[o][1:] != peer[o][:-1])
        gid = np.cumsum(new_grp) - 1
        first_ins = np.empty(R, dtype=np.int64)
        first_ins[o] = ins[o][new_grp][gid]
        # Within an edge: `sorted(seq, key=time)`, which is stable.
        f = np.lexsort((ins, t, first_ins, node_pos))
        np_f, fi_f = node_pos[f], first_ins[f]
        new_e = np.ones(R, dtype=bool)
        new_e[1:] = (np_f[1:] != np_f[:-1]) | (fi_f[1:] != fi_f[:-1])
        eid = np.cumsum(new_e) - 1
        estart = np.flatnonzero(new_e)
        E = len(estart)
        elen = np.diff(np.append(estart, R))
        pos_in_e = np.arange(R) - estart[eid]
        drop = np.maximum(elen - agg.max_seq_len, 0)                  # keep the last max_seq_len
        keep = pos_in_e >= drop[eid]
        klen = elen - drop
        col = (pos_in_e - drop[eid])[keep]
        L = int(klen.max())
        idx = np.zeros((E, L), dtype=np.int64)                        # pool row 0 is all zeros
        idx[eid[keep], col] = rows[f][keep]
        times = np.zeros((E, L), dtype=np.float64)
        times[eid[keep], col] = t[f][keep]
        last_t = times[np.arange(E), klen - 1]
        valid = np.arange(L)[None, :] < klen[:, None]
        dt = ((np.maximum(last_t[:, None] - times, 0.0)) * valid).astype(np.float32)
        owner = np_f[new_e]
        counts = np.bincount(owner, minlength=n_nodes).astype(np.float32)

        lengths = torch.tensor(klen, dtype=torch.long)
        lengths_sorted, sorted_idx = torch.sort(lengths, descending=True)   # as pack_padded_sequence
        unsorted_idx = torch.empty_like(sorted_idx)
        unsorted_idx[sorted_idx] = torch.arange(E)

        ints = upload(np.concatenate([owner, sorted_idx.numpy(), unsorted_idx.numpy(), to_update, klen]),
                      dev, torch.int64)
        owner_g = ints[:E]
        sorted_g, unsorted_g = ints[E: 2 * E], ints[2 * E: 3 * E]
        to_update_g = ints[3 * E: 3 * E + n_nodes]
        lens_g = ints[3 * E + n_nodes:].to(torch.int32)
        f32 = upload(np.concatenate([dt.reshape(-1), counts]), dev)
        dt_g = f32[: E * L].view(E, L)
        counts_g = f32[E * L:]
        f64 = upload(np.concatenate([last_t, node_ts]), dev)
        last_t_g = f64[:E]
        node_ts_g = f64[E:]

        raw = pool.gather(idx)                                        # [E, L, raw]
        m = self.message_function.compute_message(raw)
        if m.shape[-1] != agg.message_dim:
            raise ValueError(f"BiTA expects {agg.message_dim}-D messages, got {m.shape[-1]}.")
        x = m + agg.time_encoder(dt_g)
        if self._fast_level >= 2:
            from fast.triton_gru import bigru_final_states
            h_n = bigru_final_states(agg.bigru, x, lens_g)
        else:
            x = x.index_select(0, sorted_g)
            data, batch_sizes = torch._VF._pack_padded_sequence(x, lengths_sorted, True)
            _, h_n = agg.bigru(PackedSequence(data, batch_sizes, sorted_g, unsorted_g))
        z_temp = torch.cat([h_n[0], h_n[1]], dim=-1)
        e = agg.W_e(z_temp)
        order = torch.argsort(last_t_g)
        C = agg.context_size
        n_groups = (E + C - 1) // C
        grouped = torch.zeros(n_groups * C, agg.d_trans, device=dev, dtype=e.dtype)
        grouped[:E] = e[order]
        pad_np = np.ones(n_groups * C, dtype=bool)
        pad_np[:E] = False
        pad = upload(pad_np, dev)
        zc = self._fast_fn("transformer", _transformer_core)(
            agg.transformer, grouped.view(n_groups, C, agg.d_trans), pad.view(n_groups, C))
        z_ctx = torch.empty_like(e)
        z_ctx[order] = zc.reshape(n_groups * C, agg.d_trans)[:E]
        summed = torch.zeros(n_nodes, agg.d_trans, device=dev, dtype=z_ctx.dtype).index_add_(0, owner_g, z_ctx)
        h_bar = summed / counts_g.unsqueeze(1)
        return to_update, to_update_g, h_bar, node_ts, node_ts_g

    def _fast_update_memory(self, nodes_np):
        mem: FastMemory = self.memory
        res = self._fast_bita(nodes_np)
        if res is None:
            return
        to_update, to_update_g, h_bar, node_ts, node_ts_g = res
        assert (mem._lu[to_update] <= node_ts).all(), "Trying to update memory to time in the past"
        rows = mem.fast_read(to_update, to_update_g)
        mem.last_update[to_update_g] = node_ts_g
        mem._lu[to_update] = node_ts
        updated = self.memory_updater.memory_updater(h_bar, rows)
        mem.fast_write(to_update, to_update_g, updated)

    # --------------------------------------------------------- messages
    def _fast_raw_messages(self, bt: _Batch, owner_np, owner_g, peer_np, peer_g):
        mem: FastMemory = self.memory
        owner_mem = mem.fast_read(owner_np, owner_g)
        peer_mem = mem.fast_read(peer_np, peer_g)
        lu = mem._lu[owner_np]
        delta = np.where(lu > 0, bt.ts - lu, 0.0).astype(np.float32)
        te = self.time_encoder(upload(delta, self.device).unsqueeze(1)).view(bt.B, -1)
        return torch.cat([owner_mem, peer_mem, bt.ef, te], dim=1)

    # -------------------------------------------------------- embedding
    def _fast_embedding(self, bt: _Batch, n_neighbors):
        em = self.embedding_module
        mem: FastMemory = self.memory
        if not self._fast_ga1:
            return em.compute_embedding(memory=_MemView(mem), source_nodes=bt.nodes, timestamps=bt.ts3,
                                        n_layers=self.n_layers, n_neighbors=n_neighbors, time_diffs=None)
        dev = self.device
        c = bt.cache
        te_frozen = not any(p.requires_grad for p in em.time_encoder.parameters())
        if "nbr" not in c or c["n_neighbors"] != n_neighbors:
            neighbors, edge_idxs, edge_times = em.neighbor_finder.get_temporal_neighbor(
                bt.nodes, bt.ts3, n_neighbors=n_neighbors)
            deltas = (bt.ts3[:, None] - edge_times).astype(np.float32)
            mask = neighbors == 0
            invalid = mask.all(axis=1, keepdims=True)
            mask_fixed = mask.copy()
            mask_fixed[invalid[:, 0], 0] = False
            nbr = neighbors.reshape(-1).astype(np.int64)
            ints = upload(np.concatenate([nbr, edge_idxs.reshape(-1).astype(np.int64)]), dev, torch.int64)
            bools = upload(np.concatenate([mask_fixed.reshape(-1), invalid[:, 0]]), dev)
            n = len(bt.nodes)
            c.update(n_neighbors=n_neighbors, nbr=nbr, nbr_g=ints[: nbr.size],
                     eidx_n_g=ints[nbr.size:].view(n, -1),
                     deltas_g=upload(deltas, dev),
                     mask_g=bools[: mask.size].view(n, -1), invalid_g=bools[mask.size:].view(n, 1))
            c["nf_src"] = em.node_features[bt.nodes_g, :] if em.node_features is not None \
                else torch.zeros((n, self.n_node_features), device=dev)
            c["nf_nbr"] = em.node_features[c["nbr_g"], :] if em.node_features is not None \
                else torch.zeros((nbr.size, self.n_node_features), device=dev)
            c["ef_n"] = em.edge_features[c["eidx_n_g"], :] if em.edge_features is not None \
                else torch.zeros((n, n_neighbors, self.n_edge_features), device=dev)
        n = len(bt.nodes)
        if te_frozen and "src_time" in c:
            src_time, edge_time_emb = c["src_time"], c["edge_time_emb"]
        else:
            src_time = em.time_encoder(torch.zeros((n, 1), device=dev))
            edge_time_emb = em.time_encoder(c["deltas_g"])
            if te_frozen:
                c["src_time"], c["edge_time_emb"] = src_time, edge_time_emb

        return self._fast_fn("attention", _attention_core)(
            em.attention_models[0], mem.fast_read(bt.nodes, bt.nodes_g), c["nf_src"],
            mem.fast_read(c["nbr"], c["nbr_g"]), c["nf_nbr"], src_time, c["ef_n"], edge_time_emb,
            c["mask_g"], c["invalid_g"])

    def _fast_fn(self, name, fn):
        """`fn`, or with CYBERWORLD_FAST_TGN_COMPILE=1 a torch.compile'd copy.

        Measured on CTU-13 scenario 7 it is SLOWER (45 vs 62 batch/s eager;
        6.6 with mode=reduce-overhead, whose CUDA graphs re-record as shapes
        change), so it is off by default and kept only for experiments."""
        import os
        if os.environ.get("CYBERWORLD_FAST_TGN_COMPILE", "") not in ("1", "true", "True"):
            return fn
        cache = self.__dict__.setdefault("_fast_compiled", {})
        if name not in cache:
            import torch._inductor.config as inductor_config
            # Random ops (dropout) lowered to the same eager aten calls, so a
            # compiled region draws exactly the masks the reference draws.
            inductor_config.fallback_random = True
            mode = os.environ.get("CYBERWORLD_FAST_TGN_COMPILE_MODE") or None
            cache[name] = torch.compile(fn, dynamic=False, mode=mode)
        return cache[name]

    # ------------------------------------------------------------- steps
    def _fast_cte(self, bt: _Batch, n_neighbors):
        B = bt.B
        self._fast_update_memory(np.concatenate([bt.src, bt.dst]))
        self.memory.clear_messages(np.concatenate([bt.src, bt.dst]))
        src_msg = self._fast_raw_messages(bt, bt.src, bt.src_g, bt.dst, bt.dst_g)
        dst_msg = self._fast_raw_messages(bt, bt.dst, bt.dst_g, bt.src, bt.src_g)
        self.memory._pool.write(np.concatenate([bt.src, bt.dst]), np.concatenate([bt.dst, bt.src]),
                                np.concatenate([bt.ts, bt.ts]), torch.cat([src_msg, dst_msg], dim=0))
        emb = self._fast_embedding(bt, n_neighbors)
        return emb[:B], emb[B: 2 * B], emb[2 * B:]

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes,
                                    edge_times, edge_idxs, n_neighbors=20):
        bt = _Batch(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs)
        return self._fast_cte(bt, n_neighbors)

    def _fast_scores(self, s, d, n):
        B = s.shape[0]
        score = self.affinity_score(torch.cat([s, s], dim=0), torch.cat([d, n], dim=0)).squeeze(dim=-1)
        return score[:B].sigmoid(), score[B:].sigmoid()

    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes,
                                   edge_times, edge_idxs, n_neighbors=20):
        bt = _Batch(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs)
        return self._fast_scores(*self._fast_cte(bt, n_neighbors))

    def compute_edge_probabilities_and_categories(self, source_nodes, destination_nodes, negative_nodes,
                                                  edge_times, edge_idxs, n_neighbors=20):
        # The reference runs compute_temporal_embeddings twice per batch (once
        # inside compute_edge_probabilities, once for the category head), and
        # the second call consumes the messages the first one stored. Both
        # calls are kept; what they share is only what does not depend on
        # memory (uploads, neighbours, feature gathers).
        bt = _Batch(self, source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs)
        pos, neg = self._fast_scores(*self._fast_cte(bt, n_neighbors))
        if getattr(self.embedding_module.neighbor_finder, "uniform", False):
            bt.cache = {}       # uniform sampling draws again, as the reference does
        s2, d2, _ = self._fast_cte(bt, n_neighbors)
        combined = torch.cat([s2, d2, bt.ef.float()], dim=1)
        return pos, neg, self.category_predictor(combined)

    # ---------------------------------------------- streaming (serving)
    @torch.no_grad()
    def update_memory_for(self, node_ids) -> None:
        if len(node_ids) == 0:
            return
        nodes = np.unique(np.asarray(node_ids, dtype=np.int64))
        self._fast_update_memory(nodes)
        self.memory.clear_messages(nodes)

    @torch.no_grad()
    def store_interactions(self, sources, destinations, timestamps, edge_idxs) -> None:
        if len(sources) == 0:
            return
        bt = _Batch(self, sources, destinations, destinations, timestamps, edge_idxs)
        src_msg = self._fast_raw_messages(bt, bt.src, bt.src_g, bt.dst, bt.dst_g)
        dst_msg = self._fast_raw_messages(bt, bt.dst, bt.dst_g, bt.src, bt.src_g)
        # store_raw_messages EXTENDS lists; the pool replaces them. Streaming
        # always updates before it stores, so the lists are empty here.
        both = np.concatenate([bt.src, bt.dst])
        assert (self.memory._pool.cnt[both] == 0).all(), "fast path: store before update"
        self.memory._pool.write(both, np.concatenate([bt.dst, bt.src]),
                                np.concatenate([bt.ts, bt.ts]), torch.cat([src_msg, dst_msg], dim=0))


def _check_supported(tgn):
    from modules.message_aggregator import BiTAAggregator
    from modules.memory_updater import SequenceMemoryUpdater
    problems = []
    if not tgn.use_memory:
        problems.append("memory is off (the fast path is the memory path)")
    else:
        if not tgn.memory_update_at_start:
            problems.append("--memory_update_at_end")
        if not isinstance(tgn.message_aggregator, BiTAAggregator):
            problems.append(f"aggregator {type(tgn.message_aggregator).__name__} (only BiTA)")
        if not isinstance(tgn.memory_updater, SequenceMemoryUpdater):
            problems.append(f"memory updater {type(tgn.memory_updater).__name__}")
        if any(len(v) for v in tgn.memory.messages.values()):
            problems.append("pending messages already stored in the reference format")
    if tgn.use_destination_embedding_in_message or tgn.use_source_embedding_in_message or tgn.dyrep:
        problems.append("embedding-in-message / dyrep")
    return problems


def enable(tgn, level: int = 2):
    from modules.embedding_module import GraphAttentionEmbedding
    problems = _check_supported(tgn)
    if problems:
        raise ValueError("fast TGN step does not support: " + "; ".join(problems))
    cls = type(tgn)
    if not isinstance(tgn, FastTGNMixin):
        tgn.__class__ = type("Fast" + cls.__name__, (FastTGNMixin, cls), {})
    tgn._fast_level = int(level)
    tgn._fast_ga1 = isinstance(tgn.embedding_module, GraphAttentionEmbedding) and tgn.n_layers == 1
    if not isinstance(tgn.memory, FastMemory):
        tgn.memory.__class__ = FastMemory
    raw_width = 2 * tgn.memory_dimension + tgn.n_edge_features + tgn.time_encoder.dimension
    tgn.memory._fast_init_store(raw_width)
    return tgn
