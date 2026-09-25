"""Bidirectional single-layer GRU over variable-length sequences, final
states only -- what BiTAAggregator asks nn.GRU + pack_padded_sequence for --
as two Triton kernels (forward recurrence, backward recurrence).

Why: BiTA's per-call workload is ~180 sequences, most of length 1, max ~9
(64 at most). cuDNN's packed RNN runs it as a per-timestep launch sequence
and spent ~1.9 ms of host time per backward call, the largest single cost of
the fast step. Here each program owns a block of sequences in one direction
and loops over time with the hidden state in registers.

Math is PyTorch's GRU (gate order r, z, n):
    r = s(Wir x + bir + Whr h + bhr)     z = s(Wiz x + biz + Whz h + bhz)
    n = tanh(Win x + bin + r * (Whn h + bhn))       h' = (1 - z) n + z h
The input projections for all timesteps are one GEMM outside the kernel
(autograd handles it); the kernels take gi = x W_ih^T + b_ih. Weight
gradients of W_hh are one GEMM per direction from the saved states. All dots
run in IEEE fp32 (no TF32). Numerics differ from cuDNN at fp32 rounding level
(tests/test_fast_tgn_equivalence.py and bench.py report the magnitudes).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gru_fwd_kernel(gi_ptr, whh_ptr, bhh_ptr, len_ptr, hn_ptr, save_ptr,
                    E, L, H: tl.constexpr, HP: tl.constexpr, BS: tl.constexpr):
    pid = tl.program_id(0)
    d = tl.program_id(1)
    seqs = pid * BS + tl.arange(0, BS)
    smask = seqs < E
    lens = tl.load(len_ptr + seqs, mask=smask, other=0)
    j = tl.arange(0, HP)
    jm = j < H
    k = tl.arange(0, HP)
    km = k < H
    G3: tl.constexpr = 3 * H
    # W^T blocks: wt[k, j] = W[d, gate*H + j, k]
    wbase = whh_ptr + d * G3 * H
    wmask = km[:, None] & jm[None, :]
    wr = tl.load(wbase + (0 * H + j[None, :]) * H + k[:, None], mask=wmask, other=0.0)
    wz = tl.load(wbase + (1 * H + j[None, :]) * H + k[:, None], mask=wmask, other=0.0)
    wn = tl.load(wbase + (2 * H + j[None, :]) * H + k[:, None], mask=wmask, other=0.0)
    bbase = bhh_ptr + d * G3
    br = tl.load(bbase + 0 * H + j, mask=jm, other=0.0)
    bz = tl.load(bbase + 1 * H + j, mask=jm, other=0.0)
    bn = tl.load(bbase + 2 * H + j, mask=jm, other=0.0)
    h = tl.zeros((BS, HP), dtype=tl.float32)
    maxlen = tl.max(lens, axis=0)
    m2 = smask[:, None] & jm[None, :]
    for t in range(0, maxlen):
        active = t < lens
        pos = tl.where(d == 0, t, lens - 1 - t)
        am = m2 & active[:, None]
        g = gi_ptr + ((seqs * L + pos) * 2 + d)[:, None] * G3 + j[None, :]
        gir = tl.load(g + 0 * H, mask=am, other=0.0)
        giz = tl.load(g + 1 * H, mask=am, other=0.0)
        gin = tl.load(g + 2 * H, mask=am, other=0.0)
        ghr = tl.dot(h, wr, input_precision="ieee") + br[None, :]
        ghz = tl.dot(h, wz, input_precision="ieee") + bz[None, :]
        ghn = tl.dot(h, wn, input_precision="ieee") + bn[None, :]
        r = tl.sigmoid(gir + ghr)
        z = tl.sigmoid(giz + ghz)
        n = libdevice.tanh(gin + r * ghn)
        hnew = (1.0 - z) * n + z * h
        # saved per (direction, sequence, step-in-processing-order): h_prev, r, z, n, ghn
        s = save_ptr + (((d * E + seqs) * L + t) * 5)[:, None] * HP + j[None, :]
        tl.store(s + 0 * HP, h, mask=am)
        tl.store(s + 1 * HP, r, mask=am)
        tl.store(s + 2 * HP, z, mask=am)
        tl.store(s + 3 * HP, n, mask=am)
        tl.store(s + 4 * HP, ghn, mask=am)
        h = tl.where(active[:, None], hnew, h)
    tl.store(hn_ptr + (d * E + seqs)[:, None] * H + j[None, :], h, mask=m2)


@triton.jit
def _gru_bwd_kernel(dhn_ptr, whh_ptr, len_ptr, save_ptr, dgi_ptr, dgh_ptr,
                    E, L, H: tl.constexpr, HP: tl.constexpr, BS: tl.constexpr):
    pid = tl.program_id(0)
    d = tl.program_id(1)
    seqs = pid * BS + tl.arange(0, BS)
    smask = seqs < E
    lens = tl.load(len_ptr + seqs, mask=smask, other=0)
    j = tl.arange(0, HP)
    jm = j < H
    k = tl.arange(0, HP)
    km = k < H
    G3: tl.constexpr = 3 * H
    # W blocks as [j, k] = W[d, gate*H + j, k]  (dh_prev[b,k] = sum_j dg[b,j] W[j,k])
    wbase = whh_ptr + d * G3 * H
    wmask = jm[:, None] & km[None, :]
    wr = tl.load(wbase + (0 * H + j[:, None]) * H + k[None, :], mask=wmask, other=0.0)
    wz = tl.load(wbase + (1 * H + j[:, None]) * H + k[None, :], mask=wmask, other=0.0)
    wn = tl.load(wbase + (2 * H + j[:, None]) * H + k[None, :], mask=wmask, other=0.0)
    m2 = smask[:, None] & jm[None, :]
    dh = tl.load(dhn_ptr + (d * E + seqs)[:, None] * H + j[None, :], mask=m2, other=0.0)
    maxlen = tl.max(lens, axis=0)
    for tt in range(0, maxlen):
        t = maxlen - 1 - tt
        active = t < lens
        am = m2 & active[:, None]
        s = save_ptr + (((d * E + seqs) * L + t) * 5)[:, None] * HP + j[None, :]
        hp = tl.load(s + 0 * HP, mask=am, other=0.0)
        r = tl.load(s + 1 * HP, mask=am, other=0.0)
        z = tl.load(s + 2 * HP, mask=am, other=0.0)
        n = tl.load(s + 3 * HP, mask=am, other=0.0)
        ghn = tl.load(s + 4 * HP, mask=am, other=0.0)
        dn = dh * (1.0 - z)
        dz = dh * (hp - n)
        dnp = dn * (1.0 - n * n)
        dghn = dnp * r
        dr = dnp * ghn
        drp = dr * r * (1.0 - r)
        dzp = dz * z * (1.0 - z)
        drp = tl.where(am, drp, 0.0)
        dzp = tl.where(am, dzp, 0.0)
        dnp = tl.where(am, dnp, 0.0)
        dghn = tl.where(am, dghn, 0.0)
        pos = tl.where(d == 0, t, lens - 1 - t)
        g = dgi_ptr + ((seqs * L + pos) * 2 + d)[:, None] * G3 + j[None, :]
        tl.store(g + 0 * H, drp, mask=am)
        tl.store(g + 1 * H, dzp, mask=am)
        tl.store(g + 2 * H, dnp, mask=am)
        q = dgh_ptr + ((d * E + seqs) * L + t)[:, None] * G3 + j[None, :]
        tl.store(q + 0 * H, drp, mask=am)
        tl.store(q + 1 * H, dzp, mask=am)
        tl.store(q + 2 * H, dghn, mask=am)
        dprev = dh * z + tl.dot(drp, wr, input_precision="ieee") \
            + tl.dot(dzp, wz, input_precision="ieee") + tl.dot(dghn, wn, input_precision="ieee")
        dh = tl.where(active[:, None], dprev, dh)


_BS = 16


class _BiGRURecurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gi, whh, bhh, lens):
        E, L = gi.shape[0], gi.shape[1]
        H = whh.shape[-1]
        HP = max(16, triton.next_power_of_2(H))
        gi = gi.contiguous()
        whh_c, bhh_c = whh.contiguous(), bhh.contiguous()
        hn = torch.empty(2, E, H, device=gi.device, dtype=torch.float32)
        save = torch.zeros(2, E, L, 5, HP, device=gi.device, dtype=torch.float32)
        grid = (triton.cdiv(E, _BS), 2)
        _gru_fwd_kernel[grid](gi, whh_c, bhh_c, lens, hn, save, E, L, H=H, HP=HP, BS=_BS, num_warps=4, num_stages=1)
        ctx.save_for_backward(whh_c, lens, save)
        ctx.dims = (E, L, H, HP)
        return hn

    @staticmethod
    def backward(ctx, dhn):
        whh, lens, save = ctx.saved_tensors
        E, L, H, HP = ctx.dims
        dgi = torch.zeros(E, L, 2, 3 * H, device=dhn.device, dtype=torch.float32)
        dgh = torch.zeros(2, E, L, 3 * H, device=dhn.device, dtype=torch.float32)
        grid = (triton.cdiv(E, _BS), 2)
        _gru_bwd_kernel[grid](dhn.contiguous(), whh, lens, save, dgi, dgh, E, L, H=H, HP=HP, BS=_BS, num_warps=4, num_stages=1)
        hprev = save[:, :, :, 0, :H].reshape(2, E * L, H)
        dgh2 = dgh.view(2, E * L, 3 * H)
        dwhh = torch.bmm(dgh2.transpose(1, 2), hprev)          # [2, 3H, H]
        dbhh = dgh2.sum(1)                                      # [2, 3H]
        return dgi, dwhh, dbhh, None


def bigru_final_states(gru: torch.nn.GRU, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """h_n [2, E, H] of `gru` (1 layer, bidirectional, batch_first, no dropout)
    over x [E, L, in] with per-sequence lengths `lens` (device int32), in the
    ORIGINAL sequence order -- what nn.GRU returns for an unsorted packed input."""
    w_ih = torch.cat([gru.weight_ih_l0, gru.weight_ih_l0_reverse], 0)
    b_ih = torch.cat([gru.bias_ih_l0, gru.bias_ih_l0_reverse], 0)
    gi = torch.nn.functional.linear(x, w_ih, b_ih)             # [E, L, 2*3H] == [E, L, 2, 3H]
    whh = torch.stack([gru.weight_hh_l0, gru.weight_hh_l0_reverse], 0)
    bhh = torch.stack([gru.bias_hh_l0, gru.bias_hh_l0_reverse], 0)
    E, L = x.shape[0], x.shape[1]
    return _BiGRURecurrence.apply(gi.view(E, L, 2, -1), whh, bhh, lens)
