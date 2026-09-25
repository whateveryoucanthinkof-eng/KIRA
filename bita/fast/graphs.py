"""CUDA-graphed regions of the fast TGN step (fast path level 3).

A region is a pure function of fixed-shape tensors and module parameters
(bita/fast/fast_tgn.py: the graph-attention embedding, both heads and both
losses of one 128-edge batch). It is captured ONCE per slot -- forward and
backward, as torch.cuda.make_graphed_callables does -- and then replayed:
~300 kernel launches and their autograd bookkeeping become two replays.

Why a slot per batch of an accumulation group: backprop_every batches run
their forwards before ONE backward, and a replay overwrites the graph's saved
activations. Slot j serves the j-th batch since the last detach_memory(), so
no slot is replayed twice before the backward that consumes it. Each slot has
its own memory pool, so no replay can write into another slot's live
activations whatever order autograd runs the backward in.

Why the results equal the eager (level 2) step bit for bit:
  * capture records the same kernels eager would launch, on the same shapes;
  * dropout: a captured philox op reads (seed, offset) from device memory that
    graph.replay() fills with the generator's CURRENT state, and replay then
    advances the generator by the graph's total increment -- exactly the
    offsets the same ops draw eagerly, in the same order;
  * warmup iterations draw random numbers, so the CUDA RNG state is saved
    before capture and restored after it: capturing consumes nothing;
  * gradients: the region's parameter gradients reach AccumulateGrad through
    the autograd Function, one per batch, in the order the eager subgraphs
    would deliver them (each parameter is used once per region), so the
    8-batch sums associate identically.
bench.py --compare and tests/test_fast_tgn_graphs.py measure it.
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import torch


class GraphSlot:
    """One captured forward + backward of `fn`.

    fn(*inputs) returns a tuple of tensors. `diff_inputs` are the indices of
    inputs that receive a gradient (their static copies are leaves with
    requires_grad), `diff_outputs` the indices of outputs a gradient flows
    into; the other outputs are returned non-differentiable.
    """

    def __init__(self, fn: Callable, example_inputs: Sequence[torch.Tensor], params: List[torch.nn.Parameter],
                 diff_inputs: Sequence[int], diff_outputs: Sequence[int], warmup: int = 2):
        self.fn = fn
        self.diff_inputs = list(diff_inputs)
        self.diff_outputs = list(diff_outputs)
        self.static_inputs = []
        for i, x in enumerate(example_inputs):
            s = x.detach().clone()
            if i in self.diff_inputs:
                s.requires_grad_(True)
            self.static_inputs.append(s)
        self.pool = torch.cuda.graph_pool_handle()
        dev = self.static_inputs[0].device
        rng = torch.cuda.get_rng_state(dev)
        try:
            self._capture(params, warmup)
        finally:
            torch.cuda.set_rng_state(rng, dev)       # capture draws nothing from the stream

    def _grad_targets(self, params):
        return [self.static_inputs[i] for i in self.diff_inputs] + list(params)

    def _capture(self, params, warmup):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):                  # lazy init, autotuning, cuBLAS workspaces
                outs = self.fn(*self.static_inputs)
                grads = torch.autograd.grad([outs[i] for i in self.diff_outputs], self._grad_targets(params),
                                            [torch.ones_like(outs[i]) for i in self.diff_outputs],
                                            allow_unused=True)
            del outs
        torch.cuda.current_stream().wait_stream(side)
        # Parameters the region does not reach get no gradient from it.
        self.params = [p for p, g in zip(params, grads[len(self.diff_inputs):]) if g is not None]
        del grads

        self.fwd = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.fwd, pool=self.pool):
            outs = self.fn(*self.static_inputs)
        self.static_outputs = tuple(outs)
        self.static_grad_outputs = [torch.empty_like(self.static_outputs[i]) for i in self.diff_outputs]
        self.bwd = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.bwd, pool=self.pool):
            grads = torch.autograd.grad([self.static_outputs[i] for i in self.diff_outputs],
                                        self._grad_targets(self.params), self.static_grad_outputs,
                                        allow_unused=True)
        n = len(self.diff_inputs)
        self.static_grad_inputs = grads[:n]
        self.static_grad_params = grads[n:]
        self._held = ()

    def __call__(self, *inputs):
        """Non-differentiable inputs are copied into the static buffers here
        (pass None for one already written in place); differentiable ones are
        copied inside the autograd Function."""
        diff_live = []
        for i, x in enumerate(inputs):
            if i in self.diff_inputs:
                diff_live.append(x)
            elif x is not None:
                self.static_inputs[i].copy_(x)
        return _SlotFunction.apply(self, *diff_live, *self.params)


class _SlotFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, slot: GraphSlot, *args):
        ctx.slot = slot
        for i, x in zip(slot.diff_inputs, args):
            slot.static_inputs[i].copy_(x)
        slot.fwd.replay()
        outs = tuple(o.detach() for o in slot.static_outputs)
        ctx.mark_non_differentiable(*[o for k, o in enumerate(outs) if k not in slot.diff_outputs])
        return outs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grads):
        slot: GraphSlot = ctx.slot
        for buf, k in zip(slot.static_grad_outputs, slot.diff_outputs):
            g = grads[k]
            if g is None:
                buf.zero_()
            else:
                buf.copy_(g)
        slot.bwd.replay()
        n = len(slot.diff_inputs)
        gin = tuple(g.detach() if (g is not None and ctx.needs_input_grad[1 + i]) else None
                    for i, g in enumerate(slot.static_grad_inputs))
        gpar = tuple(g.detach() for g in slot.static_grad_params)
        # Holding a second reference stops AccumulateGrad from adopting the
        # static buffer itself as p.grad (it clones instead), so the next
        # replay of this slot can never write into a live .grad.
        slot._held = gpar
        assert len(gin) == n
        return (None,) + gin + gpar
