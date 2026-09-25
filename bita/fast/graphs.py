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

import contextlib
from typing import Callable, List, Sequence

import torch


_WARMUP_STREAMS = {}


def _warmup_stream(device) -> torch.cuda.Stream:
    """ONE side stream for every warmup: cuBLAS keeps a workspace per
    (handle, stream) for the life of the process, so a new stream per
    capture cost ~32-64 MB of GPU memory each."""
    key = torch.device(device).index
    if key not in _WARMUP_STREAMS:
        _WARMUP_STREAMS[key] = torch.cuda.Stream(device)
    return _WARMUP_STREAMS[key]


@contextlib.contextmanager
def _capturing(graph: torch.cuda.CUDAGraph, pool, stream: torch.cuda.Stream):
    """torch.cuda.graph(graph, pool, stream) WITHOUT its torch.cuda.empty_cache().

    Slots are captured lazily, in the middle of an accumulation group, while
    other slots' graphs are live. Measured: once graphs of ANOTHER model had
    been destroyed in the same process (tests do this; one training process
    never does), an empty_cache() while this model's graphs lived made a
    later backward replay hit an illegal memory access
    (tests/test_fast_tgn_graphs.py as one file: 5/5 failures with
    torch.cuda.graph, 4/4 passes with this; an explicit empty_cache() before
    every replay fails the same way unless the dead graphs were collected
    before the new ones were captured). So captures here never empty the
    cache, and fast_tgn never destroys a model's graphs while it trains.
    """
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        graph.capture_begin(pool=pool)
        try:
            yield
        finally:
            graph.capture_end()


class GraphSlot:
    """One captured forward + backward of `fn`.

    fn(*inputs) returns a tuple of tensors. `diff_inputs` are the indices of
    inputs that receive a gradient (their static copies are leaves with
    requires_grad), `diff_outputs` the indices of outputs a gradient flows
    into; the other outputs are returned non-differentiable.
    """

    def __init__(self, fn: Callable, example_inputs: Sequence[torch.Tensor], params: List[torch.nn.Parameter],
                 diff_inputs: Sequence[int], diff_outputs: Sequence[int], module: torch.nn.Module,
                 warmup: int = 2, pool=None, watch: Sequence[torch.Tensor] = ()):
        self.fn = fn
        self.diff_inputs = list(diff_inputs)
        self.diff_outputs = list(diff_outputs)
        self.static_inputs = []
        for i, x in enumerate(example_inputs):
            s = x.detach().clone()
            if i in self.diff_inputs:
                s.requires_grad_(True)
            self.static_inputs.append(s)
        # A pool may be shared only by graphs that are never live at the same
        # time (the variants of one slot); see fast_tgn._graph_slot.
        self.pool = pool if pool is not None else torch.cuda.graph_pool_handle()
        dev = self.static_inputs[0].device
        rng = torch.cuda.get_rng_state(dev)
        try:
            self._capture(module, params, warmup)
        finally:
            torch.cuda.set_rng_state(rng, dev)       # capture draws nothing from the stream
        # The graphs read these tensors at their capture-time addresses.
        self._watch = list(self.params) + [t for t in watch if torch.is_tensor(t)]
        self._ptrs = [t.data_ptr() for t in self._watch]

    def stale(self) -> bool:
        """True if a parameter or watched tensor was re-allocated since capture
        (e.g. `p.data = ...`); in-place updates (the optimizer,
        load_state_dict) keep the address and need nothing."""
        return any(t.data_ptr() != q for t, q in zip(self._watch, self._ptrs))

    def _capture(self, module, params, warmup):
        """Warmup and capture run against SHADOW parameters: leaves that alias
        each parameter's storage (so a replay reads the current weights) but
        are not the parameters. The real parameters' AccumulateGrad nodes are
        therefore only ever created and run by the training backward, on the
        training stream; capture never touches them (torch refuses a capture
        that would, and a node first made on the side stream would make the
        training backward accumulate .grad on that stream)."""
        from torch.nn.utils.stateless import _reparametrize_module
        names = {id(p): n for n, p in module.named_parameters()}
        shadows = [p.detach().requires_grad_(True) for p in params]
        swap = {names[id(p)]: q for p, q in zip(params, shadows)}
        side = _warmup_stream(self.static_inputs[0].device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), _reparametrize_module(module, swap):
            for _ in range(warmup):                  # lazy init, autotuning, cuBLAS workspaces
                outs = self.fn(*self.static_inputs)
                grads = torch.autograd.grad([outs[i] for i in self.diff_outputs], self._grad_targets(shadows),
                                            [torch.ones_like(outs[i]) for i in self.diff_outputs],
                                            allow_unused=True)
            del outs
            # Parameters the region does not reach get no gradient from it.
            used = [g is not None for g in grads[len(self.diff_inputs):]]
            del grads
        torch.cuda.current_stream().wait_stream(side)
        self.params = [p for p, u in zip(params, used) if u]
        shadows = [q for q, u in zip(shadows, used) if u]

        self.fwd = torch.cuda.CUDAGraph()
        self.bwd = torch.cuda.CUDAGraph()
        with _reparametrize_module(module, swap):
            with _capturing(self.fwd, self.pool, side):
                outs = self.fn(*self.static_inputs)
            self.static_outputs = tuple(outs)
            self.static_grad_outputs = [torch.empty_like(self.static_outputs[i]) for i in self.diff_outputs]
            with _capturing(self.bwd, self.pool, side):
                grads = torch.autograd.grad([self.static_outputs[i] for i in self.diff_outputs],
                                            self._grad_targets(shadows), self.static_grad_outputs,
                                            allow_unused=True)
        n = len(self.diff_inputs)
        self.static_grad_inputs = grads[:n]
        self.static_grad_params = grads[n:]
        assert all(g is not None for g in self.static_grad_params)
        self._held = ()

    def _grad_targets(self, params):
        return [self.static_inputs[i] for i in self.diff_inputs] + list(params)

    def __call__(self, *inputs):
        """Non-differentiable inputs are copied into the static buffers here
        (pass None for one already written in place); differentiable ones are
        copied inside the autograd Function. An input with fewer rows than its
        static buffer fills the leading rows (the caller's padding contract
        says what the remaining rows may hold); its gradient is those rows'."""
        diff_live = []
        for i, x in enumerate(inputs):
            if i in self.diff_inputs:
                diff_live.append(x)
            elif x is not None:
                _copy_in(self.static_inputs[i], x)
        return _SlotFunction.apply(self, *diff_live, *self.params)


def _copy_in(static: torch.Tensor, x: torch.Tensor) -> None:
    if x.shape[0] == static.shape[0]:
        static.copy_(x)
    else:
        static[: x.shape[0]].copy_(x)


class _SlotFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, slot: GraphSlot, *args):
        ctx.slot = slot
        ctx.rows = []
        for i, x in zip(slot.diff_inputs, args):
            _copy_in(slot.static_inputs[i], x)
            ctx.rows.append(x.shape[0])
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
        gin = tuple(g.detach()[: ctx.rows[i]] if (g is not None and ctx.needs_input_grad[1 + i]) else None
                    for i, g in enumerate(slot.static_grad_inputs))
        gpar = tuple(g.detach() for g in slot.static_grad_params)
        # Holding a second reference stops AccumulateGrad from adopting the
        # static buffer itself as p.grad (it clones instead), so the next
        # replay of this slot can never write into a live .grad.
        slot._held = gpar
        assert len(gin) == n
        return (None,) + gin + gpar
