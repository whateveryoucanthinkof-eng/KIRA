"""Replay a training step's forward + backward as CUDA graphs.

## Why

Branch A, Branch B and DeepOP are small models: a training step is ~1 ms of
GPU work issued as a few hundred tiny kernels, and launching them from Python
cost ~3.4 ms per step (measured, Branch A). The GPU sat ~20% busy at idle
clocks while one CPU thread was pegged. A CUDA graph records the launches
once and replays them with one call.

## What is graphed, and why it is bit-identical

`loss = fn(*inputs)` and its backward, via torch.cuda.make_graphed_callables:
the same kernels, on the same inputs, in the same order. The optimizer step,
clipping and the guard stay eager (Adam's bias correction is computed on the
host from a Python step count, which a graph would freeze).

Two things would otherwise make it differ, and both are handled:

  * make_graphed_callables runs warm-up iterations, which consume the CUDA
    (dropout) and CPU RNG streams. Both states are saved before and restored
    after, so the first real step draws exactly the masks eager would.
  * A graph has fixed shapes. A batch of any other shape (the last, partial
    batch of an epoch) runs the eager callable -- the same function.

tests/test_graphed_step.py checks losses and every weight against eager over
hundreds of steps with dropout live; the trainers' end-to-end runs are
compared leaf by leaf (see PERF_REPORT.md).

Training mode is captured; call the eager module for evaluation.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

import torch


def graphs_enabled(default: bool = True) -> bool:
    """CYBERWORLD_CUDA_GRAPH=0 turns the graphed steps off everywhere."""
    v = os.environ.get("CYBERWORLD_CUDA_GRAPH")
    if v is None:
        return default
    return v not in ("0", "false", "False", "")


class _Wrap(torch.nn.Module):
    def __init__(self, fn: Callable, modules: Sequence[torch.nn.Module]):
        super().__init__()
        self.mods = torch.nn.ModuleList(modules)    # so make_graphed_callables sees the parameters
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)


class GraphedLoss:
    """`GraphedLoss(fn, modules, example)(*inputs) == fn(*inputs)`, replayed as a graph.

    `fn` returns a scalar loss (or a tuple of tensors) from the tensors it is
    given and the parameters of `modules`. `example` fixes the graphed shapes
    and dtypes; inputs of any other shape fall back to calling `fn` eagerly.
    Built lazily on the first matching call, so it costs nothing when unused.
    """

    def __init__(self, fn: Callable, modules: Sequence[torch.nn.Module], enabled: bool = True):
        self.fn = fn
        self.modules = list(modules)
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self._graphed = None
        self._sig = None
        self.n_graphed = 0
        self.n_eager = 0

    @staticmethod
    def _signature(args):
        return tuple((tuple(a.shape), a.dtype, a.device) if torch.is_tensor(a) else ("x", a)
                     for a in args)

    def build(self, example) -> None:
        dev = next((a.device for a in example if torch.is_tensor(a)), None)
        if dev is None or dev.type != "cuda":
            self.enabled = False
            return
        cuda_rng = torch.cuda.get_rng_state(dev)
        cpu_rng = torch.get_rng_state()
        # warm-up and capture must not see .grad from a previous step
        saved = [(p, p.grad) for m in self.modules for p in m.parameters()]
        for p, _ in saved:
            p.grad = None
        static = tuple(a.detach().clone() if torch.is_tensor(a) else a for a in example)
        self._graphed = torch.cuda.make_graphed_callables(
            _Wrap(self.fn, self.modules), static, allow_unused_input=True)
        for p, g in saved:
            p.grad = g
        torch.cuda.set_rng_state(cuda_rng, dev)
        torch.set_rng_state(cpu_rng)
        self._sig = self._signature(example)

    def __call__(self, *args):
        if self.enabled and all(m.training for m in self.modules):
            sig = self._signature(args)
            if self._graphed is None:
                self.build(args)
            if self._graphed is not None and sig == self._sig:
                self.n_graphed += 1
                return self._graphed(*args)
        self.n_eager += 1
        return self.fn(*args)
