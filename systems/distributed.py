"""
Distributed training utilities for Assignment 3 (Systems).

Implements:
  - DDPIndividualParameters: per-parameter async all-reduce
  - DDPBucketed: bucket-based async all-reduce
  - ShardedOptimizer: ZeRO-style optimizer state sharding
"""

from __future__ import annotations

from typing import Type

import torch
import torch.distributed as dist
import torch.nn as nn


def _dedup_trainable(module: nn.Module) -> list[nn.Parameter]:
    """Return deduplicated list of trainable parameters in forward-pass order."""
    seen: set[int] = set()
    out: list[nn.Parameter] = []
    for p in module.parameters():
        if p.requires_grad and id(p) not in seen:
            out.append(p)
            seen.add(id(p))
    return out


# ---------------------------------------------------------------------------
# DDP – individual parameter synchronisation
# ---------------------------------------------------------------------------

class DDPIndividualParameters(nn.Module):
    """
    DDP wrapper that fires an async all-reduce for each parameter
    immediately after its gradient is accumulated during backward.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._handles: list = []

        # Broadcast rank-0 parameters to every other rank
        for p in module.parameters():
            dist.broadcast(p.data, src=0)

        # One post-accumulate hook per unique trainable parameter
        for p in _dedup_trainable(module):
            p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, p: nn.Parameter) -> None:
        handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._handles.append(handle)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def ddp_individual_parameters_on_after_backward(
    ddp_model: DDPIndividualParameters,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Wait for all pending all-reduces, then normalise gradients to their mean."""
    for h in ddp_model._handles:
        h.wait()
    ws = dist.get_world_size()
    for p in _dedup_trainable(ddp_model.module):
        if p.grad is not None:
            p.grad.div_(ws)
    ddp_model._handles.clear()


# ---------------------------------------------------------------------------
# DDP – bucketed gradient synchronisation
# ---------------------------------------------------------------------------

class DDPBucketed(nn.Module):
    """
    DDP wrapper that groups parameters into fixed-size buckets and launches
    one all-reduce per bucket as soon as every parameter in it has a gradient.
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        max_bytes = int(bucket_size_mb * 1024 * 1024)

        # Broadcast rank-0 parameters
        for p in module.parameters():
            dist.broadcast(p.data, src=0)

        trainable = _dedup_trainable(module)

        # Build buckets in reverse parameter order so that the last parameters
        # (whose gradients are computed first in backward) fill buckets first.
        buckets: list[list[nn.Parameter]] = []
        cur: list[nn.Parameter] = []
        cur_bytes = 0
        for p in reversed(trainable):
            pb = p.numel() * p.element_size()
            if cur_bytes > 0 and cur_bytes + pb > max_bytes:
                buckets.append(cur)
                cur = [p]
                cur_bytes = pb
            else:
                cur.append(p)
                cur_bytes += pb
        if cur:
            buckets.append(cur)

        self._buckets = buckets
        self._expected = [len(b) for b in buckets]
        self._ready: list[int] = [0] * len(buckets)
        self._handles: list[tuple] = []
        # Map each parameter id to its bucket index
        self._p2b: dict[int, int] = {id(p): i for i, b in enumerate(buckets) for p in b}

        for p in trainable:
            p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, p: nn.Parameter):
        bidx = self._p2b[id(p)]
        expected = self._expected[bidx]

        def hook(param: nn.Parameter) -> None:
            self._ready[bidx] += 1
            if self._ready[bidx] == expected:
                flat = torch.cat([q.grad.flatten() for q in self._buckets[bidx]])
                handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
                self._handles.append((handle, bidx, flat))

        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def ddp_bucketed_on_after_backward(
    ddp_model: DDPBucketed,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Wait for every bucket all-reduce and write the averaged gradient back."""
    ws = dist.get_world_size()
    for handle, bidx, flat in ddp_model._handles:
        handle.wait()
        offset = 0
        for p in ddp_model._buckets[bidx]:
            n = p.numel()
            p.grad.copy_(flat[offset : offset + n].view_as(p.grad) / ws)
            offset += n
    ddp_model._handles.clear()


def ddp_bucketed_on_train_batch_start(
    ddp_model: DDPBucketed,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Reset per-bucket gradient-ready counters at the start of each training step."""
    ddp_model._ready = [0] * len(ddp_model._buckets)


# ---------------------------------------------------------------------------
# Sharded optimizer (ZeRO stage 1)
# ---------------------------------------------------------------------------

class ShardedOptimizer:
    """
    Wraps an optimizer so that each rank only maintains state for its assigned
    subset of parameters (round-robin partition).  After every step, updated
    parameters are broadcast from their owning rank to all others.
    """

    def __init__(
        self,
        params,
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs,
    ):
        self._ws = dist.get_world_size()
        self._rank = dist.get_rank()
        self._all_params: list[nn.Parameter] = list(params)

        local = [p for i, p in enumerate(self._all_params) if i % self._ws == self._rank]
        self._optimizer = optimizer_cls(local, **kwargs) if local else None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self._all_params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def step(self) -> None:
        if self._optimizer is not None:
            self._optimizer.step()
        # Each parameter is broadcast from its owning rank so all ranks converge.
        for i, p in enumerate(self._all_params):
            dist.broadcast(p.data, src=i % self._ws)

    @property
    def param_groups(self):
        return self._optimizer.param_groups if self._optimizer else []
