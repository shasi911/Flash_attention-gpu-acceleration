from __future__ import annotations

from typing import Type

import torch

from systems.distributed import (
    DDPBucketed,
    DDPIndividualParameters,
    ShardedOptimizer,
    ddp_bucketed_on_after_backward as _ddp_bucketed_on_after_backward,
    ddp_bucketed_on_train_batch_start as _ddp_bucketed_on_train_batch_start,
    ddp_individual_parameters_on_after_backward as _ddp_individual_on_after_backward,
)
from systems.flash_attention import FlashAttentionPyTorch, FlashAttentionTriton


def get_flashattention_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using only standard PyTorch operations (no Triton).
    """
    return FlashAttentionPyTorch


def get_flashattention_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using custom Triton kernels in the forward pass and a compiled PyTorch
    backward pass.
    """
    return FlashAttentionTriton


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a DDP wrapper that synchronises each parameter's gradient individually
    (asynchronously, overlapping with backprop).
    """
    return DDPIndividualParameters(module)


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Called after the backward pass, before the optimizer step."""
    _ddp_individual_on_after_backward(ddp_model, optimizer)


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a DDP wrapper that synchronises gradients in buckets (asynchronously,
    overlapping with backprop).
    """
    return DDPBucketed(module, bucket_size_mb)


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Called after the backward pass, before the optimizer step."""
    _ddp_bucketed_on_after_backward(ddp_model, optimizer)


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """Called at the very start of each training step."""
    _ddp_bucketed_on_train_batch_start(ddp_model, optimizer)


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns an optimizer that shards optimizer state across distributed ranks.
    """
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
