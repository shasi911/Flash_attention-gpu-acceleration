"""
End-to-end benchmarking harness for the Transformer model (Assignment 3).

Usage examples
--------------
Forward pass only (small model, context length 128):
    uv run python -m systems.benchmark --size small --context 128 --mode forward

Forward + backward (medium model, bfloat16, context length 512):
    uv run python -m systems.benchmark --size medium --context 512 --mode both --dtype bf16

Memory profiling (small model, context 256):
    uv run python -m systems.benchmark --size small --context 256 --profile-memory
"""

from __future__ import annotations

import argparse
import contextlib
import math
import statistics
import timeit
from contextlib import nullcontext

import torch

# Model size configurations (Table 1 in the assignment)
MODEL_CONFIGS = {
    "small": dict(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}

VOCAB_SIZE = 10_000
BATCH_SIZE = 4


def get_model(size: str, device: torch.device, dtype: torch.dtype = torch.float32):
    try:
        import basics.model as bm
    except ImportError:
        raise ImportError(
            "Could not import basics.model. Make sure the basics package is installed "
            "and the pyproject.toml path is correct."
        )

    cfg = MODEL_CONFIGS[size]
    model = bm.BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=8192,  # large enough for all context lengths we test
        **cfg,
    )
    model = model.to(device=device, dtype=dtype)
    return model


def time_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    mode: str,
    autocast_ctx,
    n_warmup: int,
    n_steps: int,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (mean_seconds, std_seconds) across n_steps measurement steps."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def run_one():
        with autocast_ctx:
            logits = model(inputs)
        if mode in ("both", "backward"):
            loss = logits.sum()
            loss.backward()
            if mode == "both":
                optimizer.step()
                optimizer.zero_grad()
        torch.cuda.synchronize() if device.type == "cuda" else None

    # Warm-up
    for _ in range(n_warmup):
        with torch.no_grad() if mode == "forward" else contextlib.nullcontext():
            run_one()

    times = []
    for _ in range(n_steps):
        t0 = timeit.default_timer()
        if mode == "forward":
            with torch.no_grad():
                run_one()
        else:
            run_one()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(timeit.default_timer() - t0)

    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def run_memory_profile(model, inputs, mode, autocast_ctx, output_path="memory_snapshot.pickle"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if not torch.cuda.is_available():
        print("Memory profiling requires CUDA. Skipping.")
        return

    # Warm-up step
    with torch.no_grad():
        with autocast_ctx:
            _ = model(inputs)
    torch.cuda.synchronize()

    torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    with autocast_ctx:
        logits = model(inputs)
    if mode in ("both", "backward"):
        loss = logits.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    torch.cuda.memory._dump_snapshot(output_path)
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"Memory snapshot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Transformer end-to-end benchmarking")
    parser.add_argument("--size", choices=list(MODEL_CONFIGS), default="small")
    parser.add_argument("--context", type=int, default=128, help="Sequence length")
    parser.add_argument("--mode", choices=["forward", "both"], default="forward",
                        help="'forward' = inference only; 'both' = fwd+bwd+optim step")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--memory-output", default="memory_snapshot.pickle")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dtype = torch.float32
    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else None

    print(f"Device: {device}  |  base dtype: {base_dtype}  |  autocast: {autocast_dtype}")

    model = get_model(args.size, device, base_dtype)

    # Random input tokens
    inputs = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, args.context), device=device)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )

    if args.profile_memory:
        run_memory_profile(model, inputs, args.mode, autocast_ctx, args.memory_output)
        return

    mean_t, std_t = time_step(
        model, inputs, args.mode, autocast_ctx,
        n_warmup=args.warmup, n_steps=args.steps, device=device,
    )

    print(
        f"Model: {args.size:<8}  context: {args.context:<6}  mode: {args.mode:<8}  "
        f"dtype: {args.dtype:<5}  "
        f"mean: {mean_t*1e3:.2f} ms  std: {std_t*1e3:.2f} ms"
    )


if __name__ == "__main__":
    main()
