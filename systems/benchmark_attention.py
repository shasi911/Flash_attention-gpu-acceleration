"""
Attention-specific benchmarking script (Assignment 3 – pytorch_attention / torch_compile problems).

Sweeps over head-embedding dimensions and sequence lengths, timing both forward
and backward passes for:
  - Vanilla PyTorch attention
  - torch.compile-d PyTorch attention
  - FlashAttention-2 (Triton forward + compiled backward)

Usage
-----
    uv run python -m systems.benchmark_attention
"""

from __future__ import annotations

import argparse
import math
from contextlib import nullcontext

import torch

try:
    import triton.testing
    _TRITON = True
except ImportError:
    _TRITON = False

from systems.flash_attention import FlashAttentionPyTorch, FlashAttentionTriton


# ---------------------------------------------------------------------------
# Vanilla attention (no FlashAttention)
# ---------------------------------------------------------------------------

def vanilla_attention(Q, K, V, is_causal=False):
    """Standard scaled dot-product attention."""
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    S = torch.bmm(Q, K.transpose(1, 2)) * scale
    if is_causal:
        nq, nk = Q.shape[1], K.shape[1]
        mask = torch.tril(torch.ones(nq, nk, device=Q.device, dtype=torch.bool))
        S = S.masked_fill(~mask.unsqueeze(0), float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.bmm(P, V)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _bench_fwd(fn, Q, K, V, n=100):
    """Returns mean forward pass time in ms."""
    if not torch.cuda.is_available():
        return float("nan")

    def run():
        fn(Q, K, V)
        torch.cuda.synchronize()

    # warm-up
    for _ in range(10):
        run()

    if _TRITON:
        ms = triton.testing.do_bench(lambda: (fn(Q, K, V), torch.cuda.synchronize()))
        return ms
    else:
        import timeit
        t = timeit.timeit(run, number=n) / n * 1000
        return t


def _bench_bwd(fn, Q, K, V, n=100):
    """Returns (mem_before_bwd_MB, mean_bwd_time_ms)."""
    if not torch.cuda.is_available():
        return float("nan"), float("nan")

    Q = Q.detach().requires_grad_(True)
    K = K.detach().requires_grad_(True)
    V = V.detach().requires_grad_(True)

    # Forward to get output and capture graph
    out = fn(Q, K, V)
    do = torch.randn_like(out)

    torch.cuda.synchronize()
    mem_mb = torch.cuda.memory_allocated() / 1024**2

    def run():
        out = fn(Q, K, V)
        out.backward(do)
        torch.cuda.synchronize()
        Q.grad = K.grad = V.grad = None

    for _ in range(5):
        run()

    if _TRITON:
        ms = triton.testing.do_bench(run)
    else:
        import timeit
        ms = timeit.timeit(run, number=n) / n * 1000

    return mem_mb, ms


def benchmark_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("No CUDA device found – skipping attention benchmark.")
        return

    BATCH = 8
    D_VALS = [16, 32, 64, 128]
    SEQ_VALS = [256, 1024, 4096, 8192, 16384]

    compiled_vanilla = torch.compile(vanilla_attention)

    header = (
        f"{'D':>5} {'SeqLen':>7}  "
        f"{'VanFwd(ms)':>12} {'VanBwd(ms)':>12} "
        f"{'CmpFwd(ms)':>12} {'CmpBwd(ms)':>12} "
        f"{'FA2Fwd(ms)':>12} {'FA2Bwd(ms)':>12}"
    )
    print(header)
    print("-" * len(header))

    for d in D_VALS:
        for seq in SEQ_VALS:
            try:
                Q = torch.randn(BATCH, seq, d, device=device, requires_grad=True)
                K = torch.randn(BATCH, seq, d, device=device, requires_grad=True)
                V = torch.randn(BATCH, seq, d, device=device, requires_grad=True)

                van_fwd = _bench_fwd(vanilla_attention, Q, K, V)
                _, van_bwd = _bench_bwd(vanilla_attention, Q, K, V)

                cmp_fwd = _bench_fwd(compiled_vanilla, Q, K, V)
                _, cmp_bwd = _bench_bwd(compiled_vanilla, Q, K, V)

                fa2_fwd = _bench_fwd(lambda q, k, v: FlashAttentionTriton.apply(q, k, v, False), Q, K, V)
                _, fa2_bwd = _bench_bwd(lambda q, k, v: FlashAttentionTriton.apply(q, k, v, False), Q, K, V)

                print(
                    f"{d:>5} {seq:>7}  "
                    f"{van_fwd:>12.3f} {van_bwd:>12.3f} "
                    f"{cmp_fwd:>12.3f} {cmp_bwd:>12.3f} "
                    f"{fa2_fwd:>12.3f} {fa2_bwd:>12.3f}"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"{d:>5} {seq:>7}  OOM")
                torch.cuda.empty_cache()


def benchmark_flash_vs_pytorch():
    """
    Benchmarking for the flash_benchmarking problem:
    sweep over seq_len ∈ powers-of-2 from 128..65536,
    embed_dim ∈ powers-of-2 from 16..128, both dtypes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("No CUDA device found.")
        return

    SEQ_LENS = [2**i for i in range(7, 17)]   # 128..65536
    EMBED_DIMS = [2**i for i in range(4, 8)]   # 16..128
    DTYPES = [torch.bfloat16, torch.float32]
    BATCH = 1

    for dtype in DTYPES:
        print(f"\n=== dtype={dtype} ===")
        header = f"{'SeqLen':>7} {'EmbDim':>7}  {'PyFwd':>10} {'PyBwd':>10} {'FA2Fwd':>10} {'FA2Bwd':>10}"
        print(header)
        print("-" * len(header))
        for seq in SEQ_LENS:
            for d in EMBED_DIMS:
                try:
                    Q = torch.randn(BATCH, seq, d, device=device, dtype=dtype, requires_grad=True)
                    K = torch.randn(BATCH, seq, d, device=device, dtype=dtype, requires_grad=True)
                    V = torch.randn(BATCH, seq, d, device=device, dtype=dtype, requires_grad=True)

                    py_fwd = _bench_fwd(
                        lambda q, k, v: vanilla_attention(q.float(), k.float(), v.float()), Q, K, V)
                    _, py_bwd = _bench_bwd(
                        lambda q, k, v: vanilla_attention(q.float(), k.float(), v.float()), Q, K, V)

                    fa_fwd = _bench_fwd(
                        lambda q, k, v: FlashAttentionTriton.apply(q, k, v, True), Q, K, V)
                    _, fa_bwd = _bench_bwd(
                        lambda q, k, v: FlashAttentionTriton.apply(q, k, v, True), Q, K, V)

                    print(f"{seq:>7} {d:>7}  {py_fwd:>10.3f} {py_bwd:>10.3f} {fa_fwd:>10.3f} {fa_bwd:>10.3f}")
                except torch.cuda.OutOfMemoryError:
                    print(f"{seq:>7} {d:>7}  OOM")
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["attention", "flash"], default="attention")
    args = parser.parse_args()

    if args.mode == "attention":
        benchmark_attention()
    else:
        benchmark_flash_vs_pytorch()
