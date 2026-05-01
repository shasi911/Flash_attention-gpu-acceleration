"""
FlashAttention-2 implementations for Assignment 3 (Systems).

Provides two implementations:
  - FlashAttentionPyTorch: tiled forward pass in pure PyTorch, compiled backward
  - FlashAttentionTriton: tiled forward pass via a custom Triton kernel, compiled backward
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared backward pass (PyTorch + torch.compile)
# ---------------------------------------------------------------------------

def _flash_attn_backward(Q, K, V, O, dO, L, is_causal: bool):
    """
    Backward pass for FlashAttention-2 following equations 13-19 in the assignment.

    Recomputes the attention matrix P from Q, K, and the saved log-sum-exp L,
    avoiding any O(N^2) memory stored from the forward pass.
    """
    batch, nq, d = Q.shape
    nk = K.shape[1]
    scale = 1.0 / math.sqrt(d)

    # Work in float32 throughout for numerical stability
    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()
    O_f = O.float()
    dO_f = dO.float()
    L_f = L.float()

    # D = rowsum(O ∘ dO)  [section before eq. 13]
    D = (O_f * dO_f).sum(dim=-1)  # (batch, nq)

    # Eq. 13 – recompute S
    S = torch.bmm(Q_f, K_f.transpose(1, 2)) * scale  # (batch, nq, nk)

    if is_causal:
        q_idx = torch.arange(nq, device=Q.device).unsqueeze(1)   # (nq, 1)
        k_idx = torch.arange(nk, device=Q.device).unsqueeze(0)   # (1, nk)
        S = S.masked_fill((q_idx < k_idx).unsqueeze(0), -1e6)

    # Eq. 14 – recompute P without storing the full N×N matrix
    P = torch.exp(S - L_f.unsqueeze(2))  # (batch, nq, nk)

    # Eq. 15 – dV
    dV = torch.bmm(P.transpose(1, 2), dO_f)  # (batch, nk, d)

    # Eq. 16 – dP
    dP = torch.bmm(dO_f, V_f.transpose(1, 2))  # (batch, nq, nk)

    # Eq. 17 – dS
    dS = P * (dP - D.unsqueeze(2))  # (batch, nq, nk)

    # Eq. 18 – dQ
    dQ = torch.bmm(dS, K_f) * scale  # (batch, nq, d)

    # Eq. 19 – dK
    dK = torch.bmm(dS.transpose(1, 2), Q_f) * scale  # (batch, nk, d)

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


try:
    _compiled_backward = torch.compile(_flash_attn_backward)
except Exception:
    _compiled_backward = _flash_attn_backward


def _run_backward(Q, K, V, O, dO, L, is_causal):
    try:
        return _compiled_backward(Q, K, V, O, dO, L, is_causal)
    except Exception:
        return _flash_attn_backward(Q, K, V, O, dO, L, is_causal)


# ---------------------------------------------------------------------------
# Pure PyTorch implementation (Algorithm 1 tiled forward, compiled backward)
# ---------------------------------------------------------------------------

class FlashAttentionPyTorch(torch.autograd.Function):
    """
    FlashAttention-2 forward pass in pure PyTorch (no Triton).

    The forward pass tiles Q and K/V, maintaining running (m, l, O) accumulators
    so the full N×N attention matrix is never materialised.  The saved log-sum-exp
    vector L is used by the backward pass to recompute P without storing it.
    """

    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        """
        Args:
            Q:         (batch, n_queries, d)
            K:         (batch, n_keys, d)
            V:         (batch, n_keys, d)
            is_causal: whether to apply a causal attention mask

        Returns:
            O: (batch, n_queries, d)
        """
        batch, nq, d = Q.shape
        nk = K.shape[1]
        scale = 1.0 / math.sqrt(d)

        Q_TILE = max(16, min(64, nq))
        K_TILE = max(16, min(64, nk))

        O = torch.zeros(batch, nq, d, device=Q.device, dtype=Q.dtype)
        L = torch.zeros(batch, nq, device=Q.device, dtype=torch.float32)

        Tq = math.ceil(nq / Q_TILE)
        Tk = math.ceil(nk / K_TILE)

        for i in range(Tq):
            qi_s = i * Q_TILE
            qi_e = min(qi_s + Q_TILE, nq)
            bq = qi_e - qi_s

            Qi = Q[:, qi_s:qi_e, :]  # (batch, bq, d)

            Oi = torch.zeros(batch, bq, d, device=Q.device, dtype=torch.float32)
            li = torch.zeros(batch, bq, device=Q.device, dtype=torch.float32)
            mi = torch.full((batch, bq), float("-inf"), device=Q.device, dtype=torch.float32)

            for j in range(Tk):
                kj_s = j * K_TILE
                kj_e = min(kj_s + K_TILE, nk)

                Kj = K[:, kj_s:kj_e, :]  # (batch, bk, d)
                Vj = V[:, kj_s:kj_e, :]  # (batch, bk, d)

                Sij = torch.bmm(Qi.float(), Kj.float().transpose(1, 2)) * scale

                if is_causal:
                    q_idx = torch.arange(qi_s, qi_e, device=Q.device).unsqueeze(1)
                    k_idx = torch.arange(kj_s, kj_e, device=Q.device).unsqueeze(0)
                    Sij = Sij.masked_fill((q_idx < k_idx).unsqueeze(0), -1e6)

                mij = Sij.max(dim=-1).values            # (batch, bq)
                mi_new = torch.maximum(mi, mij)

                Ptilde = torch.exp(Sij - mi_new.unsqueeze(2))   # (batch, bq, bk)

                li_new = torch.exp(mi - mi_new) * li + Ptilde.sum(dim=-1)

                Oi = torch.exp(mi - mi_new).unsqueeze(2) * Oi + torch.bmm(Ptilde, Vj.float())

                mi = mi_new
                li = li_new

            Oi = Oi / li.unsqueeze(2)
            Li = mi + torch.log(li)

            O[:, qi_s:qi_e, :] = Oi.to(Q.dtype)
            L[:, qi_s:qi_e] = Li

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = _run_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None   # None for is_causal (non-tensor arg)


# ---------------------------------------------------------------------------
# Triton implementation
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:

    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """
        FlashAttention-2 forward kernel.

        Launch grid: (ceil(N_QUERIES / Q_TILE_SIZE), batch_size)
        Each program instance handles one query tile for one batch element.
        """
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        # ------------------------------------------------------------------
        # Set up block pointers
        # ------------------------------------------------------------------
        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D),
            strides=(stride_qq, stride_qd),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )

        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(0, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        # ------------------------------------------------------------------
        # Load query tile into SRAM (stays resident across all key tiles)
        # ------------------------------------------------------------------
        Q_tile = tl.load(Q_block_ptr)  # (Q_TILE_SIZE, D)

        # Running accumulators – always float32 for precision
        O_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        l_acc = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
        m_acc = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

        q_start = query_tile_index * Q_TILE_SIZE
        n_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

        # ------------------------------------------------------------------
        # Iterate over key tiles
        # ------------------------------------------------------------------
        for j in range(n_key_tiles):
            K_tile = tl.load(K_block_ptr)  # (K_TILE_SIZE, D)
            V_tile = tl.load(V_block_ptr)  # (K_TILE_SIZE, D)

            # S_ij = Q_i @ K_j^T * scale → (Q_TILE_SIZE, K_TILE_SIZE) in float32
            S_ij = tl.dot(Q_tile, tl.trans(K_tile)).to(tl.float32) * scale

            # Optional causal mask (constexpr → zero runtime cost when False)
            if IS_CAUSAL:
                q_idx = (q_start + tl.arange(0, Q_TILE_SIZE))[:, None]
                k_idx = (j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE))[None, :]
                S_ij = tl.where(q_idx >= k_idx, S_ij, -1e6)

            # Running max update
            m_ij = tl.max(S_ij, axis=1)               # (Q_TILE_SIZE,)
            m_new = tl.maximum(m_acc, m_ij)

            # Unnormalized softmax numerators
            P_tilde = tl.exp(S_ij - m_new[:, None])   # (Q_TILE_SIZE, K_TILE_SIZE)

            # Update running denominator
            l_acc = tl.exp(m_acc - m_new) * l_acc + tl.sum(P_tilde, axis=1)

            # Update output accumulator:
            #   O = diag(exp(m_old - m_new)) @ O + P_tilde @ V
            O_acc = O_acc * tl.exp(m_acc - m_new)[:, None]
            O_acc = tl.dot(P_tilde.to(V_tile.dtype), V_tile, acc=O_acc)

            m_acc = m_new

            # Advance to next key tile
            K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
            V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        # ------------------------------------------------------------------
        # Final normalisation and log-sum-exp
        # ------------------------------------------------------------------
        O_acc = O_acc / l_acc[:, None]
        L_tile = m_acc + tl.log(l_acc)   # (Q_TILE_SIZE,)

        # ------------------------------------------------------------------
        # Store results
        # ------------------------------------------------------------------
        O_block_ptr = tl.make_block_ptr(
            O_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D),
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        tl.store(O_block_ptr, O_acc.to(O_block_ptr.type.element_ty))

        L_block_ptr = tl.make_block_ptr(
            L_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,),
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,),
            order=(0,),
        )
        tl.store(L_block_ptr, L_tile)

    # -----------------------------------------------------------------------

    class FlashAttentionTriton(torch.autograd.Function):
        """FlashAttention-2 with a Triton kernel for the forward pass."""

        @staticmethod
        def forward(ctx, Q, K, V, is_causal: bool = False):
            batch, nq, d = Q.shape
            nk = K.shape[1]
            scale = 1.0 / math.sqrt(d)

            # Tile sizes must be powers of 2 and at least 16
            Q_TILE_SIZE = max(16, min(64, triton.next_power_of_2(nq)))
            K_TILE_SIZE = max(16, min(64, triton.next_power_of_2(nk)))

            O = torch.empty(batch, nq, d, device=Q.device, dtype=Q.dtype)
            L = torch.empty(batch, nq, device=Q.device, dtype=torch.float32)

            Tq = math.ceil(nq / Q_TILE_SIZE)

            flash_fwd_kernel[(Tq, batch)](
                Q, K, V,
                O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                N_QUERIES=nq, N_KEYS=nk,
                scale=scale,
                D=d,
                Q_TILE_SIZE=Q_TILE_SIZE,
                K_TILE_SIZE=K_TILE_SIZE,
                IS_CAUSAL=is_causal,
            )

            ctx.save_for_backward(Q, K, V, O, L)
            ctx.is_causal = is_causal

            return O

        @staticmethod
        def backward(ctx, dO):
            Q, K, V, O, L = ctx.saved_tensors
            dQ, dK, dV = _run_backward(Q, K, V, O, dO, L, ctx.is_causal)
            return dQ, dK, dV, None

else:
    # Stub when Triton is not importable (CPU-only systems)
    class FlashAttentionTriton(torch.autograd.Function):  # type: ignore[no-redef]
        @staticmethod
        def forward(ctx, Q, K, V, is_causal=False):
            raise RuntimeError(
                "Triton is not available on this system. "
                "Install triton or run on a CUDA-capable machine."
            )

        @staticmethod
        def backward(ctx, dO):
            raise RuntimeError("Triton is not available.")
