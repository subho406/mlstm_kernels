# Author Korbinian Poeppel
from typing import Optional, Literal

import torch
import triton
import triton.language as tl
from torch.amp import custom_fwd, custom_bwd
from dataclasses import dataclass

from ...kernel_utils import contiguous


@triton.jit
def chunk_mlstm_fwd_kernel_C(
    k,
    v,
    C,
    n,
    m,
    i,  # log igates
    f,  # accumulated log fgate
    initial_C,  # initial state of the chunk [B, H, D_head_K, D_head_V]
    initial_n,
    initial_m,
    final_C,  # final state of the chunk [B, H, D_head_K, D_head_V]
    final_n,
    final_m,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vh_h,
    s_vh_t,
    s_vh_d,
    s_C_h,
    s_C_t,
    s_n_h,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_k, i_v, i_bC = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if USE_INITIAL_STATE:
        p_C0 = tl.make_block_ptr(
            initial_C + i_bC * K * V,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        p_n0 = tl.make_block_ptr(
            initial_n + i_bC * K,
            (K,),
            (1,),
            (i_k * BK,),
            (BK,),
            (0,),
        )
        p_m0 = initial_m + i_bC

        b_C = tl.load(p_C0, boundary_check=(0, 1))
        b_n = tl.load(p_n0, boundary_check=(0,))
        b_m = tl.load(p_m0)
    else:
        b_C = tl.zeros([BK, BV], dtype=tl.load(C).dtype)
        b_n = tl.zeros([BK], dtype=b_C.dtype)
        b_m = 0.0

    b_m_next = 0.0
    for i_t in range(NT):
        p_k = tl.make_block_ptr(
            k + i_bC * s_qk_h,
            (K, T),
            (s_qk_d, s_qk_t),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        p_v = tl.make_block_ptr(
            v + i_bC * s_vh_h,
            (T, V),
            (s_vh_t, s_vh_d),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_C = tl.make_block_ptr(
            C + i_bC * s_C_h + i_t * K * V,
            (K, V),
            (s_C_t, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        p_n = tl.make_block_ptr(
            n + i_bC * s_n_h + i_t * K, (K,), (1,), (i_k * BK,), (BK,), (0,)
        )
        tl.store(p_C, b_C.to(p_C.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_n, b_n.to(p_n.dtype.element_ty), boundary_check=(0,))
        tl.store(m + i_bC * (NT + 1) + i_t, b_m)
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BK, BV]
        b_f_last = tl.load(f + i_bC * T + i_t * BT + BT - 1)
        b_f = tl.load(f + i_bC * T + i_t * BT + tl.arange(0, BT))
        b_i = tl.load(i + i_bC * T + i_t * BT + tl.arange(0, BT))
        b_g = b_i + b_f_last - b_f

        b_m_next, _ = tl.max(b_g)
        b_m_next = tl.maximum(b_f_last + b_m, b_m_next)

        b_C *= tl.math.exp2(b_f_last - b_m_next + b_m).to(b_C.dtype)
        b_n *= tl.math.exp2(b_f_last - b_m_next + b_m).to(b_n.dtype)
        b_C += tl.dot(
            b_k,
            b_v * (tl.math.exp2(b_g - b_m_next)[:, None]).to(b_k.dtype),
            allow_tf32=False,
        ).to(b_C.dtype)
        b_n += tl.sum(b_k * tl.math.exp2(b_g - b_m_next).to(b_k.dtype), axis=1).to(
            b_n.dtype
        )
        b_m = b_m_next

    tl.store(m + i_bC * (NT + 1) + NT, b_m)
    if STORE_FINAL_STATE:
        p_Ct = tl.make_block_ptr(
            final_C + i_bC * K * V,
            (K, V),
            (V, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        p_n = tl.make_block_ptr(
            final_n + i_bC * K, (K,), (1,), (i_k * BK,), (BK,), (0,)
        )
        tl.store(p_Ct, b_C.to(p_Ct.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_n, b_n.to(p_n.dtype.element_ty), boundary_check=(0,))
        tl.store(final_m + i_bC, b_m)


@triton.jit
def chunk_mlstm_fwd_kernel_h(
    q,
    k,
    v,
    C,
    n,
    m,
    m_total,
    i,
    f,
    h,
    norm,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vh_h,
    s_vh_t,
    s_vh_d,
    s_C_h,
    s_C_t,
    s_n_h,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    i_v, i_t, i_bC = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    h_i = tl.arange(0, BT)
    m_s = h_i[:, None] >= h_i[None, :]

    b_h = tl.zeros([BT, BV], dtype=tl.load(q).dtype)
    b_s = tl.zeros([BT, BT], dtype=b_h.dtype)
    b_norm = tl.zeros([BT, BV], dtype=b_h.dtype)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(
            q + i_bC * s_qk_h,
            (T, K),
            (s_qk_t, s_qk_d),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k + i_bC * s_qk_h,
            (K, T),
            (s_qk_d, s_qk_t),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        p_C = tl.make_block_ptr(
            C + i_bC * s_C_h + i_t * K * V,
            (K, V),
            (s_C_t, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        p_n = tl.make_block_ptr(
            n + i_bC * s_n_h + i_t * K,
            (K, BV),
            (1, 0),
            (i_k * BK, 0),
            (BK, BV),
            (0, 1),
        )

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BT]

        # [BK, BV]
        b_C = tl.load(p_C, boundary_check=(0, 1))
        b_n = tl.load(p_n, boundary_check=(0,))
        b_h += tl.dot(b_q, b_C.to(b_q.dtype), allow_tf32=False).to(b_h.dtype)
        b_s += tl.dot(b_q, b_k, allow_tf32=False).to(b_s.dtype)
        b_n2 = tl.dot(b_q, b_n.to(b_q.dtype), allow_tf32=False).to(b_norm.dtype)
        b_norm += b_n2

    p_f = f + i_bC * T + i_t * BT + tl.arange(0, BT)
    b_f = tl.load(p_f)
    p_i = i + i_bC * T + i_t * BT + tl.arange(0, BT)
    b_i = tl.load(p_i)
    b_m = tl.load(m + i_bC * (NT + 1) + i_t)

    # TODO revise this to the stabilized version
    b_logD = b_i[None, :] + b_f[:, None] - b_f[None, :]
    b_logD = tl.where(m_s, b_logD, -float("inf"))
    b_mlogD = tl.max(b_logD, axis=1)

    b_m_total = tl.maximum(b_f + b_m, b_mlogD)
    p_m_total = tl.make_block_ptr(
        m_total + T * i_bC, (T,), (1,), (i_t * BT,), (BT,), (0,)
    )
    tl.store(p_m_total, b_m_total.to(p_m_total.dtype.element_ty), boundary_check=(0,))

    b_D = tl.math.exp2(b_logD - b_m_total[:, None])
    b_h = b_h * tl.math.exp2(b_f + b_m - b_m_total)[:, None] * scale
    b_s = b_s * b_D * scale
    b_norm = b_norm * tl.math.exp2(b_f + b_m - b_m_total)[:, None] * scale

    b_s = tl.where(m_s, b_s, 0)
    b_norm += tl.sum(b_s, axis=1)[:, None]
    b_norm = tl.abs(b_norm)

    b_norm = tl.maximum(b_norm.to(b_m_total.dtype), tl.math.exp2(-b_m_total)[:, None])

    tl.store(norm + i_bC * T + i_t * BT + tl.arange(0, BT), tl.max(b_norm, axis=1))

    p_v = tl.make_block_ptr(
        v + i_bC * s_vh_h,
        (T, V),
        (s_vh_t, s_vh_d),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_h = (b_h + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) / b_norm.to(
        b_h.dtype
    )
    p_h = tl.make_block_ptr(
        h + i_bC * s_vh_h,
        (T, V),
        (s_vh_t, s_vh_d),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_mlstm_bwd_kernel_dC(
    q,
    f,
    m,
    m_total,
    norm,
    dh,
    dC,
    final_dC,
    final_m,
    initial_dC,
    initial_m,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vh_h,
    s_vh_t,
    s_vh_d,
    s_C_h,
    s_C_t,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    i_k, i_v, i_bC = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BK, BV]
    p_dC = tl.make_block_ptr(
        final_dC + i_bC * K * V,
        (K, V),
        (s_C_t, 1),
        (i_k * BK, i_v * BV),
        (BK, BV),
        (1, 0),
    )
    b_dC = tl.load(p_dC, boundary_check=(0, 1))
    b_m = tl.load(final_m + i_bC)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(
            q + i_bC * s_qk_h,
            (K, T),
            (s_qk_d, s_qk_t),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        p_dh = tl.make_block_ptr(
            dh + i_bC * s_vh_h,
            (T, V),
            (s_vh_t, s_vh_d),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_dC = tl.make_block_ptr(
            dC + i_bC * s_C_h + i_t * K * V,
            (K, V),
            (s_C_t, 1),
            (i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )

        tl.store(p_dC, b_dC.to(p_dC.dtype.element_ty), boundary_check=(0, 1))
        b_f_last = tl.load(f + i_bC * T + i_t * BT + BT - 1)
        b_f = tl.load(f + i_bC * T + i_t * BT + tl.arange(0, BT))
        b_m_p = tl.load(m + i_bC * (NT + 1) + i_t)
        b_m_total = tl.load(m_total + i_bC * T + i_t * BT + tl.arange(0, BT))
        b_norm = tl.load(norm + i_bC * T + i_t * BT + tl.arange(0, BT))

        # [BK, BT]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale * tl.math.exp2(b_f + b_m_p - b_m_total)[None, :]).to(
            b_q.dtype
        )
        # [BT, V]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        b_dh /= b_norm[:, None].to(b_dh.dtype)
        # [BK, BV]
        b_dC *= tl.math.exp2(b_f_last + b_m_p - b_m).to(b_dC.dtype)
        b_dC += tl.dot(b_q, b_dh.to(b_q.dtype), allow_tf32=False).to(b_dC.dtype)
        b_m = b_m_p

    p_initial_dC = tl.make_block_ptr(
        initial_dC + i_bC * K * V,
        (K, V),
        (V, 1),
        (i_k * BK, i_v * BV),
        (BK, BV),
        (1, 0),
    )
    tl.store(
        p_initial_dC, b_dC.to(p_initial_dC.dtype.element_ty), boundary_check=(0, 1)
    )
    tl.store(initial_m + i_bC, b_m)


@triton.jit
def chunk_mlstm_bwd_kernel_dqkvif(
    q,
    k,
    v,
    C,
    m,
    m_total,
    final_m,
    norm,
    i,
    f,
    dh,
    dC,
    dq,
    dk,
    dv,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vh_h,
    s_vh_t,
    s_vh_d,
    s_C_h,
    s_C_t,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
):
    i_k, i_t, i_bC = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)

    p_q = tl.make_block_ptr(
        q + i_bC * s_qk_h,
        (K, T),
        (s_qk_d, s_qk_t),
        (i_k * BK, i_t * BT),
        (BK, BT),
        (0, 1),
    )
    p_k = tl.make_block_ptr(
        k + i_bC * s_qk_h,
        (T, K),
        (s_qk_t, s_qk_d),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_s = tl.dot(b_k, b_q, allow_tf32=False)
    p_f = f + i_bC * T + i_t * BT + tl.arange(0, BT)
    p_i = i + i_bC * T + i_t * BT + tl.arange(0, BT)
    b_f = tl.load(p_f)
    b_f_last = tl.load(f + i_bC * T + i_t * BT + BT - 1)
    b_m = tl.load(m + i_bC * (NT + 1) + i_t)
    b_m_total = tl.load(m_total + i_bC * T + i_t * BT + tl.arange(0, BT))
    b_norm = tl.load(norm + i_bC * T + i_t * BT + tl.arange(0, BT))
    b_i = tl.load(p_i)

    # TODO: update to stable version of Mamba2
    mask_f = b_f[None, :] - b_f[:, None]
    b_logDT = b_i[:, None] + mask_f - b_m_total[None, :]
    mask = tl.where(o_i[:, None] <= o_i[None, :], tl.math.exp2(b_logDT) * scale, 0)
    b_s = b_s * mask

    b_m_next = tl.load(m + i_bC * (NT + 1) + i_t + 1)

    b_dq = tl.zeros([BT, BK], dtype=b_q.dtype)
    b_dk = tl.zeros([BT, BK], dtype=b_q.dtype)
    b_ds = tl.zeros([BT, BT], dtype=b_q.dtype)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + i_bC * s_vh_h,
            (T, V),
            (s_vh_t, s_vh_d),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_C = tl.make_block_ptr(
            C + i_bC * s_C_h,
            (V, NT * K),
            (1, s_C_t),
            (i_v * BV, i_t * K + i_k * BK),
            (BV, BK),
            (0, 1),
        )
        p_dh = tl.make_block_ptr(
            dh + i_bC * s_vh_h,
            (T, V),
            (s_vh_t, s_vh_d),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        p_dC = tl.make_block_ptr(
            dC + i_bC * s_C_h,
            (NT * K, V),
            (s_C_t, 1),
            (i_t * K + i_k * BK, i_v * BV),
            (BK, BV),
            (1, 0),
        )
        p_dv = tl.make_block_ptr(
            dv + (i_k * n_bh + i_bC) * s_vh_h,
            (T, V),
            (s_vh_t, s_vh_d),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))
        # [BV, BK]
        b_C = tl.load(p_C, boundary_check=(0, 1))
        # [BK, BV]
        b_dC = tl.load(p_dC, boundary_check=(0, 1))
        # [BT, BT]
        b_ds += tl.dot(b_dh, tl.trans(b_v), allow_tf32=False).to(b_ds.dtype)
        # [BT, BK]
        b_dq += (tl.dot(b_dh, b_C.to(b_dh.dtype), allow_tf32=False) * scale).to(
            b_dq.dtype
        )
        b_dk += tl.dot(b_v, tl.trans(b_dC.to(b_v.dtype)), allow_tf32=False).to(
            b_dk.dtype
        )
        # [BT, BV]
        b_dv = tl.dot(b_k, b_dC.to(b_k.dtype), allow_tf32=False).to(
            b_q.dtype
        ) * tl.math.exp2(b_i - b_f + b_f_last - b_m_next)[:, None].to(b_q.dtype)
        b_dv += tl.dot(
            (b_s / b_norm[None, :].to(b_s.dtype)).to(b_q.dtype),
            b_dh.to(b_q.dtype),
            allow_tf32=False,
        ).to(b_q.dtype)

        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    b_dq *= (tl.math.exp2(b_f + b_m - b_m_total)[:, None] / b_norm[:, None]).to(
        b_dq.dtype
    )
    b_dk *= tl.math.exp2(b_i - b_f + b_f_last - b_m_next)[:, None].to(b_dk.dtype)

    b_ds = b_ds * tl.trans(mask)
    b_ds = b_ds.to(b_k.dtype)
    # [BT, BK]
    b_dq += tl.dot(b_ds, b_k, allow_tf32=False) / b_norm[:, None].to(b_q.dtype)
    b_dk += tl.trans(
        tl.dot(
            (b_q / b_norm[None, :].to(b_q.dtype)).to(b_q.dtype), b_ds, allow_tf32=False
        )
    )

    p_dq = tl.make_block_ptr(
        dq + i_bC * s_qk_h,
        (T, K),
        (s_qk_t, s_qk_d),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + i_bC * s_qk_h,
        (T, K),
        (s_qk_t, s_qk_d),
        (i_t * BT, i_k * BK),
        (BT, BK),
        (1, 0),
    )
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


class mLSTMKernelC(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    @contiguous
    def forward(
        ctx,
        k,
        v,
        C,
        n,
        m,
        i,
        f,
        initial_C,
        initial_n,
        initial_m,
        final_C,
        final_n,
        final_m,
    ):
        B, H, NT, BT, K, V = *k.shape, v.shape[-1]
        T = BT * NT
        BK, BV = min(64, triton.next_power_of_2(K)), min(64, triton.next_power_of_2(V))
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        scale = K**-0.5

        grid = (NK, NV, B * H)
        chunk_mlstm_fwd_kernel_C[grid](
            k,
            v,
            C,
            n,
            m,
            i,
            f,
            initial_C,
            initial_n,
            initial_m,
            final_C,
            final_n,
            final_m,
            k.stride(2),
            k.stride(3),
            k.stride(4),
            v.stride(2),
            v.stride(3),
            v.stride(4),
            C.stride(2),
            C.stride(3),
            n.stride(1),
            H=H,
            T=T,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            NT=NT,
            USE_INITIAL_STATE=initial_C is not None,
            STORE_FINAL_STATE=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    @staticmethod
    @custom_bwd(device_type="cuda")
    @contiguous
    def backward(ctx, dh, d_ht=None):
        pass


class mLSTMKernelH(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        C,
        n,
        m,
        m_total,
        i,
        f,
        h,
        norm,
    ):
        B, H, NT, BT, K, V = *k.shape, v.shape[-1]
        T = BT * NT
        BK, BV = min(64, triton.next_power_of_2(K)), min(64, triton.next_power_of_2(V))
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        scale = K**-0.5

        grid = (NV, NT, B * H)
        chunk_mlstm_fwd_kernel_h[grid](
            q,
            k,
            v,
            C,
            n,
            m,
            m_total,
            i,
            f,
            h,
            norm,
            q.stride(2),
            q.stride(3),
            q.stride(4),
            v.stride(2),
            v.stride(3),
            v.stride(4),
            C.stride(2),
            C.stride(3),
            n.stride(1),
            scale,
            H=H,
            T=T,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    @staticmethod
    @custom_bwd(device_type="cuda")
    @contiguous
    def backward(ctx, dh, d_ht=None):
        pass


class mLSTMKerneldC(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    @contiguous
    def forward(
        ctx,
        q,
        f,
        m,
        m_total,
        norm,
        dh,
        dC,
        final_dC,
        final_m,
        initial_dC,
        initial_m,
    ):
        B, H, NT, BT, K, V = *q.shape, dh.shape[-1]
        T = BT * NT
        BK, BV = min(64, triton.next_power_of_2(K)), min(64, triton.next_power_of_2(V))
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        scale = K**-0.5

        grid = (NK, NV, B * H)
        chunk_mlstm_bwd_kernel_dC[grid](
            q,
            f,
            m,
            m_total,
            norm,
            dh,
            dC,
            final_dC,
            final_m,
            initial_dC,
            initial_m,
            q.stride(2),
            q.stride(3),
            q.stride(4),
            dh.stride(2),
            dh.stride(3),
            dh.stride(4),
            dC.stride(2),
            dC.stride(3),
            scale=scale,
            H=H,
            T=T,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            NT=NT,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    @staticmethod
    @custom_bwd(device_type="cuda")
    @contiguous
    def backward(ctx, dh, d_ht=None):
        pass


class mLSTMKerneldqkv(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    @contiguous
    def forward(
        ctx,
        q,
        k,
        v,
        C,
        m,
        m_total,
        final_m,
        norm,
        i,
        f,
        dh,
        dC,
        dq,
        dk,
        dv,
    ):
        B, H, NT, BT, K, V = *q.shape, v.shape[-1]
        T = NT * BT
        BK, BV = (
            min(32 if q.dtype == torch.float32 else 64, triton.next_power_of_2(K)),
            min(32 if q.dtype == torch.float32 else 64, triton.next_power_of_2(V)),
        )
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        grid = (NK, NT, B * H)
        dv_internal = v.new_empty(NK, *v.shape)
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        scale = K**-0.5

        chunk_mlstm_bwd_kernel_dqkvif[grid](
            q,
            k,
            v,
            C,
            m,
            m_total,
            final_m,
            norm,
            i,
            f,
            dh,
            dC,
            dq,
            dk,
            dv_internal,
            q.stride(2),
            q.stride(3),
            q.stride(4),
            dh.stride(2),
            dh.stride(3),
            dh.stride(4),
            dC.stride(2),
            dC.stride(3),
            scale=scale,
            B=B,
            H=H,
            T=T,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            NT=NT,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        dv[:] = dv_internal.sum(0)

        def rev_cumsum(x):
            cumsum_x = x.cumsum(-1)
            rev_cumsum_x = cumsum_x[..., -1, None] - cumsum_x
            return rev_cumsum_x + x

    @staticmethod
    @custom_bwd(device_type="cuda")
    @contiguous
    def backward(ctx, dh, d_ht=None):
        pass


def mLSTMFunctionGenerator(
    chunk_size: int = 64,
    keep_states: bool = False,
    dtype_state: Optional[torch.dtype] = torch.float32,
    dtype_gate: Optional[torch.dtype] = torch.float32,
):
    class mLSTMFunction(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda")
        @contiguous
        def forward(
            ctx, q, k, v, i, f, initial_C, initial_n, initial_m, output_final_state
        ):
            B, H, T, K, V = *q.shape, v.shape[-1]
            BT = chunk_size
            BK, BV = (
                min(64, triton.next_power_of_2(K)),
                min(64, triton.next_power_of_2(V)),
            )
            NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
            num_stages = 1
            num_warps = 4 if BK == 64 else 2
            scale = K**-0.5

            if dtype_state is None:
                dtype_states = q.dtype
            else:
                dtype_states = dtype_state
            if dtype_gate is None:
                dtype_gates = q.dtype
            else:
                dtype_gates = dtype_gate

            assert T % BT == 0, "sequence length must be divisible by BT"
            f_orig = f
            f = torch.nn.functional.logsigmoid(f.to(dtype_gates))
            f = f.reshape(B, H, -1, BT)
            f = f.cumsum(-1) * 1.44269504
            f = f.reshape(B, H, -1)
            i = (i.reshape(B, H, -1) * 1.44269504).to(dtype_gates)

            final_C, final_n, final_m = None, None, None
            if output_final_state:
                final_C = q.new_empty(B, H, K, V, requires_grad=False)
                final_n = q.new_empty(B, H, K, requires_grad=False)
                final_m = q.new_empty(B, H, requires_grad=False)

            C = q.new_empty(B, H, NT * K, V, dtype=dtype_states)
            n = q.new_empty(B, H, NT, K, dtype=dtype_states)
            m = q.new_full((B, H, NT + 1), float("nan"), dtype=dtype_states)
            m_total = q.new_empty(B, H, NT, BT, dtype=dtype_states)
            norm = q.new_empty(B, H, NT, BT, dtype=dtype_states)
            grid = (NK, NV, B * H)
            chunk_mlstm_fwd_kernel_C[grid](
                k,
                v,
                C,
                n,
                m,
                i,
                f,
                initial_C,
                initial_n,
                initial_m,
                final_C,
                final_n,
                final_m,
                q.stride(1),
                q.stride(2),
                q.stride(3),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                C.stride(1),
                C.stride(2),
                n.stride(1),
                H=H,
                T=T,
                K=K,
                V=V,
                BT=BT,
                BK=BK,
                BV=BV,
                NT=NT,
                USE_INITIAL_STATE=initial_C is not None,
                STORE_FINAL_STATE=output_final_state,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            grid = (NV, NT, B * H)
            h = torch.empty_like(v)

            chunk_mlstm_fwd_kernel_h[grid](
                q,
                k,
                v,
                C,
                n,
                m,
                m_total,
                i,
                f,
                h,
                norm,
                q.stride(1),
                q.stride(2),
                q.stride(3),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                C.stride(1),
                C.stride(2),
                n.stride(1),
                scale,
                H=H,
                T=T,
                K=K,
                V=V,
                BT=BT,
                BK=BK,
                BV=BV,
                NT=NT,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            if keep_states:
                ctx.save_for_backward(
                    q,
                    k,
                    v,
                    C,
                    f,
                    i,
                    m,
                    m_total,
                    norm,
                    final_m,
                    initial_C,
                    initial_n,
                    initial_m,
                    f_orig,
                )
            else:
                ctx.save_for_backward(
                    q,
                    k,
                    v,
                    None,
                    f,
                    i,
                    m,
                    m_total,
                    norm,
                    final_m,
                    initial_C,
                    initial_n,
                    initial_m,
                    f_orig,
                )
            return h.to(q.dtype), final_C, final_n, final_m

        @staticmethod
        @custom_bwd(device_type="cuda")
        @contiguous
        def backward(ctx, dh, final_dC=None, final_dn=None, final_dm=None):
            (
                q,
                k,
                v,
                C,
                f,
                i,
                m,
                m_total,
                norm,
                final_m,
                initial_C,
                initial_n,
                initial_m,
                f_orig,
            ) = ctx.saved_tensors

            B, H, T, K, V = *q.shape, v.shape[-1]
            BT = chunk_size
            BK, BV = (
                min(32 if q.dtype == torch.float32 else 64, triton.next_power_of_2(K)),
                min(32 if q.dtype == torch.float32 else 64, triton.next_power_of_2(V)),
            )
            NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)

            if dtype_state is None:
                dtype_states = q.dtype
            else:
                dtype_states = dtype_state

            if C is None:
                num_stages = 1
                num_warps = 4 if BK == 64 else 2
                scale = K**-0.5

                C = q.new_empty(B, H, NT * K, V, dtype=dtype_states)
                n = q.new_empty(B, H, NT, K, dtype=dtype_states)
                grid = (NK, NV, B * H)
                final_C, final_n, final_m = None, None, None

                chunk_mlstm_fwd_kernel_C[grid](
                    k,
                    v,
                    C,
                    n,
                    m,
                    i,
                    f,
                    initial_C,
                    initial_n,
                    initial_m,
                    final_C,
                    final_n,
                    final_m,
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    v.stride(1),
                    v.stride(2),
                    v.stride(3),
                    C.stride(1),
                    C.stride(2),
                    n.stride(1),
                    H=H,
                    T=T,
                    K=K,
                    V=V,
                    BT=BT,
                    BK=BK,
                    BV=BV,
                    NT=NT,
                    USE_INITIAL_STATE=initial_C is not None,
                    STORE_FINAL_STATE=False,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

            num_stages = 1
            num_warps = 4 if BK == 64 else 2
            scale = K**-0.5
            dC = q.new_empty(B, H, NT * K, V, dtype=dtype_states)

            initial_dC = q.new_empty(
                B, H, K, V, requires_grad=False, dtype=dtype_states
            )
            initial_m = q.new_empty(B, H, requires_grad=False, dtype=dtype_states)

            if final_dC is None:
                final_dC = q.new_full(initial_dC.shape, 0.0, dtype=dtype_states)
                final_m = q.new_full(initial_m.shape, 0.0, dtype=dtype_states)
            else:
                final_dC = final_dC.to(dtype_states)
                final_m = final_m.to(dtype_states)

            grid = (NK, NV, B * H)
            chunk_mlstm_bwd_kernel_dC[grid](
                q,
                f,
                m,
                m_total,
                norm,
                dh,
                dC,
                final_dC,
                final_m,
                initial_dC,
                initial_m,
                q.stride(1),
                q.stride(2),
                q.stride(3),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                dC.stride(1),
                dC.stride(2),
                scale,
                H=H,
                T=T,
                K=K,
                V=V,
                BT=BT,
                BK=BK,
                BV=BV,
                NT=NT,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            grid = (NK, NT, B * H)
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = v.new_empty(NK, *v.shape)
            num_stages = 1
            num_warps = 4 if BK == 64 else 2
            chunk_mlstm_bwd_kernel_dqkvif[grid](
                q,
                k,
                v,
                C,
                m,
                m_total,
                final_m,
                norm,
                i,
                f,
                dh,
                dC,
                dq,
                dk,
                dv,
                q.stride(1),
                q.stride(2),
                q.stride(3),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                dC.stride(1),
                dC.stride(2),
                scale,
                B=B,
                H=H,
                T=T,
                K=K,
                V=V,
                BT=BT,
                BK=BK,
                BV=BV,
                NT=NT,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            def rev_cumsum(x):
                return x.flip(dims=(-1,)).cumsum(-1).flip(dims=(-1,))

            dv = dv.sum(0)
            df = (dq * q - dk * k).sum(-1)
            di = (dv * v).sum(-1)

            df = rev_cumsum(df)
            df = df * torch.nn.functional.sigmoid(-f_orig)

            return (
                dq.to(q.dtype),
                dk.to(k.dtype),
                dv.to(v.dtype),
                di.to(f_orig.dtype),
                df.to(f_orig.dtype).view(f.shape),
                initial_dC.to(initial_C.dtype) if initial_C is not None else None,
                None,
                None,
                None,
            )

    return mLSTMFunction


mLSTMFunction = {}
# registry with (chunk_size, keep_state, dtype_states, dtype_gates)
mLSTMFunction[(64, False, "float32", "float32")] = mLSTMFunctionGenerator(
    chunk_size=64, keep_states=False
)
mLSTMFunction[(64, False, "bfloat16", "float32")] = mLSTMFunctionGenerator(
    chunk_size=64, keep_states=False
)
mLSTMFunction[(64, False, "float16", "float32")] = mLSTMFunctionGenerator(
    chunk_size=64, keep_states=False
)

DTYPESTR_TO_DTYPE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def mlstm_fwbw(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,  # input gate
    f: torch.Tensor,  # forget gate
    initial_C: torch.Tensor = None,
    initial_n: torch.Tensor = None,
    initial_m: torch.Tensor = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    keep_states: bool = False,
    dtype_states: Literal["float32", "bfloat16", "float16"] = "float32",
    dtype_gates: Literal["float32", "bfloat16", "float16"] = "float32",
) -> tuple[torch.Tensor, torch.Tensor]:
    # actually dtype_gates is not really supported yet
    f = f.float()
    i = i.float()
    if (chunk_size, keep_states, dtype_states, dtype_gates) not in mLSTMFunction:
        mLSTMFunction[(chunk_size, keep_states, dtype_states, dtype_gates)] = (
            mLSTMFunctionGenerator(
                chunk_size=chunk_size,
                keep_states=keep_states,
                dtype_state=DTYPESTR_TO_DTYPE[dtype_states],
                dtype_gate=DTYPESTR_TO_DTYPE[dtype_gates],
            )
        )
    mLSTMFunc = mLSTMFunction[(chunk_size, keep_states, dtype_states, dtype_gates)]
    h, final_C, final_n, final_m = mLSTMFunc.apply(
        q, k, v, i, f, initial_C, initial_n, initial_m, output_final_state
    )
    if output_final_state:
        return h, (final_C, final_n, final_m)
    else:
        return h


@dataclass
class mLSTMBackendTritonConfig:
    chunk_size: int = 64
    save_states: bool = False
    dtype_states: Literal["float32", "bfloat16", "float16"] = "float32"
    dtype_gates: Literal["float32", "bfloat16", "float16"] = "float32"

    def assign_model_config_params(self, model_config, *args, **kwargs):
        pass


class mLSTMBackendTriton(torch.nn.Module):
    config_class = mLSTMBackendTritonConfig

    def __init__(self, config: mLSTMBackendTritonConfig):
        super().__init__(config)
        self._func = mLSTMFunctionGenerator(
            config.chunk_size,
            config.save_states,
            dtype_state=config.dtype_states,
            dtype_gate=config.dtype_gates,
        )

    def forward(
        self,
        q,
        k,
        v,
        i,
        f,
        initial_C=None,
        initial_n=None,
        initial_m=None,
        output_final_state: bool = False,
    ):
        h, final_C, final_n, final_m = self._func.apply(
            q, k, v, i, f, initial_C, initial_n, initial_m, output_final_state
        )
        if output_final_state:
            return h, (final_C, final_n, final_m)
        else:
            return h
