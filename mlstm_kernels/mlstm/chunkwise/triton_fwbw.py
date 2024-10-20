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
    matK,
    matV,
    matC,
    matN,
    matM,
    vecI,  # log igates
    vecF,  # accumulated log fgate
    matC_initial,  # initial state of the chunk [B, H, D_head_K, D_head_V]
    matN_initial,
    matM_initial,
    matC_final,  # final state of the chunk [B, H, D_head_K, D_head_V]
    matN_final,
    matM_final,
    str_QK_H,
    str_QK_t,
    str_QK_d,
    str_VH_H,
    str_VH_t,
    str_VH_d,
    str_C_H,
    str_C_t,
    str_N_H,
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
    idx_K, idx_V, idx_BC = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if USE_INITIAL_STATE:
        matC0_ptr = tl.make_block_ptr(
            matC_initial + idx_BC * K * V,
            (K, V),
            (V, 1),
            (idx_K * BK, idx_V * BV),
            (BK, BV),
            (1, 0),
        )
        matN0_ptr = tl.make_block_ptr(
            matN_initial + idx_BC * K,
            (K,),
            (1,),
            (idx_K * BK,),
            (BK,),
            (0,),
        )
        matM0_ptr = matM_initial + idx_BC

        matC_val = tl.load(matC0_ptr, boundary_check=(0, 1))
        matN_val = tl.load(matN0_ptr, boundary_check=(0,))
        matM_val = tl.load(matM0_ptr)
    else:
        matC_val = tl.zeros([BK, BV], dtype=tl.load(matC).dtype)
        matN_val = tl.zeros([BK], dtype=matC_val.dtype)
        matM_val = 0.0

    matM_next_val = 0.0
    for idx_t in range(NT):
        matK_ptr = tl.make_block_ptr(
            matK + idx_BC * str_QK_H,
            (K, T),
            (str_QK_d, str_QK_t),
            (idx_K * BK, idx_t * BT),
            (BK, BT),
            (0, 1),
        )
        matV_ptr = tl.make_block_ptr(
            matV + idx_BC * str_VH_H,
            (T, V),
            (str_VH_t, str_VH_d),
            (idx_t * BT, idx_V * BV),
            (BT, BV),
            (1, 0),
        )
        matC_ptr = tl.make_block_ptr(
            matC + idx_BC * str_C_H + idx_t * K * V,
            (K, V),
            (str_C_t, 1),
            (idx_K * BK, idx_V * BV),
            (BK, BV),
            (1, 0),
        )
        matN_ptr = tl.make_block_ptr(
            matN + idx_BC * str_N_H + idx_t * K, (K,), (1,), (idx_K * BK,), (BK,), (0,)
        )
        tl.store(
            matC_ptr, matC_val.to(matC_ptr.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(matN_ptr, matN_val.to(matN_ptr.dtype.element_ty), boundary_check=(0,))
        tl.store(matM + idx_BC * (NT + 1) + idx_t, matM_val)
        # [BK, BT]
        matK_val = tl.load(matK_ptr, boundary_check=(0, 1))
        # [BT, BV]
        matV_val = tl.load(matV_ptr, boundary_check=(0, 1))
        # [BK, BV]
        scaF_last_val = tl.load(vecF + idx_BC * T + idx_t * BT + BT - 1)
        vecF_val = tl.load(vecF + idx_BC * T + idx_t * BT + tl.arange(0, BT))
        vecI_val = tl.load(vecI + idx_BC * T + idx_t * BT + tl.arange(0, BT))
        vecG_val = vecI_val + scaF_last_val - vecF_val

        matM_next_val, _ = tl.max(vecG_val)
        matM_next_val = tl.maximum(scaF_last_val + matM_val, matM_next_val)

        matC_val *= tl.math.exp2(scaF_last_val - matM_next_val + matM_val).to(
            matC_val.dtype
        )
        matN_val *= tl.math.exp2(scaF_last_val - matM_next_val + matM_val).to(
            matN_val.dtype
        )
        matC_val += tl.dot(
            matK_val,
            matV_val
            * (tl.math.exp2(vecG_val - matM_next_val)[:, None]).to(matK_val.dtype),
            allow_tf32=False,
        ).to(matC_val.dtype)
        matN_val += tl.sum(
            matK_val * tl.math.exp2(vecG_val - matM_next_val).to(matK_val.dtype), axis=1
        ).to(matN_val.dtype)
        matM_val = matM_next_val

    tl.store(matM + idx_BC * (NT + 1) + NT, matM_val)
    if STORE_FINAL_STATE:
        matC_ptr = tl.make_block_ptr(
            matC_final + idx_BC * K * V,
            (K, V),
            (V, 1),
            (idx_K * BK, idx_V * BV),
            (BK, BV),
            (1, 0),
        )
        matN_ptr = tl.make_block_ptr(
            matN_final + idx_BC * K, (K,), (1,), (idx_K * BK,), (BK,), (0,)
        )
        tl.store(
            matC_ptr, matC_val.to(matC_ptr.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(matN_ptr, matN_val.to(matN_ptr.dtype.element_ty), boundary_check=(0,))
        tl.store(matM_final + idx_BC, matM_val)


@triton.jit
def chunk_mlstm_fwd_kernel_h(
    matQ,
    matK,
    matV,
    matC,
    matN,
    matM,
    matM_total,
    vecI,
    vecF,
    matH,
    vecNorm,
    str_QK_H,
    str_QK_t,
    str_QK_d,
    str_VH_H,
    str_VH_t,
    str_VH_d,
    str_C_H,
    str_C_t,
    str_N_H,
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
    idx_V, idx_t, idx_BC = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    idx_t_cta = tl.arange(0, BT)
    matM_s = idx_t_cta[:, None] >= idx_t_cta[None, :]

    matH_val = tl.zeros([BT, BV], dtype=tl.load(matQ).dtype)
    matS_val = tl.zeros([BT, BT], dtype=matH_val.dtype)
    vecNorm_val = tl.zeros([BT, BV], dtype=matH_val.dtype)
    for idx_K in range(tl.cdiv(K, BK)):
        matQ_ptr = tl.make_block_ptr(
            matQ + idx_BC * str_QK_H,
            (T, K),
            (str_QK_t, str_QK_d),
            (idx_t * BT, idx_K * BK),
            (BT, BK),
            (1, 0),
        )
        matK_ptr = tl.make_block_ptr(
            matK + idx_BC * str_QK_H,
            (K, T),
            (str_QK_d, str_QK_t),
            (idx_K * BK, idx_t * BT),
            (BK, BT),
            (0, 1),
        )
        matC_ptr = tl.make_block_ptr(
            matC + idx_BC * str_C_H + idx_t * K * V,
            (K, V),
            (str_C_t, 1),
            (idx_K * BK, idx_V * BV),
            (BK, BV),
            (1, 0),
        )
        matN_ptr = tl.make_block_ptr(
            matN + idx_BC * str_N_H + idx_t * K,
            (K, BV),
            (1, 0),
            (idx_K * BK, 0),
            (BK, BV),
            (0, 1),
        )

        # [BT, BK]
        matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1))
        # [BK, BT]
        matK_val = tl.load(matK_ptr, boundary_check=(0, 1))
        # [BT]

        # [BK, BV]
        matC_val = tl.load(matC_ptr, boundary_check=(0, 1))
        matN_val = tl.load(matN_ptr, boundary_check=(0,))
        matH_val += tl.dot(matQ_val, matC_val.to(matQ_val.dtype), allow_tf32=False).to(
            matH_val.dtype
        )
        matS_val += tl.dot(matQ_val, matK_val, allow_tf32=False).to(matS_val.dtype)
        vecN2_val = tl.dot(matQ_val, matN_val.to(matQ_val.dtype), allow_tf32=False).to(
            vecNorm_val.dtype
        )
        vecNorm_val += vecN2_val

    vecF_ptr = vecF + idx_BC * T + idx_t * BT + tl.arange(0, BT)
    vecF_val = tl.load(vecF_ptr)
    vecI_ptr = vecI + idx_BC * T + idx_t * BT + tl.arange(0, BT)
    vecI_val = tl.load(vecI_ptr)
    matM_val = tl.load(matM + idx_BC * (NT + 1) + idx_t)

    # TODO revise this to the stabilized version
    matlogD_val = vecI_val[None, :] + vecF_val[:, None] - vecF_val[None, :]
    matlogD_val = tl.where(matM_s, matlogD_val, -float("inf"))
    matmlogD_val = tl.max(matlogD_val, axis=1)

    matM_total_val = tl.maximum(vecF_val + matM_val, matmlogD_val)
    matM_total_ptr = tl.make_block_ptr(
        matM_total + T * idx_BC, (T,), (1,), (idx_t * BT,), (BT,), (0,)
    )
    tl.store(
        matM_total_ptr,
        matM_total_val.to(matM_total_ptr.dtype.element_ty),
        boundary_check=(0,),
    )

    matD_val = tl.math.exp2(matlogD_val - matM_total_val[:, None])
    matH_val = (
        matH_val * tl.math.exp2(vecF_val + matM_val - matM_total_val)[:, None] * scale
    )
    matS_val = matS_val * matD_val * scale
    vecNorm_val = (
        vecNorm_val
        * tl.math.exp2(vecF_val + matM_val - matM_total_val)[:, None]
        * scale
    )

    matS_val = tl.where(matM_s, matS_val, 0)
    vecNorm_val += tl.sum(matS_val, axis=1)[:, None]
    vecNorm_val = tl.abs(vecNorm_val)

    vecNorm_val = tl.maximum(
        vecNorm_val.to(matM_total_val.dtype), tl.math.exp2(-matM_total_val)[:, None]
    )

    tl.store(
        vecNorm + idx_BC * T + idx_t * BT + tl.arange(0, BT),
        tl.max(vecNorm_val, axis=1),
    )

    matV_ptr = tl.make_block_ptr(
        matV + idx_BC * str_VH_H,
        (T, V),
        (str_VH_t, str_VH_d),
        (idx_t * BT, idx_V * BV),
        (BT, BV),
        (1, 0),
    )
    matV_val = tl.load(matV_ptr, boundary_check=(0, 1))
    matH_val = (
        matH_val + tl.dot(matS_val.to(matV_val.dtype), matV_val, allow_tf32=False)
    ) / vecNorm_val.to(matH_val.dtype)
    matH_ptr = tl.make_block_ptr(
        matH + idx_BC * str_VH_H,
        (T, V),
        (str_VH_t, str_VH_d),
        (idx_t * BT, idx_V * BV),
        (BT, BV),
        (1, 0),
    )
    tl.store(matH_ptr, matH_val.to(matH_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_mlstm_bwd_kernel_dC(
    matQ,
    vecF,
    matM,
    matM_total,
    vecNorm,
    matdH,
    matdC,
    matdC_final,
    matM_final,
    matdC_initial,
    matM_initial,
    str_QK_H,
    str_QK_t,
    str_QK_d,
    str_VH_H,
    str_VH_t,
    str_VH_d,
    str_C_H,
    str_C_t,
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
    idx_K, idx_V, idx_BC = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BK, BV]
    matdC_ptr = tl.make_block_ptr(
        matdC_final + idx_BC * K * V,
        (K, V),
        (str_C_t, 1),
        (idx_K * BK, idx_V * BV),
        (BK, BV),
        (1, 0),
    )
    matdC_val = tl.load(matdC_ptr, boundary_check=(0, 1))
    matM_val = tl.load(matM_final + idx_BC)
    for idx_t in range(NT - 1, -1, -1):
        matQ_ptr = tl.make_block_ptr(
            matQ + idx_BC * str_QK_H,
            (K, T),
            (str_QK_d, str_QK_t),
            (idx_K * BK, idx_t * BT),
            (BK, BT),
            (0, 1),
        )
        matdH_ptr = tl.make_block_ptr(
            matdH + idx_BC * str_VH_H,
            (T, V),
            (str_VH_t, str_VH_d),
            (idx_t * BT, idx_V * BV),
            (BT, BV),
            (1, 0),
        )
        matdC_ptr = tl.make_block_ptr(
            matdC + idx_BC * str_C_H + idx_t * K * V,
            (K, V),
            (str_C_t, 1),
            (idx_K * BK, idx_V * BV),
            (BK, BV),
            (1, 0),
        )

        tl.store(
            matdC_ptr, matdC_val.to(matdC_ptr.dtype.element_ty), boundary_check=(0, 1)
        )
        scaF_last_val = tl.load(vecF + idx_BC * T + idx_t * BT + BT - 1)
        vecF_val = tl.load(vecF + idx_BC * T + idx_t * BT + tl.arange(0, BT))
        matM_p_val = tl.load(matM + idx_BC * (NT + 1) + idx_t)
        matM_total_val = tl.load(
            matM_total + idx_BC * T + idx_t * BT + tl.arange(0, BT)
        )
        vecNorm_val = tl.load(vecNorm + idx_BC * T + idx_t * BT + tl.arange(0, BT))

        # [BK, BT]
        matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1))
        matQ_val = (
            matQ_val
            * scale
            * tl.math.exp2(vecF_val + matM_p_val - matM_total_val)[None, :]
        ).to(matQ_val.dtype)
        # [BT, V]
        matdH_val = tl.load(matdH_ptr, boundary_check=(0, 1))
        matdH_val /= vecNorm_val[:, None].to(matdH_val.dtype)
        # [BK, BV]
        matdC_val *= tl.math.exp2(scaF_last_val + matM_p_val - matM_val).to(
            matdC_val.dtype
        )
        matdC_val += tl.dot(
            matQ_val, matdH_val.to(matQ_val.dtype), allow_tf32=False
        ).to(matdC_val.dtype)
        matM_val = matM_p_val

    matdC_initial_ptr = tl.make_block_ptr(
        matdC_initial + idx_BC * K * V,
        (K, V),
        (V, 1),
        (idx_K * BK, idx_V * BV),
        (BK, BV),
        (1, 0),
    )
    tl.store(
        matdC_initial_ptr,
        matdC_val.to(matdC_initial_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(matM_initial + idx_BC, matM_val)


@triton.jit
def chunk_mlstm_bwd_kernel_dqkvif(
    matQ,
    matK,
    matV,
    matC,
    matM,
    matM_total,
    matM_final,
    vecNorm,
    vecI,
    vecF,
    matdH,
    matdC,
    matdQ,
    matdK,
    matdV,
    str_QK_H,
    str_QK_t,
    str_QK_d,
    str_VH_H,
    str_VH_t,
    str_VH_d,
    str_C_H,
    str_C_t,
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
    idx_K, idx_t, idx_BC = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    matN_bh = tl.num_programs(2)
    o_cta = tl.arange(0, BT)

    matQ_ptr = tl.make_block_ptr(
        matQ + idx_BC * str_QK_H,
        (K, T),
        (str_QK_d, str_QK_t),
        (idx_K * BK, idx_t * BT),
        (BK, BT),
        (0, 1),
    )
    matK_ptr = tl.make_block_ptr(
        matK + idx_BC * str_QK_H,
        (T, K),
        (str_QK_t, str_QK_d),
        (idx_t * BT, idx_K * BK),
        (BT, BK),
        (1, 0),
    )

    matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1))
    matK_val = tl.load(matK_ptr, boundary_check=(0, 1))
    matS_val = tl.dot(matK_val, matQ_val, allow_tf32=False)
    vecF_ptr = vecF + idx_BC * T + idx_t * BT + tl.arange(0, BT)
    vecI_ptr = vecI + idx_BC * T + idx_t * BT + tl.arange(0, BT)
    vecF_val = tl.load(vecF_ptr)
    scaF_last_val = tl.load(vecF + idx_BC * T + idx_t * BT + BT - 1)
    matM_val = tl.load(matM + idx_BC * (NT + 1) + idx_t)
    matM_total_val = tl.load(matM_total + idx_BC * T + idx_t * BT + tl.arange(0, BT))
    vecNorm_val = tl.load(vecNorm + idx_BC * T + idx_t * BT + tl.arange(0, BT))
    vecI_val = tl.load(vecI_ptr)

    # TODO: update to stable version of Mamba2
    mask_f = vecF_val[None, :] - vecF_val[:, None]
    logDT_val = vecI_val[:, None] + mask_f - matM_total_val[None, :]
    mask = tl.where(
        o_cta[:, None] <= o_cta[None, :], tl.math.exp2(logDT_val) * scale, 0
    )
    matS_val = matS_val * mask

    matM_next_val = tl.load(matM + idx_BC * (NT + 1) + idx_t + 1)

    matdQ_val = tl.zeros([BT, BK], dtype=matQ_val.dtype)
    matdK_val = tl.zeros([BT, BK], dtype=matQ_val.dtype)
    matdS_val = tl.zeros([BT, BT], dtype=matQ_val.dtype)
    for idx_V in range(tl.cdiv(V, BV)):
        matV_ptr = tl.make_block_ptr(
            matV + idx_BC * str_VH_H,
            (T, V),
            (str_VH_t, str_VH_d),
            (idx_t * BT, idx_V * BV),
            (BT, BV),
            (1, 0),
        )
        matC_ptr = tl.make_block_ptr(
            matC + idx_BC * str_C_H,
            (V, NT * K),
            (1, str_C_t),
            (idx_V * BV, idx_t * K + idx_K * BK),
            (BV, BK),
            (0, 1),
        )
        matdH_ptr = tl.make_block_ptr(
            matdH + idx_BC * str_VH_H,
            (T, V),
            (str_VH_t, str_VH_d),
            (idx_t * BT, idx_V * BV),
            (BT, BV),
            (1, 0),
        )
        matdC_ptr = tl.make_block_ptr(
            matdC + idx_BC * str_C_H,
            (NT * K, V),
            (str_C_t, 1),
            (idx_t * K + idx_K * BK, idx_V * BV),
            (BK, BV),
            (1, 0),
        )
        matdV_ptr = tl.make_block_ptr(
            matdV + (idx_K * matN_bh + idx_BC) * str_VH_H,
            (T, V),
            (str_VH_t, str_VH_d),
            (idx_t * BT, idx_V * BV),
            (BT, BV),
            (1, 0),
        )
        # [BT, BV]
        matV_val = tl.load(matV_ptr, boundary_check=(0, 1))
        matdH_val = tl.load(matdH_ptr, boundary_check=(0, 1))
        # [BV, BK]
        matC_val = tl.load(matC_ptr, boundary_check=(0, 1))
        # [BK, BV]
        matdC_val = tl.load(matdC_ptr, boundary_check=(0, 1))
        # [BT, BT]
        matdS_val += tl.dot(matdH_val, tl.trans(matV_val), allow_tf32=False).to(
            matdS_val.dtype
        )
        # [BT, BK]
        matdQ_val += (
            tl.dot(matdH_val, matC_val.to(matdH_val.dtype), allow_tf32=False) * scale
        ).to(matdQ_val.dtype)
        matdK_val += tl.dot(
            matV_val, tl.trans(matdC_val.to(matV_val.dtype)), allow_tf32=False
        ).to(matdK_val.dtype)
        # [BT, BV]
        matdV_val = tl.dot(matK_val, matdC_val.to(matK_val.dtype), allow_tf32=False).to(
            matQ_val.dtype
        ) * tl.math.exp2(vecI_val - vecF_val + scaF_last_val - matM_next_val)[
            :, None
        ].to(matQ_val.dtype)
        matdV_val += tl.dot(
            (matS_val / vecNorm_val[None, :].to(matS_val.dtype)).to(matQ_val.dtype),
            matdH_val.to(matQ_val.dtype),
            allow_tf32=False,
        ).to(matQ_val.dtype)

        tl.store(
            matdV_ptr, matdV_val.to(matdV_ptr.dtype.element_ty), boundary_check=(0, 1)
        )

    matdQ_val *= (
        tl.math.exp2(vecF_val + matM_val - matM_total_val)[:, None]
        / vecNorm_val[:, None]
    ).to(matdQ_val.dtype)
    matdK_val *= tl.math.exp2(vecI_val - vecF_val + scaF_last_val - matM_next_val)[
        :, None
    ].to(matdK_val.dtype)

    matdS_val = matdS_val * tl.trans(mask)
    matdS_val = matdS_val.to(matK_val.dtype)
    # [BT, BK]
    matdQ_val += tl.dot(matdS_val, matK_val, allow_tf32=False) / vecNorm_val[
        :, None
    ].to(matQ_val.dtype)
    matdK_val += tl.trans(
        tl.dot(
            (matQ_val / vecNorm_val[None, :].to(matQ_val.dtype)).to(matQ_val.dtype),
            matdS_val,
            allow_tf32=False,
        )
    )

    matdQ_ptr = tl.make_block_ptr(
        matdQ + idx_BC * str_QK_H,
        (T, K),
        (str_QK_t, str_QK_d),
        (idx_t * BT, idx_K * BK),
        (BT, BK),
        (1, 0),
    )
    matdK_ptr = tl.make_block_ptr(
        matdK + idx_BC * str_QK_H,
        (T, K),
        (str_QK_t, str_QK_d),
        (idx_t * BT, idx_K * BK),
        (BT, BK),
        (1, 0),
    )
    tl.store(matdQ_ptr, matdQ_val.to(matdQ_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(matdK_ptr, matdK_val.to(matdK_ptr.dtype.element_ty), boundary_check=(0, 1))


class mLSTMKernelC(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    @contiguous
    def forward(
        ctx,
        matK,
        matV,
        matC,
        matN,
        matM,
        vecI,
        vecF,
        matC_initial,
        matN_initial,
        matM_initial,
        matC_final,
        matN_final,
        matM_final,
    ):
        B, H, NT, BT, K, V = *k.shape, matV.shape[-1]
        T = BT * NT
        BK, BV = min(64, triton.next_power_of_2(K)), min(64, triton.next_power_of_2(V))
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        scale = K**-0.5

        grid = (NK, NV, B * H)
        chunk_mlstm_fwd_kernel_C[grid](
            matK,
            matV,
            matC,
            matN,
            matM,
            vecI,
            vecF,
            matC_initial,
            matN_initial,
            matM_initial,
            matC_final,
            matN_final,
            matM_final,
            matK.stride(2),
            matK.stride(3),
            matK.stride(4),
            matV.stride(2),
            matV.stride(3),
            matV.stride(4),
            matC.stride(2),
            matC.stride(3),
            matN.stride(1),
            H=H,
            T=T,
            K=K,
            V=V,
            BT=BT,
            BK=BK,
            BV=BV,
            NT=NT,
            USE_INITIAL_STATE=matC_initial is not None,
            STORE_FINAL_STATE=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    @staticmethod
    @custom_bwd(device_type="cuda")
    @contiguous
    def backward(ctx, matdH, d_ht=None):
        pass


class mLSTMKernelH(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    @contiguous
    def forward(
        ctx,
        matQ,
        matK,
        matV,
        matC,
        matN,
        matM,
        matM_total,
        vecI,
        vecF,
        matH,
        vecNorm,
    ):
        B, H, NT, BT, K, V = *k.shape, matV.shape[-1]
        T = BT * NT
        BK, BV = min(64, triton.next_power_of_2(K)), min(64, triton.next_power_of_2(V))
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        scale = K**-0.5

        grid = (NV, NT, B * H)
        chunk_mlstm_fwd_kernel_h[grid](
            matQ,
            matK,
            matV,
            matC,
            matN,
            matM,
            matM_total,
            vecI,
            vecF,
            matH,
            vecNorm,
            matQ.stride(2),
            matQ.stride(3),
            matQ.stride(4),
            matV.stride(2),
            matV.stride(3),
            matV.stride(4),
            matC.stride(2),
            matC.stride(3),
            matN.stride(1),
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
    def backward(ctx, matdH, d_ht=None):
        pass


class mLSTMKerneldC(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    @contiguous
    def forward(
        ctx,
        matQ,
        vecF,
        matM,
        matM_total,
        vecNorm,
        matdH,
        matdC,
        matdC_final,
        matM_final,
        matdC_initial,
        matM_initial,
    ):
        B, H, NT, BT, K, V = *matQ.shape, matdH.shape[-1]
        T = BT * NT
        BK, BV = min(64, triton.next_power_of_2(K)), min(64, triton.next_power_of_2(V))
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        scale = K**-0.5

        grid = (NK, NV, B * H)
        chunk_mlstm_bwd_kernel_dC[grid](
            matQ,
            vecF,
            matM,
            matM_total,
            vecNorm,
            matdH,
            matdC,
            matdC_final,
            matM_final,
            matdC_initial,
            matM_initial,
            matQ.stride(2),
            matQ.stride(3),
            matQ.stride(4),
            matdH.stride(2),
            matdH.stride(3),
            matdH.stride(4),
            matdC.stride(2),
            matdC.stride(3),
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
    def backward(ctx, matdH, d_ht=None):
        pass


class mLSTMKerneldqkv(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    @contiguous
    def forward(
        ctx,
        matQ,
        matK,
        matV,
        matC,
        matM,
        matM_total,
        matM_final,
        vecNorm,
        vecI,
        vecF,
        matdH,
        matdC,
        matdQ,
        matdK,
        matdV,
    ):
        B, H, NT, BT, K, V = *matQ.shape, matV.shape[-1]
        T = NT * BT
        BK, BV = (
            min(32 if matQ.dtype == torch.float32 else 64, triton.next_power_of_2(K)),
            min(32 if matQ.dtype == torch.float32 else 64, triton.next_power_of_2(V)),
        )
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        grid = (NK, NT, B * H)
        matdV_internal = matV.new_full((NK, *matV.shape), float("nan"))
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        scale = K**-0.5

        chunk_mlstm_bwd_kernel_dqkvif[grid](
            matQ,
            matK,
            matV,
            matC,
            matM,
            matM_total,
            matM_final,
            vecNorm,
            vecI,
            vecF,
            matdH,
            matdC,
            matdQ,
            matdK,
            matdV_internal,
            matQ.stride(2),
            matQ.stride(3),
            matQ.stride(4),
            matdH.stride(2),
            matdH.stride(3),
            matdH.stride(4),
            matdC.stride(2),
            matdC.stride(3),
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

        matdV[:] = matdV_internal.sum(0)

        def rev_cumsum(x):
            cumsum_x = x.cumsum(-1)
            rev_cumsum_x = cumsum_x[..., -1, None] - cumsum_x
            return rev_cumsum_x + x

    @staticmethod
    @custom_bwd(device_type="cuda")
    @contiguous
    def backward(ctx, matdH, d_ht=None):
        pass


def mLSTMFunctionGenerator(
    chunk_size: int = 64,
    keep_states: bool = False,
    dtype_state: torch.dtype | None = torch.float32,
    dtype_gate: torch.dtype | None = torch.float32,
):
    class mLSTMFunction(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda")
        @contiguous
        def forward(
            ctx,
            matQ,
            matK,
            matV,
            vecI,
            vecF,
            matC_initial,
            matN_initial,
            matM_initial,
            return_last_states,
        ):
            B, H, T, K, V = *matQ.shape, matV.shape[-1]
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
                dtype_states = matQ.dtype
            else:
                dtype_states = dtype_state
            if dtype_gate is None:
                dtype_gates = matQ.dtype
            else:
                dtype_gates = dtype_gate

            assert T % BT == 0, "sequence length must be divisible by BT"
            scaF_orig = vecF
            vecF = torch.nn.functional.logsigmoid(vecF.to(dtype_gates))
            vecF = vecF.reshape(B, H, -1, BT)
            vecF = vecF.cumsum(-1) * 1.44269504
            vecF = vecF.reshape(B, H, -1)
            vecI = (vecI.reshape(B, H, -1) * 1.44269504).to(dtype_gates)

            matC_final, matN_final, matM_final = None, None, None
            if return_last_states:
                matC_final = matQ.new_full(
                    (B, H, K, V), float("nan"), requires_grad=False
                )
                matN_final = matQ.new_full((B, H, K), float("nan"), requires_grad=False)
                matM_final = matQ.new_full((B, H), float("nan"), requires_grad=False)

            matC = matQ.new_full((B, H, NT * K, V), float("nan"), dtype=dtype_states)
            matN = matQ.new_full((B, H, NT, K), float("nan"), dtype=dtype_states)
            matM = matQ.new_full((B, H, NT + 1), float("nan"), dtype=dtype_states)
            matM_total = matQ.new_full((B, H, NT, BT), float("nan"), dtype=dtype_states)
            vecNorm = matQ.new_full((B, H, NT, BT), float("nan"), dtype=dtype_states)
            grid = (NK, NV, B * H)
            chunk_mlstm_fwd_kernel_C[grid](
                matK,
                matV,
                matC,
                matN,
                matM,
                vecI,
                vecF,
                matC_initial,
                matN_initial,
                matM_initial,
                matC_final,
                matN_final,
                matM_final,
                matQ.stride(1),
                matQ.stride(2),
                matQ.stride(3),
                matV.stride(1),
                matV.stride(2),
                matV.stride(3),
                matC.stride(1),
                matC.stride(2),
                matN.stride(1),
                H=H,
                T=T,
                K=K,
                V=V,
                BT=BT,
                BK=BK,
                BV=BV,
                NT=NT,
                USE_INITIAL_STATE=matC_initial is not None,
                STORE_FINAL_STATE=return_last_states,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            grid = (NV, NT, B * H)
            matH = torch.empty_like(matV)

            chunk_mlstm_fwd_kernel_h[grid](
                matQ,
                matK,
                matV,
                matC,
                matN,
                matM,
                matM_total,
                vecI,
                vecF,
                matH,
                vecNorm,
                matQ.stride(1),
                matQ.stride(2),
                matQ.stride(3),
                matV.stride(1),
                matV.stride(2),
                matV.stride(3),
                matC.stride(1),
                matC.stride(2),
                matN.stride(1),
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
                    matQ,
                    matK,
                    matV,
                    matC,
                    vecF,
                    vecI,
                    matM,
                    matM_total,
                    vecNorm,
                    matM_final,
                    matC_initial,
                    matN_initial,
                    matM_initial,
                    scaF_orig,
                )
            else:
                ctx.save_for_backward(
                    matQ,
                    matK,
                    matV,
                    None,
                    vecF,
                    vecI,
                    matM,
                    matM_total,
                    vecNorm,
                    matM_final,
                    matC_initial,
                    matN_initial,
                    matM_initial,
                    scaF_orig,
                )
            return matH.to(matQ.dtype), matC_final, matN_final, matM_final

        @staticmethod
        @custom_bwd(device_type="cuda")
        @contiguous
        def backward(ctx, matdH, matdC_final=None, d_finaln=None, d_finalm=None):
            (
                matQ,
                matK,
                matV,
                matC,
                vecF,
                vecI,
                matM,
                matM_total,
                vecNorm,
                matM_final,
                matC_initial,
                matN_initial,
                matM_initial,
                scaF_orig,
            ) = ctx.saved_tensors

            B, H, T, K, V = *matQ.shape, matV.shape[-1]
            BT = chunk_size
            BK, BV = (
                min(
                    32 if matQ.dtype == torch.float32 else 64, triton.next_power_of_2(K)
                ),
                min(
                    32 if matQ.dtype == torch.float32 else 64, triton.next_power_of_2(V)
                ),
            )
            NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)

            if dtype_state is None:
                dtype_states = matQ.dtype
            else:
                dtype_states = dtype_state

            if matC is None:
                num_stages = 1
                num_warps = 4 if BK == 64 else 2
                scale = K**-0.5

                matC = matQ.new_full(
                    (B, H, NT * K, V), float("nan"), dtype=dtype_states
                )
                matN = matQ.new_full((B, H, NT, K), float("nan"), dtype=dtype_states)
                grid = (NK, NV, B * H)
                matC_final, matN_final, matM_final = None, None, None

                chunk_mlstm_fwd_kernel_C[grid](
                    matK,
                    matV,
                    matC,
                    matN,
                    matM,
                    vecI,
                    vecF,
                    matC_initial,
                    matN_initial,
                    matM_initial,
                    matC_final,
                    matN_final,
                    matM_final,
                    matQ.stride(1),
                    matQ.stride(2),
                    matQ.stride(3),
                    matV.stride(1),
                    matV.stride(2),
                    matV.stride(3),
                    matC.stride(1),
                    matC.stride(2),
                    matN.stride(1),
                    H=H,
                    T=T,
                    K=K,
                    V=V,
                    BT=BT,
                    BK=BK,
                    BV=BV,
                    NT=NT,
                    USE_INITIAL_STATE=matC_initial is not None,
                    STORE_FINAL_STATE=False,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

            num_stages = 1
            num_warps = 4 if BK == 64 else 2
            scale = K**-0.5
            matdC = matQ.new_full((B, H, NT * K, V), float("nan"), dtype=dtype_states)

            matdC_initial = matQ.new_full(
                (B, H, K, V), float("nan"), requires_grad=False, dtype=dtype_states
            )
            matM_initial = matQ.new_full(
                (B, H), float("nan"), requires_grad=False, dtype=dtype_states
            )

            if matdC_final is None:
                matdC_final = matQ.new_full(
                    matdC_initial.shape, 0.0, dtype=dtype_states
                )
                matM_final = matQ.new_full(matM_initial.shape, 0.0, dtype=dtype_states)
            else:
                matdC_final = matdC_final.to(dtype_states)
                matM_final = matM_final.to(dtype_states)

            grid = (NK, NV, B * H)
            chunk_mlstm_bwd_kernel_dC[grid](
                matQ,
                vecF,
                matM,
                matM_total,
                vecNorm,
                matdH,
                matdC,
                matdC_final,
                matM_final,
                matdC_initial,
                matM_initial,
                matQ.stride(1),
                matQ.stride(2),
                matQ.stride(3),
                matV.stride(1),
                matV.stride(2),
                matV.stride(3),
                matdC.stride(1),
                matdC.stride(2),
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
            matdQ = torch.empty_like(matQ)
            matdK = torch.empty_like(matK)
            matdV = matV.new_full((NK, *matV.shape), float("nan"))
            num_stages = 1
            num_warps = 4 if BK == 64 else 2
            chunk_mlstm_bwd_kernel_dqkvif[grid](
                matQ,
                matK,
                matV,
                matC,
                matM,
                matM_total,
                matM_final,
                vecNorm,
                vecI,
                vecF,
                matdH,
                matdC,
                matdQ,
                matdK,
                matdV,
                matQ.stride(1),
                matQ.stride(2),
                matQ.stride(3),
                matV.stride(1),
                matV.stride(2),
                matV.stride(3),
                matdC.stride(1),
                matdC.stride(2),
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

            matdV = matdV.sum(0)
            vecdF = (matdQ * matQ - matdK * matK).sum(-1)
            vecdI = (matdV * matV).sum(-1)

            vecdF = rev_cumsum(vecdF)
            vecdF = vecdF * torch.nn.functional.sigmoid(-scaF_orig)

            return (
                matdQ.to(matQ.dtype),
                matdK.to(matK.dtype),
                matdV.to(matV.dtype),
                vecdI.to(scaF_orig.dtype),
                vecdF.to(scaF_orig.dtype).view(vecF.shape),
                matdC_initial.to(matC_initial.dtype)
                if matC_initial is not None
                else None,
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
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = False,
    chunk_size: int = 64,
    keep_states: bool = False,
    eps: float = 1e-6,  # is ignored
    dtype_states: Literal["float32", "bfloat16", "float16"] = "float32",
    autocast_kernel_dtype: torch.dtype | None = torch.float32,
    dtype_gates: Literal["float32", "bfloat16", "float16"] = "float32",
    **kwargs,  # are ignored
) -> tuple[torch.Tensor, torch.Tensor]:
    # actually dtype_gates is not really supported yet
    vecF = f.float()
    vecI = i.float()
    if autocast_kernel_dtype is not None:
        dtype_states = str(autocast_kernel_dtype).split(".")[1]
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
    matH, matC_final, matN_final, matM_final = mLSTMFunc.apply(
        q, k, v, vecI, vecF, c_initial, n_initial, m_initial, return_last_states
    )
    if return_last_states:
        return matH, (matC_final, matN_final, matM_final)
    else:
        return matH


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
        matQ,
        matK,
        matV,
        vecI,
        vecF,
        matC_initial=None,
        matN_initial=None,
        matM_initial=None,
        return_last_states: bool = False,
    ):
        matH, matC_final, matN_final, matM_final = self._func.apply(
            matQ,
            matK,
            matV,
            vecI,
            vecF,
            matC_initial,
            matN_initial,
            matM_initial,
            return_last_states,
        )
        if return_last_states:
            return matH, (matC_final, matN_final, matM_final)
        else:
            return h
