# Author Korbinian Poeppel
from typing import Literal

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
    vecI,  # log igates
    vecF,  # accumulated log fgate
    matC_initial,  # initial state of the chunk [B, H, D_head_K, D_head_V]
    matN_initial,
    matM_initial,
    matC,
    matN,
    matM,
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
    str_C_K,
    str_N_H,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BHQK: tl.constexpr,
    BHHV: tl.constexpr,
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
            (idx_K * BHQK, idx_V * BHHV),
            (BHQK, BHHV),
            (1, 0),
        )
        matN0_ptr = tl.make_block_ptr(
            matN_initial + idx_BC * K,
            (K,),
            (1,),
            (idx_K * BHQK,),
            (BHQK,),
            (0,),
        )
        matM0_ptr = matM_initial + idx_BC

        matC_val = tl.load(matC0_ptr, boundary_check=(0, 1))
        matN_val = tl.load(matN0_ptr, boundary_check=(0,))
        matM_val = tl.load(matM0_ptr)
    else:
        matC_val = tl.zeros([BHQK, BHHV], dtype=tl.load(matC).dtype)
        matN_val = tl.zeros([BHQK], dtype=matC_val.dtype)
        matM_val = 0.0

    matM_next_val = 0.0
    for idx_t in range(NT):
        matK_ptr = tl.make_block_ptr(
            matK + idx_BC * str_QK_H,
            (K, T),
            (str_QK_d, str_QK_t),
            (idx_K * BHQK, idx_t * BT),
            (BHQK, BT),
            (0, 1),
        )
        matV_ptr = tl.make_block_ptr(
            matV + idx_BC * str_VH_H,
            (T, V),
            (str_VH_t, str_VH_d),
            (idx_t * BT, idx_V * BHHV),
            (BT, BHHV),
            (1, 0),
        )
        matC_ptr = tl.make_block_ptr(
            matC + idx_BC * str_C_H + idx_t * K * V,
            (K, V),
            (str_C_K, 1),
            (idx_K * BHQK, idx_V * BHHV),
            (BHQK, BHHV),
            (1, 0),
        )
        matN_ptr = tl.make_block_ptr(
            matN + idx_BC * str_N_H + idx_t * K,
            (K,),
            (1,),
            (idx_K * BHQK,),
            (BHQK,),
            (0,),
        )
        tl.store(
            matC_ptr, matC_val.to(matC_ptr.dtype.element_ty), boundary_check=(0, 1)
        )
        tl.store(matN_ptr, matN_val.to(matN_ptr.dtype.element_ty), boundary_check=(0,))
        tl.store(matM + idx_BC * (NT + 1) + idx_t, matM_val)
        # [BHQK, BT]
        matK_val = tl.load(matK_ptr, boundary_check=(0, 1))
        # [BT, BHHV]
        matV_val = tl.load(matV_ptr, boundary_check=(0, 1))
        # [BHQK, BHHV]

        idx_t_cta = tl.arange(0, BT)
        vecF_val = tl.load(
            vecF + idx_BC * T + idx_t * BT + idx_t_cta + 1,
            mask=idx_t_cta < BT - 1,
            other=0.0,
        )
        vecF_first_val = tl.load(vecF + idx_BC * T + idx_t * BT)
        vecI_val = tl.load(vecI + idx_BC * T + idx_t * BT + idx_t_cta)
        vecG_val = vecI_val + tl.flip(tl.cumsum(tl.flip(vecF_val), axis=0))

        scaF_all_val = tl.sum(vecF_val) + vecF_first_val

        matM_next_val, _ = tl.max(vecG_val)
        matM_next_val = tl.maximum(scaF_all_val + matM_val, matM_next_val)
        matC_val *= tl.math.exp2(scaF_all_val - matM_next_val + matM_val).to(
            matC_val.dtype
        )
        matN_val *= tl.math.exp2(scaF_all_val - matM_next_val + matM_val).to(
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
            (idx_K * BHQK, idx_V * BHHV),
            (BHQK, BHHV),
            (1, 0),
        )
        matN_ptr = tl.make_block_ptr(
            matN_final + idx_BC * K, (K,), (1,), (idx_K * BHQK,), (BHQK,), (0,)
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
    vecI,
    vecF,
    matH,
    vecNorm,
    matM_total,
    str_QK_H,
    str_QK_t,
    str_QK_d,
    str_VH_H,
    str_VH_t,
    str_VH_d,
    str_C_H,
    str_C_K,
    str_N_H,
    scale: tl.constexpr,
    EPS: tl.constexpr,
    STABILIZE_CORRECTLY: tl.constexpr,
    NORM_VAL: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BHQK: tl.constexpr,
    BHHV: tl.constexpr,
    NT: tl.constexpr,
):
    idx_V, idx_t, idx_BC = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    idx_t_cta = tl.arange(0, BT)[:, None]
    idx_s_cta = tl.arange(0, BS)[None, :]
    matM_s = idx_t_cta >= idx_s_cta

    matH_val = tl.zeros([BT, BHHV], dtype=tl.load(matQ).dtype)
    matS_val = tl.zeros([BT, BT], dtype=matH_val.dtype)
    vecNorm_val = tl.zeros([BT, BHHV], dtype=matH_val.dtype)
    for idx_K in range(tl.cdiv(K, BHQK)):
        matQ_ptr = tl.make_block_ptr(
            matQ + idx_BC * str_QK_H,
            (T, K),
            (str_QK_t, str_QK_d),
            (idx_t * BT, idx_K * BHQK),
            (BT, BHQK),
            (1, 0),
        )
        matK_ptr = tl.make_block_ptr(
            matK + idx_BC * str_QK_H,
            (K, T),
            (str_QK_d, str_QK_t),
            (idx_K * BHQK, idx_t * BT),
            (BHQK, BT),
            (0, 1),
        )
        matC_ptr = tl.make_block_ptr(
            matC + idx_BC * str_C_H + idx_t * K * V,
            (K, V),
            (str_C_K, 1),
            (idx_K * BHQK, idx_V * BHHV),
            (BHQK, BHHV),
            (1, 0),
        )
        matN_ptr = tl.make_block_ptr(
            matN + idx_BC * str_N_H + idx_t * K,
            (K, BHHV),
            (1, 0),
            (idx_K * BHQK, 0),
            (BHQK, BHHV),
            (0, 1),
        )

        # [BT, BHQK]
        matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1))
        # [BHQK, BT]
        matK_val = tl.load(matK_ptr, boundary_check=(0, 1))
        # [BT]

        # [BHQK, BHHV]
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

    vecF_ptr = vecF + idx_BC * T + idx_t * BT + idx_t_cta
    vecF_val = tl.load(vecF_ptr)
    vecFc_val = tl.cumsum(vecF_val, axis=0)

    vecI_ptr = vecI + idx_BC * T + idx_t * BT + idx_s_cta
    vecI_val = tl.load(vecI_ptr)
    matM_val = tl.load(matM + idx_BC * (NT + 1) + idx_t)

    vecF_bottom_cum_val = tl.cumsum(
        tl.where(idx_t_cta > idx_s_cta, vecF_val.broadcast_to(BT, BS), 0.0)
    )
    matlogD_val = vecI_val + vecF_bottom_cum_val
    matlogD_val = tl.where(matM_s, vecI_val + vecF_bottom_cum_val, -float("inf"))

    matmlogD_val = tl.max(matlogD_val, axis=1)

    matM_total_val = tl.maximum(tl.max(vecF_val, axis=1) + matM_val, matmlogD_val)
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
        matH_val * tl.math.exp2(vecFc_val + matM_val - matM_total_val[:, None]) * scale
    )
    matS_val = matS_val * matD_val * scale
    vecNorm_val = (
        vecNorm_val
        * tl.math.exp2(vecFc_val + matM_val - matM_total_val[:, None])
        * scale
    )

    matS_val = tl.where(matM_s, matS_val, 0)
    vecNorm_val += tl.sum(matS_val, axis=1)[:, None]
    vecNorm_val = tl.abs(vecNorm_val)

    if STABILIZE_CORRECTLY:
        vecNorm_val = (
            tl.maximum(
                vecNorm_val.to(matM_total_val.dtype),
                NORM_VAL * tl.math.exp2(-matM_total_val)[:, None],
            )
            + EPS
        )
    else:
        vecNorm_val = (
            tl.maximum(
                vecNorm_val.to(matM_total_val.dtype),
                NORM_VAL + tl.zeros(vecNorm_val.shape, matM_total_val.dtype),
            )
            + EPS
        )

    tl.store(
        vecNorm + idx_BC * T + idx_t * BT + tl.arange(0, BT),
        tl.max(vecNorm_val, axis=1),
    )

    matV_ptr = tl.make_block_ptr(
        matV + idx_BC * str_VH_H,
        (T, V),
        (str_VH_t, str_VH_d),
        (idx_t * BT, idx_V * BHHV),
        (BT, BHHV),
        (1, 0),
    )
    matV_val = tl.load(matV_ptr, boundary_check=(0, 1))
    matH_val = (
        matH_val + tl.dot(matS_val.to(matV_val.dtype), matV_val, allow_tf32=False)
    ) / vecNorm_val
    matH_ptr = tl.make_block_ptr(
        matH + idx_BC * str_VH_H,
        (T, V),
        (str_VH_t, str_VH_d),
        (idx_t * BT, idx_V * BHHV),
        (BT, BHHV),
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
    matM_final,
    matdH,
    matdC_final,
    matdC,
    matdC_initial,
    matM_initial,
    str_QK_H,
    str_QK_t,
    str_QK_d,
    str_VH_H,
    str_VH_t,
    str_VH_d,
    str_C_H,
    str_C_K,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BHQK: tl.constexpr,
    BHHV: tl.constexpr,
    NT: tl.constexpr,
    USE_LAST_STATE: tl.constexpr,
    STORE_INITIAL_STATE: tl.constexpr,
):
    idx_K, idx_V, idx_BC = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # [BHQK, BHHV]
    if USE_LAST_STATE:
        # [BHQK, BHHV]
        matdC_ptr = tl.make_block_ptr(
            matdC_final + idx_BC * K * V,
            (K, V),
            (str_C_K, 1),
            (idx_K * BHQK, idx_V * BHHV),
            (BHQK, BHHV),
            (1, 0),
        )
        matdC_val = tl.load(matdC_ptr, boundary_check=(0, 1))
        matM_val = tl.load(matM_final + idx_BC)
    else:
        matdC_val = tl.zeros((BHQK, BHHV), dtype=tl.load(matdC_final).dtype)
        matM_val = tl.load(matM + idx_BC * (NT + 1) + NT)
    for idx_t in range(NT - 1, -1, -1):
        matQ_ptr = tl.make_block_ptr(
            matQ + idx_BC * str_QK_H,
            (K, T),
            (str_QK_d, str_QK_t),
            (idx_K * BHQK, idx_t * BT),
            (BHQK, BT),
            (0, 1),
        )
        matdH_ptr = tl.make_block_ptr(
            matdH + idx_BC * str_VH_H,
            (T, V),
            (str_VH_t, str_VH_d),
            (idx_t * BT, idx_V * BHHV),
            (BT, BHHV),
            (1, 0),
        )
        matdC_ptr = tl.make_block_ptr(
            matdC + idx_BC * str_C_H + idx_t * K * V,
            (K, V),
            (str_C_K, 1),
            (idx_K * BHQK, idx_V * BHHV),
            (BHQK, BHHV),
            (1, 0),
        )

        tl.store(
            matdC_ptr, matdC_val.to(matdC_ptr.dtype.element_ty), boundary_check=(0, 1)
        )
        vecF_val = tl.load(vecF + idx_BC * T + idx_t * BT + tl.arange(0, BT))
        scaF_all_val = tl.sum(vecF_val)

        matM_p_val = tl.load(matM + idx_BC * (NT + 1) + idx_t)
        matM_total_val = tl.load(
            matM_total + idx_BC * T + idx_t * BT + tl.arange(0, BT)
        )
        vecNorm_val = tl.load(vecNorm + idx_BC * T + idx_t * BT + tl.arange(0, BT))

        # [BHQK, BT]
        matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1))
        matQ_val = (
            matQ_val
            * scale
            * tl.math.exp2(tl.cumsum(vecF_val) + matM_p_val - matM_total_val)[None, :]
        ).to(matQ_val.dtype)
        # [BT, V]
        matdH_val = tl.load(matdH_ptr, boundary_check=(0, 1))
        matdH_val /= vecNorm_val[:, None]
        # [BHQK, BHHV]
        matdC_val *= tl.math.exp2(scaF_all_val + matM_p_val - matM_val).to(
            matdC_val.dtype
        )
        matdC_val += tl.dot(
            matQ_val, matdH_val.to(matQ_val.dtype), allow_tf32=False
        ).to(matdC_val.dtype)
        matM_val = matM_p_val

    if STORE_INITIAL_STATE:
        matdC_initial_ptr = tl.make_block_ptr(
            matdC_initial + idx_BC * K * V,
            (K, V),
            (V, 1),
            (idx_K * BHQK, idx_V * BHHV),
            (BHQK, BHHV),
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
    vecNorm,
    vecI,
    vecF,
    matdH,
    matdC,
    matdQ,
    matdK,
    matdV,
    vecdI,
    vecdFq,
    vecdFk,
    scadFc,
    str_QK_H,
    str_QK_t,
    str_QK_d,
    str_VH_H,
    str_VH_t,
    str_VH_d,
    str_C_H,
    str_C_K,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BHQK: tl.constexpr,
    BHHV: tl.constexpr,
    NT: tl.constexpr,
):
    idx_K, idx_t, idx_BC = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    matN_bh = tl.num_programs(2)
    o_cta = tl.arange(0, BT)

    matQ_ptr = tl.make_block_ptr(
        matQ + idx_BC * str_QK_H,
        (K, T),
        (str_QK_d, str_QK_t),
        (idx_K * BHQK, idx_t * BT),
        (BHQK, BT),
        (0, 1),
    )
    matK_ptr = tl.make_block_ptr(
        matK + idx_BC * str_QK_H,
        (T, K),
        (str_QK_t, str_QK_d),
        (idx_t * BT, idx_K * BHQK),
        (BT, BHQK),
        (1, 0),
    )

    matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1))
    matK_val = tl.load(matK_ptr, boundary_check=(0, 1))
    matS_val = tl.dot(matK_val, matQ_val, allow_tf32=False)
    vecF_ptr = vecF + idx_BC * T + idx_t * BT + o_cta
    vecI_ptr = vecI + idx_BC * T + idx_t * BT + o_cta
    vecF_val = tl.load(vecF_ptr)
    vecFc_val = tl.cumsum(vecF_val, axis=0)
    scaF_all_val = tl.sum(vecF_val, axis=0)
    matM_val = tl.load(matM + idx_BC * (NT + 1) + idx_t)
    matM_total_val = tl.load(matM_total + idx_BC * T + idx_t * BT + o_cta)
    vecI_val = tl.load(vecI_ptr)

    matM_s = o_cta[None, :] >= o_cta[:, None]
    vecF_right_cum_val = tl.cumsum(
        tl.where(
            o_cta[None, :] > o_cta[:, None], vecF_val[None, :].broadcast_to(BS, BT), 0.0
        ),
        axis=1,
    )
    logDT_val = vecI_val[:, None] + vecF_right_cum_val - matM_total_val[None, :]
    DT_val = tl.where(matM_s, tl.math.exp2(logDT_val) * scale, 0.0)
    matS_val = matS_val * DT_val

    vecF_rev_val = tl.load(
        vecF + idx_BC * T + idx_t * BT + o_cta + 1, mask=o_cta < BT - 1, other=0.0
    )
    vecF_rc_val = tl.flip(tl.cumsum(tl.flip(vecF_rev_val), axis=0))
    vecF_ac_val = tl.load(vecF + idx_BC * T + idx_t * BT + o_cta)
    vecFc_val = tl.cumsum(vecF_ac_val, axis=0)
    idx_vecG_val = tl.load(vecI + idx_BC * T + idx_t * BT + o_cta)
    vecG_val = idx_vecG_val + vecF_rc_val
    vecNorm_val = tl.load(vecNorm + idx_BC * T + idx_t * BT + o_cta)

    # if idx_t == NT - 1:
    #     matM_next_val = tl.load(matM_final)
    # else:

    matM_next_val = tl.load(matM + idx_BC * (NT + 1) + idx_t + 1)

    matdQ_val = tl.zeros([BT, BHQK], dtype=matQ_val.dtype)
    matdK_val = tl.zeros([BT, BHQK], dtype=matQ_val.dtype)
    matdS_val = tl.zeros([BT, BT], dtype=matQ_val.dtype)
    vecdI_val = tl.zeros([BT], dtype=vecG_val.dtype)

    vecdI_ptr = tl.make_block_ptr(
        vecdI + (idx_K * matN_bh + idx_BC) * T, (T,), (1,), (idx_t * BT,), (BT,), (0,)
    )
    scadFc_ptr = scadFc + (idx_K * matN_bh + idx_BC) * NT + idx_t
    scadFc_val = 0.0

    for idx_V in range(tl.cdiv(V, BHHV)):
        matV_ptr = tl.make_block_ptr(
            matV + idx_BC * str_VH_H,
            (T, V),
            (str_VH_t, str_VH_d),
            (idx_t * BT, idx_V * BHHV),
            (BT, BHHV),
            (1, 0),
        )
        matC_ptr = tl.make_block_ptr(
            matC + idx_BC * str_C_H,
            (V, NT * K),
            (1, str_C_K),
            (idx_V * BHHV, idx_t * K + idx_K * BHQK),
            (BHHV, BHQK),
            (0, 1),
        )
        matdH_ptr = tl.make_block_ptr(
            matdH + idx_BC * str_VH_H,
            (T, V),
            (str_VH_t, str_VH_d),
            (idx_t * BT, idx_V * BHHV),
            (BT, BHHV),
            (1, 0),
        )
        matdC_ptr = tl.make_block_ptr(
            matdC + idx_BC * str_C_H,
            (NT * K, V),
            (str_C_K, 1),
            (idx_t * K + idx_K * BHQK, idx_V * BHHV),
            (BHQK, BHHV),
            (1, 0),
        )
        matdV_ptr = tl.make_block_ptr(
            matdV + (idx_K * matN_bh + idx_BC) * str_VH_H,
            (T, V),
            (str_VH_t, str_VH_d),
            (idx_t * BT, idx_V * BHHV),
            (BT, BHHV),
            (1, 0),
        )
        # [BT, BHHV]
        matV_val = tl.load(matV_ptr, boundary_check=(0, 1))
        matdH_val = tl.load(matdH_ptr, boundary_check=(0, 1))
        # [BHHV, BHQK]
        matC_val = tl.load(matC_ptr, boundary_check=(0, 1))
        # [BHQK, BHHV]
        matdC_val = tl.load(matdC_ptr, boundary_check=(0, 1))
        # [BT, BT]
        matdS_val += tl.dot(matdH_val, tl.trans(matV_val), allow_tf32=False).to(
            matdS_val.dtype
        )
        # [BT, BHQK]
        matdQ_val += (
            tl.dot(matdH_val, matC_val.to(matdH_val.dtype), allow_tf32=False) * scale
        ).to(matdQ_val.dtype)
        matdK_val += tl.dot(
            matV_val, tl.trans(matdC_val.to(matV_val.dtype)), allow_tf32=False
        ).to(matdK_val.dtype)
        # [BT, BHHV]
        matdV_val = (
            tl.dot(matK_val, matdC_val.to(matK_val.dtype), allow_tf32=False).to(
                matQ_val.dtype
            )
            * tl.math.exp2(vecG_val - matM_next_val).to(matQ_val.dtype)[:, None]
        )
        matdV_val += tl.dot(
            (matS_val / vecNorm_val[None, :]).to(matQ_val.dtype),
            matdH_val.to(matQ_val.dtype),
            allow_tf32=False,
        ).to(matQ_val.dtype)
        vecdI_val += tl.sum(matdV_val * matV_val, axis=1).to(tl.float32)

        scadFc_val += tl.sum(
            tl.sum(tl.trans(matC_val) * matdC_val, axis=0).to(tl.float32), axis=0
        )
        tl.store(
            matdV_ptr, matdV_val.to(matdV_ptr.dtype.element_ty), boundary_check=(0, 1)
        )

    tl.store(vecdI_ptr, vecdI_val.to(vecdI_ptr.dtype.element_ty), boundary_check=(0,))
    tl.store(
        scadFc_ptr, scadFc_val * tl.math.exp2(scaF_all_val + matM_val - matM_next_val)
    )

    matdQ_val *= (
        tl.math.exp2(vecFc_val + matM_val - matM_total_val)[:, None]
        / vecNorm_val[:, None]
    ).to(matdQ_val.dtype)
    vecdFq1_val = tl.flip(
        tl.cumsum(
            tl.flip(
                tl.sum((tl.trans(matQ_val) * matdQ_val).to(vecF_val.dtype), axis=1)
            ),
            axis=0,
        )
    )
    matdK_val = matdK_val * tl.math.exp2(vecG_val - matM_next_val)[:, None]
    vecdFk_val = tl.cumsum(tl.sum((matdK_val * matK_val).to(vecF_val.dtype), axis=1))

    matdS_val = tl.trans(DT_val) * matdS_val
    matdS_val = matdS_val.to(matK_val.dtype)

    # [BT, BHQK]
    matdQ_p_val = tl.dot(matdS_val, matK_val, allow_tf32=False) / vecNorm_val[
        :, None
    ].to(matdQ_val.dtype)
    matdQ_val += matdQ_p_val
    matdK_p_val = tl.trans(
        tl.dot(
            (matQ_val / vecNorm_val[None, :]).to(matQ_val.dtype),
            matdS_val,
            allow_tf32=False,
        )
    )
    matdK_val += matdK_p_val
    vecdG_val = tl.trans(matdS_val) * tl.dot(
        matK_val,
        (matQ_val / vecNorm_val[None, :].to(matQ_val.dtype)).to(matQ_val.dtype),
    )

    mask = o_cta[:, None] < o_cta[None, :]
    vecdG_val = tl.where(mask, vecdG_val, 0.0)
    vecdFg_val = tl.sum(
        tl.where(mask, tl.flip(tl.cumsum(tl.flip(vecdG_val), axis=1)), 0.0), axis=0
    )

    vecdFq_val = vecdFq1_val + vecdFg_val

    matdQ_ptr = tl.make_block_ptr(
        matdQ + idx_BC * str_QK_H,
        (T, K),
        (str_QK_t, str_QK_d),
        (idx_t * BT, idx_K * BHQK),
        (BT, BHQK),
        (1, 0),
    )
    matdK_ptr = tl.make_block_ptr(
        matdK + idx_BC * str_QK_H,
        (T, K),
        (str_QK_t, str_QK_d),
        (idx_t * BT, idx_K * BHQK),
        (BT, BHQK),
        (1, 0),
    )
    vecdFq_ptr = tl.make_block_ptr(
        vecdFq + (idx_K * matN_bh + idx_BC) * T, (T,), (1,), (idx_t * BT,), (BT,), (0,)
    )
    vecdFk_ptr = tl.make_block_ptr(
        vecdFk + (idx_K * matN_bh + idx_BC) * (T + 1) + 1,
        (T + 1,),
        (1,),
        (idx_t * BT,),
        (BT,),
        (0,),
    )
    vecdFk_val = tl.where(o_cta < BT - 1, vecdFk_val, 0.0)
    tl.store(matdQ_ptr, matdQ_val.to(matdQ_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(matdK_ptr, matdK_val.to(matdK_ptr.dtype.element_ty), boundary_check=(0, 1))

    tl.store(
        vecdFq_ptr, vecdFq_val.to(vecdFq_ptr.dtype.element_ty), boundary_check=(0,)
    )
    tl.store(
        vecdFk_ptr, vecdFk_val.to(vecdFk_ptr.dtype.element_ty), boundary_check=(0,)
    )


def mLSTMforward(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    matC_initial: torch.Tensor,
    matN_initial: torch.Tensor,
    matM_initial: torch.Tensor,
    dtype_state: torch.dtype | None = torch.float32,
    dtype_gate: torch.dtype | None = torch.float32,
    chunk_size: int = 64,
    return_last_states: bool = False,
    EPS: float = 1e-6,
    STABILIZE_CORRECTLY: bool = True,
    NORM_VAL: float = 1.0,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    B, H, T, K, V = *matQ.shape, matV.shape[-1]
    BT = chunk_size
    BHQK, BHHV = (
        min(64, triton.next_power_of_2(K)),
        min(64, triton.next_power_of_2(V)),
    )
    NT, siz_K, siz_V = triton.cdiv(T, BT), triton.cdiv(K, BHQK), triton.cdiv(V, BHHV)
    num_stages = 1
    num_warps = 4 if BHQK == 64 else 2
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
    vecF = torch.nn.functional.logsigmoid(vecF.to(dtype_gates))
    vecF = vecF.reshape(B, H, -1, BT)
    vecF = vecF * 1.44269504
    vecF = vecF.reshape(B, H, -1)
    vecI = (vecI.reshape(B, H, -1) * 1.44269504).to(dtype_gates)

    matC_final, matN_final, matM_final = None, None, None
    if return_last_states:
        matC_final = matQ.new_full(
            (B, H, K, V), float("nan"), requires_grad=False, dtype=dtype_states
        )
        matN_final = matQ.new_full(
            (B, H, K), float("nan"), requires_grad=False, dtype=dtype_states
        )
        matM_final = matQ.new_full(
            (B, H), float("nan"), requires_grad=False, dtype=dtype_states
        )

    matC = matQ.new_full((B, H, NT * K, V), float("nan"), dtype=dtype_states)
    matN = matQ.new_full((B, H, NT, K), float("nan"), dtype=dtype_states)
    matM = matQ.new_full((B, H, NT + 1), float("nan"), dtype=dtype_states)
    matM_total = matQ.new_full((B, H, NT, BT), float("nan"), dtype=dtype_states)
    vecNorm = matQ.new_full((B, H, NT, BT), float("nan"), dtype=dtype_states)
    grid = (siz_K, siz_V, B * H)
    chunk_mlstm_fwd_kernel_C[grid](
        matK,
        matV,
        vecI,
        vecF,
        matC_initial,
        matN_initial,
        matM_initial,
        matC,
        matN,
        matM,
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
        T=T,
        K=K,
        V=V,
        BT=BT,
        BHQK=BHQK,
        BHHV=BHHV,
        NT=NT,
        USE_INITIAL_STATE=matC_initial is not None,
        STORE_FINAL_STATE=return_last_states,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    grid = (siz_V, NT, B * H)
    matH = torch.empty_like(matV)

    chunk_mlstm_fwd_kernel_h[grid](
        matQ,
        matK,
        matV,
        matC,
        matN,
        matM,
        vecI,
        vecF,
        matH,
        vecNorm,
        matM_total,
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
        EPS=EPS,
        STABILIZE_CORRECTLY=STABILIZE_CORRECTLY,
        NORM_VAL=NORM_VAL,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BS=BT,
        BHQK=BHQK,
        BHHV=BHHV,
        NT=NT,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return matH, matC, vecNorm, matM, matM_total, matC_final, matN_final, matM_final


def mLSTMbackward(
    matdH: torch.Tensor,
    matdC_final: torch.Tensor | None,
    matdN_final: torch.Tensor | None,
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    matC: torch.Tensor | None,
    vecF: torch.Tensor,
    vecI: torch.Tensor,
    matM: torch.Tensor,
    matM_total: torch.Tensor,
    vecNorm: torch.Tensor,
    matM_final: torch.Tensor,
    matC_initial: torch.Tensor | None,
    matN_initial: torch.Tensor | None,
    matM_initial: torch.Tensor | None,
    dtype_state: torch.dtype,
    dtype_gate: torch.dtype,
    chunk_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    None,
    None,
    None,
]:
    B, H, T, K, V = *matQ.shape, matV.shape[-1]
    BT = chunk_size
    BHQK, BHHV = (
        min(32 if matQ.dtype == torch.float32 else 64, triton.next_power_of_2(K)),
        min(32 if matQ.dtype == torch.float32 else 64, triton.next_power_of_2(V)),
    )
    NT, siz_K, siz_V = triton.cdiv(T, BT), triton.cdiv(K, BHQK), triton.cdiv(V, BHHV)

    if dtype_state is None:
        dtype_states = matQ.dtype
    else:
        dtype_states = dtype_state
    if dtype_gate is None:
        dtype_gates = matQ.dtype
    else:
        dtype_gates = dtype_gate
    vecF_orig = vecF
    vecF = torch.nn.functional.logsigmoid(vecF.to(dtype_gates))
    vecF = vecF.reshape(B, H, -1, BT)
    vecF = vecF * 1.44269504
    vecF = vecF.reshape(B, H, -1)
    vecI = (vecI.reshape(B, H, -1) * 1.44269504).to(dtype_gates)

    if matC is None:
        num_stages = 1
        num_warps = 4 if BHQK == 64 else 2
        scale = K**-0.5

        matC = matQ.new_full((B, H, NT * K, V), float("nan"), dtype=dtype_states)
        matN = matQ.new_full((B, H, NT, K), float("nan"), dtype=dtype_states)
        grid = (siz_K, siz_V, B * H)
        matC_final, matN_final, matM_final = None, None, None

        chunk_mlstm_fwd_kernel_C[grid](
            matK,
            matV,
            vecI,
            vecF,
            matC_initial,
            matN_initial,
            matM_initial,
            matC,
            matN,
            matM,
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
            T=T,
            K=K,
            V=V,
            BT=BT,
            BHQK=BHQK,
            BHHV=BHHV,
            NT=NT,
            USE_INITIAL_STATE=matC_initial is not None,
            STORE_FINAL_STATE=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    num_stages = 1
    num_warps = 4 if BHQK == 64 else 2
    scale = K**-0.5
    matdC = matQ.new_full((B, H, NT * K, V), float("nan"), dtype=dtype_states)

    USE_INITIAL_STATE = matC_initial is not None
    if USE_INITIAL_STATE:
        matdC_initial = matQ.new_full(
            (B, H, K, V), float("nan"), requires_grad=False, dtype=dtype_states
        )
        matM_initial = matQ.new_full(
            (B, H), float("nan"), requires_grad=False, dtype=dtype_states
        )
    else:
        matdC_initial = matQ.new_zeros((1,), requires_grad=False, dtype=dtype_states)
        matM_initial = matQ.new_zeros((1,), requires_grad=False, dtype=dtype_states)

    if matdC_final is None:
        matdC_final = matQ.new_empty((1,), dtype=dtype_states)
        matM_final = matQ.new_empty((1,), dtype=dtype_states)
        USE_LAST_STATE = False
    else:
        USE_LAST_STATE = True
        matdC_final = matdC_final.to(dtype_states)
        matM_final = matM_final.to(dtype_states)

    grid = (siz_K, siz_V, B * H)
    chunk_mlstm_bwd_kernel_dC[grid](
        matQ,
        vecF,
        matM,
        matM_total,
        vecNorm,
        matM_final,
        matdH,
        matdC_final,
        matdC,
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
        T=T,
        K=K,
        V=V,
        BT=BT,
        BHQK=BHQK,
        BHHV=BHHV,
        NT=NT,
        USE_LAST_STATE=USE_LAST_STATE,
        STORE_INITIAL_STATE=USE_INITIAL_STATE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if not USE_INITIAL_STATE:
        matdC_initial = None
    grid = (siz_K, NT, B * H)
    matdQ = torch.empty_like(matQ)
    matdK = torch.empty_like(matK)
    matdV = matV.new_full((siz_K, *matV.shape), float("nan"))
    vecdI = vecI.new_full((siz_K, *vecI.shape), float("nan"))
    vecdFq = vecF.new_zeros(siz_K, B * H, T)
    vecdFk = vecF.new_zeros(siz_K, B * H, T + 1)
    scadFc = vecF.new_zeros(siz_K, B * H, NT)
    num_stages = 1
    num_warps = 4 if BHQK == 64 else 2
    chunk_mlstm_bwd_kernel_dqkvif[grid](
        matQ,
        matK,
        matV,
        matC,
        matM,
        matM_total,
        vecNorm,
        vecI,
        vecF,
        matdH,
        matdC,
        matdQ,
        matdK,
        matdV,
        vecdI,
        vecdFq,
        vecdFk,
        scadFc,
        matQ.stride(1),
        matQ.stride(2),
        matQ.stride(3),
        matV.stride(1),
        matV.stride(2),
        matV.stride(3),
        matdC.stride(1),
        matdC.stride(2),
        scale,
        T=T,
        K=K,
        V=V,
        BT=BT,
        BS=BT,
        BHQK=BHQK,
        BHHV=BHHV,
        NT=NT,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    matdV = matdV.sum(0)
    # baselines
    # vecdF = (matdQ * matQ - matdK * matK).sum(-1)
    #  vecdI = (matdV * matV).sum(-1)

    # stabilized version without differences
    vecdI = vecdI.sum(0)
    vecdF = vecdFq.sum(0)
    vecdF += vecdFk[:, :, :-1].sum(0)
    scadFc = scadFc.sum(0)

    vecdF = vecdF.view(B * H, NT, BT) + scadFc.view(B * H, NT, 1)
    vecdF = vecdF.view(vecF_orig.shape) * torch.nn.functional.sigmoid(-vecF_orig)
    return (
        matdQ.to(matQ.dtype),
        matdK.to(matK.dtype),
        matdV.to(matV.dtype),
        vecdI.to(vecF_orig.dtype),
        vecdF.to(vecF_orig.dtype).view(vecF.shape),
        matdC_initial.to(matC_initial.dtype) if matC_initial is not None else None,
        None,
        None,
        None,
    )


def mLSTMFunctionGenerator(
    chunk_size: int = 64,
    keep_states: bool = False,
    dtype_state: torch.dtype | None = torch.float32,
    dtype_gate: torch.dtype | None = torch.float32,
    EPS: float = 1e-6,
    NORM_VAL: float = 1.0,
    STABILIZE_CORRECTLY: bool = True,
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
            (
                matH,
                matC,
                vecNorm,
                matM,
                matM_total,
                matC_final,
                matN_final,
                matM_final,
            ) = mLSTMforward(
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                matC_initial=matC_initial,
                matN_initial=matN_initial,
                matM_initial=matM_initial,
                dtype_state=dtype_state,
                dtype_gate=dtype_gate,
                chunk_size=chunk_size,
                return_last_states=return_last_states,
                EPS=EPS,
                STABILIZE_CORRECTLY=STABILIZE_CORRECTLY,
                NORM_VAL=NORM_VAL,
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
                )
            return matH.to(matQ.dtype), matC_final, matN_final, matM_final

        @staticmethod
        @custom_bwd(device_type="cuda")
        @contiguous
        def backward(ctx, matdH, matdC_final=None, matdN_final=None, matdM_final=None):
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
            ) = ctx.saved_tensors
            _ = matdM_final
            return mLSTMbackward(
                matdH=matdH,
                matdC_final=matdC_final,
                matdN_final=matdN_final,
                matQ=matQ,
                matK=matK,
                matV=matV,
                matC=matC,
                vecF=vecF,
                vecI=vecI,
                matM=matM,
                matM_total=matM_total,
                vecNorm=vecNorm,
                matM_final=matM_final,
                matC_initial=matC_initial,
                matN_initial=matN_initial,
                matM_initial=matM_initial,
                dtype_state=dtype_state,
                dtype_gate=dtype_gate,
                chunk_size=chunk_size,
            )

    return mLSTMFunction


mLSTMFunctionDict = {}

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
    eps: float = 1e-6,
    norm_val: float = 1.0,
    stabilize_correctly: bool = True,
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
    signature = (
        chunk_size,
        keep_states,
        dtype_states,
        dtype_gates,
        eps,
        norm_val,
        stabilize_correctly,
    )
    if signature not in mLSTMFunctionDict:
        mLSTMFunctionDict[signature] = mLSTMFunctionGenerator(
            chunk_size=chunk_size,
            keep_states=keep_states,
            dtype_state=DTYPESTR_TO_DTYPE[dtype_states],
            dtype_gate=DTYPESTR_TO_DTYPE[dtype_gates],
            NORM_VAL=norm_val,
            EPS=eps,
            STABILIZE_CORRECTLY=stabilize_correctly,
        )
    mLSTMFunc = mLSTMFunctionDict[signature]
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
            return matH
