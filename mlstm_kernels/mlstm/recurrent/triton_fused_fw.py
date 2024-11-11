# Copyright JKU Linz 2024
# Author: Maximilian Beck
"""
Triton.

This module contains the recurrent step of the mLSTM in triton.

We want to compare this to the torch implementation in mlstm_kernels/mlstm/recurrent/torch_fw.py.

This is a fused forward decoding step kernel for the mLSTM. Factor of 2 speedup compared to torch.compile.
Ca. 30% faster than non-fused version.

TODO this kernel still does not use tensor cores.
Not sure how to use tensor cores with triton in this case. One needs to pad a block with zeros in SRAM.
I don't know how to do this with triton, yet.
"""

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ...kernel_utils import contiguous_noctx, is_power_of_2, torch2triton_dtype
from .sequence_loop import recurrent_sequence_fw


def mlstm_recurrent_sequence_torch_step_triton_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    ret_tuple = recurrent_sequence_fw(
        mlstm_step_fn=recurrent_step_fw,
        matQ=q,
        matK=k,
        matV=v,
        vecI=i,
        vecF=f,
        matC_initial=c_initial,
        vecN_initial=n_initial,
        scaM_initial=m_initial,
        return_last_states=return_last_states,
        EPS=eps,
        return_all_states=False,
    )
    if return_last_states:
        return ret_tuple[0], ret_tuple[3]
    else:
        return ret_tuple[0]


@triton.jit
def _recurrent_step_fw_kernel(
    matC_old,  # (B, NH, DHQK, DHHV)
    vecN_old,  # (B, NH, DHQK)
    scaM_old,  # (B, NH, 1)
    vecQ,  # (B, NH, DHQK)
    vecK,  # (B, NH, DHQK)
    vecV,  # (B, NH, DHHV)
    scaI,  # (B, NH, 1)
    scaF,  # (B, NH, 1)
    vecH,  # (B, NH, DHHV)
    matC_new,  # (B, NH, DHQK, DHHV)
    vecN_new,  # (B, NH, DHQK)
    scaM_new,  # (B, NH, 1)
    qk_scale: tl.constexpr,
    str_matC_B_NH: tl.constexpr,
    str_matC_DHQK: tl.constexpr,
    str_matC_DHHV: tl.constexpr,
    str_vecN_B_NH: tl.constexpr,
    str_vecN_DHQK: tl.constexpr,
    str_scaM_B_NH: tl.constexpr,
    str_vecQK_NH: tl.constexpr,
    str_vecQK_DHQK: tl.constexpr,
    str_vecVH_B_NH: tl.constexpr,
    str_vecVH_DHHV: tl.constexpr,
    str_scaIF_B_NH: tl.constexpr,
    B: tl.constexpr,
    NH: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    EPS: tl.constexpr = 1e-6,
    DTYPE: tl.constexpr = tl.float16,
):
    i_dhv, i_bnh = tl.program_id(1), tl.program_id(2)

    # ? Define pointers
    matC_old_bptr = tl.make_block_ptr(
        base=matC_old + i_bnh * str_matC_B_NH,
        shape=(DHQK, DHHV),
        strides=(str_matC_DHQK, str_matC_DHHV),
        offsets=(0, i_dhv * siz_b_DHHV),
        block_shape=(siz_b_DHQK, siz_b_DHHV),
        order=(0, 1),
    )
    matC_new_bptr = tl.make_block_ptr(
        base=matC_new + i_bnh * str_matC_B_NH,
        shape=(DHQK, DHHV),
        strides=(str_matC_DHQK, str_matC_DHHV),
        offsets=(0, i_dhv * siz_b_DHHV),
        block_shape=(siz_b_DHQK, siz_b_DHHV),
        order=(0, 1),
    )
    vecH_ptr = vecH + i_bnh * str_vecVH_B_NH + i_dhv * siz_b_DHHV * str_vecVH_DHHV + tl.arange(0, siz_b_DHHV)

    scaI_ptr = scaI + i_bnh * str_scaIF_B_NH
    scaF_ptr = scaF + i_bnh * str_scaIF_B_NH

    scaM_old_ptr = scaM_old + i_bnh * str_scaM_B_NH
    scaM_new_ptr = scaM_new + i_bnh * str_scaM_B_NH

    # ? Load data
    # gates
    # the numbers are the conversion factors from log -> log2 and exp -> exp2
    # math.log2(math.e) = 1.4426950408889634
    # (1/math.log2(math.e)) = 0.6931471805599453
    # tl.exp and tl.sigmoid only work with float32
    scaF_val = tl.load(scaF_ptr).to(tl.float32)
    scaI_val = tl.load(scaI_ptr).to(tl.float32)
    scaFlog_val = tl.log2(tl.sigmoid(scaF_val)) * 0.6931471805599453

    scaM_old_val = tl.load(scaM_old_ptr)
    scaM_new_val = tl.maximum(scaFlog_val + scaM_old_val, scaI_val)
    tl.store(scaM_new_ptr, scaM_new_val.to(DTYPE))

    max_val = tl.exp2((-scaM_new_val.to(tl.float32)) * 1.4426950408889634).to(DTYPE)

    # gate computation for all dimensions
    scaF_act = tl.exp2((scaFlog_val + scaM_old_val - scaM_new_val) * 1.4426950408889634).to(DTYPE)
    scaI_act = tl.exp2((scaI_val - scaM_new_val) * 1.4426950408889634).to(DTYPE)
    # tl.static_print("scaF_act", scaF_act)
    # ? init accumulators
    h_num = tl.zeros((siz_b_DHHV,), dtype=tl.float32)
    qn_dotproduct = tl.zeros((1,), dtype=tl.float32)

    NUM_BLOCKS_DQK = triton.cdiv(DHQK, siz_b_DHQK)

    for i_dhqk in range(NUM_BLOCKS_DQK):
        vecN_old_ptr = vecN_old + i_bnh * str_vecN_B_NH + i_dhqk * siz_b_DHQK * str_vecN_DHQK + tl.arange(0, siz_b_DHQK)
        vecN_new_ptr = vecN_new + i_bnh * str_vecN_B_NH + i_dhqk * siz_b_DHQK * str_vecN_DHQK + tl.arange(0, siz_b_DHQK)

        vecQ_ptr = vecQ + i_bnh * str_vecQK_NH + i_dhqk * siz_b_DHQK * str_vecQK_DHQK + tl.arange(0, siz_b_DHQK)
        vecK_ptr = vecK + i_bnh * str_vecQK_NH + i_dhqk * siz_b_DHQK * str_vecQK_DHQK + tl.arange(0, siz_b_DHQK)
        vecV_ptr = vecV + i_bnh * str_vecVH_B_NH + i_dhv * siz_b_DHHV * str_vecVH_DHHV + tl.arange(0, siz_b_DHHV)

        # update rule
        vecK_val = tl.load(vecK_ptr)
        vecV_val = tl.load(vecV_ptr)
        matC_old_val = tl.load(matC_old_bptr, boundary_check=(0, 1), padding_option="zero")

        matC_new_val = scaF_act * matC_old_val + scaI_act * (vecK_val[:, None] * vecV_val[None, :])

        vecN_new_val = scaF_act * tl.load(vecN_old_ptr) + scaI_act * vecK_val
        # ? Store data
        tl.store(
            matC_new_bptr,
            matC_new_val.to(matC_new.type.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(vecN_new_ptr, vecN_new_val.to(vecN_new.type.element_ty))

        # ? advance pointers
        matC_old_bptr = tl.advance(matC_old_bptr, (siz_b_DHQK, 0))
        matC_new_bptr = tl.advance(matC_new_bptr, (siz_b_DHQK, 0))

        # ? accumulate h_num & qn_dotproduct
        vecQ_val = tl.load(vecQ_ptr) * qk_scale
        # outputs
        h_num_temp = vecQ_val[:, None] * matC_new_val
        # we keep h_num and qn_dotproduct in float32 as they are accumulated
        h_num += tl.sum(h_num_temp, axis=0)
        qn_dotproduct += tl.sum(vecQ_val * vecN_new_val)

    # we compute h in float32 and then cast to DTYPE
    h_denom = tl.maximum(tl.abs(qn_dotproduct), max_val) + EPS
    h = tl.fdiv(h_num, h_denom)

    # ? Store data
    tl.store(vecH_ptr, h.to(DTYPE))


@contiguous_noctx
def recurrent_step_fw(
    matC_old: torch.Tensor,  # (B, NH, DHQK, DHHV)
    vecN_old: torch.Tensor,  # (B, NH, DHQK)
    scaM_old: torch.Tensor,  # (B, NH, 1)
    vecQ: torch.Tensor,  # (B, NH, DHQK)
    vecK: torch.Tensor,  # (B, NH, DHQK)
    vecV: torch.Tensor,  # (B, NH, DHHV)
    scaI: torch.Tensor,  # (B, NH, 1)
    scaF: torch.Tensor,  # (B, NH, 1)
    matC_new: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    vecN_new: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_new: torch.Tensor = None,  # (B, NH, 1)
    qk_scale: float = None,
    eps: float = 1e-6,
    siz_b_DHQK: int | None = None,
    siz_b_DHHV: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
):
    B, NH, DHQK, DHHV = matC_old.shape

    DTYPE = matC_old.dtype

    if qk_scale is None:
        qk_scale = 1 / math.sqrt(DHQK)

    if matC_new is None:
        assert vecN_new is None and scaM_new is None, "Initial states must be provided together."
        matC_new = torch.empty_like(matC_old)
        vecN_new = torch.empty_like(vecN_old)
        scaM_new = torch.empty_like(scaM_old)
    else:
        assert vecN_new is not None and scaM_new is not None, "Initial states must be provided together."

    min_siz_b_DHQK = 64
    min_siz_b_DHHV = 64

    assert (
        is_power_of_2(DHQK) or DHQK % min_siz_b_DHQK == 0
    ), f"DHQK must be a power of 2 or multiple of {min_siz_b_DHQK}. Got {DHQK}."
    assert (
        is_power_of_2(DHHV) or DHHV % min_siz_b_DHHV == 0
    ), f"DHHV must be a power of 2 or multiple of {min_siz_b_DHHV}. Got {DHHV}."

    siz_b_DHQK = min(min_siz_b_DHQK, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(min_siz_b_DHHV, triton.next_power_of_2(DHHV))

    # num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    grid = (1, num_b_DHHV, B * NH)
    if num_warps is None:
        num_warps = 4 if siz_b_DHQK >= 64 else 2

    num_stages = 1 if num_stages is None else num_stages

    # create output tensors
    vecH = torch.empty_like(vecV)

    _recurrent_step_fw_kernel[grid](
        matC_old=matC_old,
        vecN_old=vecN_old,
        scaM_old=scaM_old,
        vecQ=vecQ,
        vecK=vecK,
        vecV=vecV,
        scaI=scaI,
        scaF=scaF,
        vecH=vecH,
        matC_new=matC_new,
        vecN_new=vecN_new,
        scaM_new=scaM_new,
        qk_scale=qk_scale,
        str_matC_B_NH=matC_old.stride(1),
        str_matC_DHQK=matC_old.stride(2),
        str_matC_DHHV=matC_old.stride(3),
        str_vecN_B_NH=vecN_old.stride(1),
        str_vecN_DHQK=vecN_old.stride(2),
        str_scaM_B_NH=scaM_old.stride(1),
        str_vecQK_NH=vecQ.stride(1),
        str_vecQK_DHQK=vecQ.stride(2),
        str_vecVH_B_NH=vecV.stride(1),
        str_vecVH_DHHV=vecV.stride(2),
        str_scaIF_B_NH=scaI.stride(1),
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        EPS=eps,
        DTYPE=torch2triton_dtype(DTYPE),
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return vecH, (matC_new, vecN_new, scaM_new)
