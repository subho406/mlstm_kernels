# Copyright JKU Linz 2024
# Author: Maximilian Beck
from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange

from ....kernel_utils import contiguous_noctx, is_power_of_2, torch2triton_dtype

"""Triton.

Forward pass of the mLSTM chunkwise formulation.

Notation:
Dimensions:
    B: batch size
    NH: number of heads
    S: sequence length (K, V)
    T: sequence length (Q)
    DHQK: hidden dimension (Q, K)
    DHHV: hidden dimension (H, V)
    NC: number of chunks
    L: chunk size

Variables:
    vecA, a: forward gate contribution, contribution of forget gates from last chunk state C_{k-1} to current timestep t
    vecB, b: backward gate contribution, contribution of forget and input gates up to next chunk state C_k (form current timestep t)
    scaG, g: "go through" gate contribution, contribution of forget gates from C_{k-1} to C_k.
    matD, D: gating matrix for the parallel form.
"""


# TODO use the strides in the pointers for generic use case (even if they are 1 normally)
# Note: we only pass stride for the head dimension (we do not access individual batch elements directly)
@triton.jit
def _mlstm_chunkwise__recurrent_fw_C_kernel(
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    vecB,  # (B, NH, NC, L)
    vecI,  # (B, NH, NC, L)
    matC_states,  # (B, NH, (NC + 1) * DHQK, DHHV)
    vecN_states,  # (B, NH, (NC + 1) * DHQK)
    scaMinter_states,  # (B, NH, (NC + 1))
    matC_initial,  # (B, NH, DHQK, DHHV)
    vecN_initial,  # (B, NH, DHQK)
    scaMinter_initial,  # (B, NH)
    str_matK_B_NH,
    str_matK_S,
    str_matK_DHQK,
    str_matV_B_NH,
    str_matV_S,
    str_matV_DHHV,
    str_vecBI_B_NH,
    str_vecBI_NC,
    str_vecBI_L,
    str_matCstates_B_NH,
    str_matCstates_NCDHQK,
    str_matCstates_DHHV,
    str_vecNstates_B_NH,
    str_vecNstates_NCDHQK,
    str_scaMinterstates_B_NH,
    str_scaMinterstates_NC,
    str_matCinitial_B_NH,
    str_matCinitial_DHQK,
    str_matCinitial_DHHV,
    str_vecNinitial_B_NH,
    str_vecNinitial_DHQK,
    str_scaMinterinitial_B_NH,
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
):
    idx_b_DHQK, idx_b_DHHV, idx_b_BNH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )

    # create running states in shared memory
    matC_k_val = tl.zeros((siz_b_DHQK, siz_b_DHHV), dtype=tl.float32)
    vecN_k_val = tl.zeros((siz_b_DHQK,), dtype=tl.float32)
    scaMinter_k_val = 0.0  # tl.zeros((1,), dtype=tl.float32)
    # scaMinter_next_val = tl.zeros((1,), dtype=tl.float32) # TODO we create this in the loop

    if USE_INITIAL_STATE:
        # each thread block loads a (siz_b_DHQK, siz_b_DHHV) block from matC_initial
        matCinitial_ptr = tl.make_block_ptr(
            base=matC_initial + idx_b_BNH * str_matCinitial_B_NH,
            shape=(DHQK, DHHV),
            strides=(str_matCinitial_DHQK, str_matCinitial_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        # each thread block loads a (siz_b_DHQK,) chunk from vecN_initial
        vecNinitial_ptr = (
            vecN_initial + idx_b_BNH * str_vecNinitial_B_NH + idx_b_DHQK * siz_b_DHQK + tl.arange(0, siz_b_DHQK)
        )
        # each thread block loads the scaMinter_initial
        scaMinterinitial_ptr = scaMinter_initial + idx_b_BNH * str_scaMinterinitial_B_NH

        # load initial states
        matC_k_val = tl.load(matCinitial_ptr, boundary_check=(0, 1)).to(tl.float32)
        vecN_k_val = tl.load(vecNinitial_ptr).to(tl.float32)
        scaMinter_k_val = tl.load(scaMinterinitial_ptr).to(tl.float32)

    # iterate over chunks
    for k in range(NC):
        # load matK in transposed form
        matK_k_ptr = tl.make_block_ptr(
            base=matK + idx_b_BNH * str_matK_B_NH,
            shape=(DHQK, S),
            strides=(str_matK_DHQK, str_matK_S),
            offsets=(idx_b_DHQK * siz_b_DHQK, k * L),
            block_shape=(siz_b_DHQK, L),
            order=(0, 1),
        )
        matV_k_ptr = tl.make_block_ptr(
            base=matV + idx_b_BNH * str_matV_B_NH,
            shape=(S, DHHV),
            strides=(str_matV_S, str_matV_DHHV),
            offsets=(k * L, idx_b_DHHV * siz_b_DHHV),
            block_shape=(L, siz_b_DHHV),
            order=(1, 0),
        )
        # create pointer for matCstates_k, vecNstates_k, scaMinterstates_k
        # each thread block stores a (siz_b_DHQK, siz_b_DHHV) block to matC_states_k
        matCstates_k_ptr = tl.make_block_ptr(
            base=matC_states + idx_b_BNH * str_matCstates_B_NH + k * DHQK * DHHV,
            shape=(DHQK, DHHV),
            strides=(str_matCstates_NCDHQK, str_matCstates_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        vecNstates_k_ptr = (
            vecN_states
            + idx_b_BNH * str_vecNstates_B_NH
            + k * DHQK
            + idx_b_DHQK * siz_b_DHQK
            + tl.arange(0, siz_b_DHQK)
        )
        scaMinterstates_k_ptr = scaMinter_states + idx_b_BNH * str_scaMinterstates_B_NH + k

        # store the states from the previous iteration
        tl.store(matCstates_k_ptr, matC_k_val.to(dtype=DTYPE), boundary_check=(0, 1))
        if idx_b_DHHV == 0:
            tl.store(vecNstates_k_ptr, vecN_k_val.to(dtype=DTYPE))  # TODO add mask for boundary check
        if (idx_b_DHQK == 0) and (idx_b_DHHV == 0):
            tl.store(scaMinterstates_k_ptr, scaMinter_k_val.to(dtype=DTYPE))

        # load / compute vecA_k, scaG_k
        # last element of vecB in k-th chunk
        vecB_last_k_val = tl.load(vecB + idx_b_BNH * str_vecBI_B_NH + k * str_vecBI_NC + (L - 1)).to(tl.float32)
        vecB_k_val = tl.load(vecB + idx_b_BNH * str_vecBI_B_NH + k * str_vecBI_NC + tl.arange(0, L)).to(tl.float32)

        vecI_k_val = tl.load(vecI + idx_b_BNH * str_vecBI_B_NH + k * str_vecBI_NC + tl.arange(0, L)).to(tl.float32)

        vecA_k_val = (vecB_last_k_val - vecB_k_val) + vecI_k_val
        scaG_k_val = vecB_last_k_val

        # scaM_inter_k update
        scaAmax_k_val, _ = tl.max(vecA_k_val)
        scaMinter_next_val = tl.maximum(scaG_k_val + scaMinter_k_val, scaAmax_k_val)

        # load matK_k, matV_k
        matK_k_val = tl.load(matK_k_ptr, boundary_check=(0, 1)).to(tl.float32)
        matV_k_val = tl.load(matV_k_ptr, boundary_check=(0, 1)).to(DTYPE)

        # matC_k update
        vecAbar_k_val = tl.exp(vecA_k_val - scaMinter_next_val)
        scaGbar_k_val = tl.exp(scaG_k_val + scaMinter_k_val - scaMinter_next_val)

        matKbar_k_val = (matK_k_val * vecAbar_k_val[None, :]).to(DTYPE)

        matC_k_val = scaGbar_k_val * matC_k_val + tl.dot(matKbar_k_val, matV_k_val)

        # vecN_k update
        vecN_k_val = scaGbar_k_val * vecN_k_val + tl.sum(matKbar_k_val, axis=1)

        # move to next iteration
        scaMinter_k_val = scaMinter_next_val

    # store the states from the last iteration
    matCstates_k_ptr = tl.make_block_ptr(
        base=matC_states + idx_b_BNH * str_matCstates_B_NH + NC * DHQK * DHHV,
        shape=(DHQK, DHHV),
        strides=(str_matCstates_NCDHQK, str_matCstates_DHHV),
        offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
        block_shape=(siz_b_DHQK, siz_b_DHHV),
        order=(1, 0),
    )
    vecNstates_k_ptr = (
        vecN_states + idx_b_BNH * str_vecNstates_B_NH + NC * DHQK + idx_b_DHQK * siz_b_DHQK + tl.arange(0, siz_b_DHQK)
    )
    scaMinterstates_k_ptr = scaMinter_states + idx_b_BNH * str_scaMinterstates_B_NH + NC
    tl.store(matCstates_k_ptr, matC_k_val.to(dtype=DTYPE), boundary_check=(0, 1))
    if idx_b_DHHV == 0:
        tl.store(vecNstates_k_ptr, vecN_k_val.to(dtype=DTYPE))  # TODO add mask for boundary check
    if (idx_b_DHQK == 0) and (idx_b_DHHV == 0):
        tl.store(scaMinterstates_k_ptr, scaMinter_k_val.to(dtype=DTYPE))


def _mlstm_chunkwise__recurrent_fw_C(
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    matC_states: torch.Tensor = None,  # (B, NH, (NC + 1) * DHQK, DHHV)
    vecN_states: torch.Tensor = None,  # (B, NH, (NC + 1) * DHQK)
    scaMinter_states: torch.Tensor = None,  # (B, NH, (NC + 1)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaMinter_initial: torch.Tensor = None,  # (B, NH)
    qk_scale: float = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:  # matC_states (B, NH, (NC+1) * DHQK, DHHV), vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1))
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[-1]

    NC = NUM_CHUNKS
    L = CHUNK_SIZE

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1
    num_warps = 4 if siz_b_DHQK == 64 else 2

    USE_INITIAL_STATE = matC_initial is not None
    if USE_INITIAL_STATE:
        assert vecN_initial is not None and scaMinter_initial is not None
        str_matCinitial_B_NH = matC_initial.stride(1)
        str_matCinitial_DHQK = matC_initial.stride(2)
        str_matCinitial_DHHV = matC_initial.stride(3)
        str_vecNinitial_B_NH = vecN_initial.stride(1)
        str_vecNinitial_DHQK = vecN_initial.stride(2)
        str_scaMinterinitial_B_NH = scaMinter_initial.stride(1)
    else:
        str_matCinitial_B_NH = 0
        str_matCinitial_DHQK = 0
        str_matCinitial_DHHV = 0
        str_vecNinitial_B_NH = 0
        str_vecNinitial_DHQK = 0
        str_scaMinterinitial_B_NH = 0

    matC_states = (
        torch.empty(B, NH, (NC + 1) * DHQK, DHHV, device=matK.device, dtype=matK.dtype)
        if matC_states is None
        else matC_states
    )
    vecN_states = (
        torch.empty(B, NH, (NC + 1) * DHQK, device=matK.device, dtype=matK.dtype)
        if vecN_states is None
        else vecN_states
    )
    scaMinter_states = (
        torch.empty(B, NH, (NC + 1), device=matK.device, dtype=matK.dtype)
        if scaMinter_states is None
        else scaMinter_states
    )

    grid = (num_b_DHQK, num_b_DHHV, B * NH)
    _mlstm_chunkwise__recurrent_fw_C_kernel[grid](
        matK=matK,
        matV=matV,
        vecB=vecB,
        vecI=vecI,
        matC_states=matC_states,
        vecN_states=vecN_states,
        scaMinter_states=scaMinter_states,
        matC_initial=matC_initial,
        vecN_initial=vecN_initial,
        scaMinter_initial=scaMinter_initial,
        str_matK_B_NH=matK.stride(1),
        str_matK_S=matK.stride(2),
        str_matK_DHQK=matK.stride(3),
        str_matV_B_NH=matV.stride(1),
        str_matV_S=matV.stride(2),
        str_matV_DHHV=matV.stride(3),
        str_vecBI_B_NH=vecB.stride(1),
        str_vecBI_NC=vecB.stride(2),
        str_vecBI_L=vecB.stride(3),
        str_matCstates_B_NH=matC_states.stride(1),
        str_matCstates_NCDHQK=matC_states.stride(2),
        str_matCstates_DHHV=matC_states.stride(3),
        str_vecNstates_B_NH=vecN_states.stride(1),
        str_vecNstates_NCDHQK=vecN_states.stride(2),
        str_scaMinterstates_B_NH=scaMinter_states.stride(1),
        str_scaMinterstates_NC=scaMinter_states.stride(2),
        str_matCinitial_B_NH=str_matCinitial_B_NH,
        str_matCinitial_DHQK=str_matCinitial_DHQK,
        str_matCinitial_DHHV=str_matCinitial_DHHV,
        str_vecNinitial_B_NH=str_vecNinitial_B_NH,
        str_vecNinitial_DHQK=str_vecNinitial_DHQK,
        str_scaMinterinitial_B_NH=str_scaMinterinitial_B_NH,
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        USE_INITIAL_STATE=USE_INITIAL_STATE,
        DTYPE=torch2triton_dtype(matK.dtype),
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return matC_states, vecN_states, scaMinter_states


@triton.jit
def _mlstm_chunkwise_parallel_fw_H_kernel(
    matQ,  # (B, NH, S, DHQK)
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    matC_states,  # (B, NH, NC * DHQK, DHHV)
    vecN_states,  # (B, NH, NC * DHQK)
    scaMinter_states,  # (B, NH, NC)
    vecI,  # (B, NH, NC, L)
    vecB,  # (B, NH, NC, L)
    matHout,  # (B, NH, S, DHHV)
    vecNout,  # (B, NH, S)
    vecMout,  # (B, NH, S)
    qk_scale,
    str_matQK_B_NH,
    str_matQK_S,
    str_matQK_DHQK,
    str_matHV_B_NH,
    str_matHV_S,
    str_matHV_DHHV,
    str_matCstates_B_NH,
    str_matCstates_NCDHQK,
    str_matCstates_DHHV,
    str_vecNstates_B_NH,
    str_vecNstates_NCDHQK,
    str_scaMinterstates_B_NH,
    str_vecBI_B_NH,
    str_vecBI_NC,
    str_vecBI_L,
    str_vecMN_B_NH,
    str_vecMN_S,
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
    EPS: tl.constexpr = 1e-6,
):
    idx_b_DHHV, idx_b_NC, idx_b_BNH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )

    # load vecB (L,)
    vecB_val = tl.load(vecB + idx_b_BNH * str_vecBI_B_NH + idx_b_NC * str_vecBI_NC + tl.arange(0, L)).to(tl.float32)

    # load vecI (L,)
    vecI_val = tl.load(vecI + idx_b_BNH * str_vecBI_B_NH + idx_b_NC * str_vecBI_NC + tl.arange(0, L)).to(tl.float32)

    # load scaMinter_km1 (1,)
    scaMinter_km1_val = tl.load(scaMinter_states + idx_b_BNH * str_scaMinterstates_B_NH + idx_b_NC).to(tl.float32)

    # compute gate matrix matDbar (L, L)
    idx_mask = tl.arange(0, L)
    mask = idx_mask[:, None] >= idx_mask[None, :]
    matD_full_val = vecB_val[:, None] - vecB_val[None, :] + vecI_val[None, :]
    matD_val = tl.where(mask, matD_full_val, -float("inf"))

    # compute vecM_k_intra (L,) & vecM_k_combine (L,)
    vecM_intra_val = tl.max(matD_val, axis=1)
    vecM_combine_val = tl.maximum(vecB_val + scaMinter_km1_val, vecM_intra_val)
    # tl.static_print("vecM_combine_val", vecM_combine_val[:, None])
    # tl.static_print("matD_val", matD_val)
    matDbar_val = tl.exp(matD_val - vecM_combine_val[:, None])

    # compute vecBbar (L,)
    vecBbar_val = tl.exp(vecB_val + scaMinter_km1_val - vecM_combine_val)
    # tl.static_print("vecBbar_val", vecBbar_val)
    # tl.device_print("vecBbar_val",vecBbar_val)

    ## loop over DHQK blocks
    matS_val = tl.zeros((L, L), dtype=tl.float32)
    matH_inter_val = tl.zeros((L, siz_b_DHHV), dtype=tl.float32)
    vecH_inter_denom_val = tl.zeros((L,), dtype=tl.float32)
    for idx_b_DHQK in range(tl.cdiv(DHQK, siz_b_DHQK)):
        # tl.device_print("idx_b_DHQK", idx_b_DHQK)
        # define pointers for iteration
        matQ_ptr = tl.make_block_ptr(
            base=matQ + idx_b_BNH * str_matQK_B_NH,
            shape=(S, DHQK),
            strides=(str_matQK_S, str_matQK_DHQK),
            offsets=(idx_b_NC * L, idx_b_DHQK * siz_b_DHQK),
            block_shape=(L, siz_b_DHQK),
            order=(1, 0),
        )
        matK_ptr = tl.make_block_ptr(
            base=matK + idx_b_BNH * str_matQK_B_NH,
            shape=(DHQK, S),
            strides=(str_matQK_DHQK, str_matQK_S),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_NC * L),
            block_shape=(siz_b_DHQK, L),
            order=(0, 1),
        )
        matC_km1_ptr = tl.make_block_ptr(
            base=matC_states + idx_b_BNH * str_matCstates_B_NH + idx_b_NC * DHQK * DHHV,
            shape=(DHQK, DHHV),
            strides=(str_matCstates_NCDHQK, str_matCstates_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        vecN_km1_ptr = (
            vecN_states
            + idx_b_BNH * str_vecNstates_B_NH
            + idx_b_NC * DHQK
            + idx_b_DHQK * siz_b_DHQK
            + tl.arange(0, siz_b_DHQK)
        )

        # load matQ block (L, siz_b_DHQK)
        matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1)).to(DTYPE)
        # load matK transposed block (L, siz_b_DHQK)
        matK_val = tl.load(matK_ptr, boundary_check=(0, 1)).to(DTYPE)

        # accumulate matS (L, L)
        matS_val += tl.dot(matQ_val, matK_val) * qk_scale

        # compute matQbar (L, siz_b_DHQK)
        # tl.static_print("matQ_val", matQ_val)
        # tl.static_print("vecBbar_val", vecBbar_val[:, None])
        # tl.static_print("qk_scale", qk_scale)
        matQbar_val = (matQ_val * vecBbar_val[:, None] * qk_scale).to(DTYPE)

        # load matC_kminus1_tile (siz_b_DHQK, siz_b_DHHV)
        matC_km1_val = tl.load(matC_km1_ptr, boundary_check=(0, 1)).to(DTYPE)
        # accumulate matH_k_inter (L, siz_b_DHHV)
        matH_inter_val += tl.dot(matQbar_val, matC_km1_val)

        # load vecN_km1 (siz_b_DHQK,)
        vecN_km1_val = tl.load(vecN_km1_ptr).to(DTYPE)
        # accumulate vecH_k_inter_denom (L,)
        vecH_inter_denom_val += tl.sum(matQbar_val * vecN_km1_val[None, :], axis=1)

    ## loop end

    # compute matSbar (L, L)
    matSbar_val = (matS_val * matDbar_val).to(DTYPE)

    # load matV (L, siz_b_DHHV)
    matV_ptr = tl.make_block_ptr(
        base=matV + idx_b_BNH * str_matHV_B_NH,
        shape=(S, DHHV),
        strides=(str_matHV_S, str_matHV_DHHV),
        offsets=(idx_b_NC * L, idx_b_DHHV * siz_b_DHHV),
        block_shape=(L, siz_b_DHHV),
        order=(1, 0),
    )
    matV_val = tl.load(matV_ptr, boundary_check=(0, 1)).to(DTYPE)

    # compute matH_k_intra (L, siz_b_DHHV)
    matH_intra_val = tl.dot(matSbar_val, matV_val)
    # compute vecH_k_intra_denom (L,)
    vecH_intra_denom_val = tl.sum(matSbar_val, axis=1)

    # compute matH_k_num (L, siz_b_DHHV)
    matH_num_val = matH_inter_val + matH_intra_val

    # compute H_k_denom (L,)
    vecH_denom_val = tl.maximum(tl.abs(vecH_inter_denom_val + vecH_intra_denom_val), tl.exp(-vecM_combine_val))

    # compute matH_k_out (L, siz_b_DHHV)
    matHout_val = matH_num_val / (vecH_denom_val[:, None] + EPS)

    # store matH_k_out, vecN_k_denom, vecM_k_combine
    matHout_ptr = tl.make_block_ptr(
        base=matHout + idx_b_BNH * str_matHV_B_NH,
        shape=(S, DHHV),
        strides=(str_matHV_S, str_matHV_DHHV),
        offsets=(idx_b_NC * L, idx_b_DHHV * siz_b_DHHV),
        block_shape=(L, siz_b_DHHV),
        order=(1, 0),
    )
    vecNout_ptr = vecNout + idx_b_BNH * str_vecMN_B_NH + (idx_b_NC * L + tl.arange(0, L)) * str_vecMN_S
    vecMout_ptr = vecMout + idx_b_BNH * str_vecMN_B_NH + (idx_b_NC * L + tl.arange(0, L)) * str_vecMN_S
    tl.store(matHout_ptr, matHout_val.to(DTYPE), boundary_check=(0, 1))
    tl.store(vecNout_ptr, vecH_denom_val.to(DTYPE))
    tl.store(vecMout_ptr, vecM_combine_val.to(DTYPE))


"""Debug plan _mlstm_chunkwise_parallel_fw_H_kernel.

After first attempt (all inputs randn, one chunk, dhqk=dhv):
- vecMout match: vecBbar val is correct, vecM_combine val is correct, vecMout val is correct
- vecNout does not match: ???
- matHout does not match: ???

Fix 1: the Sbar matrix was wrong. too late yesterday probably :D
Fix 2: pointer to matC_km1 was wrong. fixed now.
Fix 3: pointer to vecN_km1 was wrong. fixed now.
Added also output deviation plots. Looks good now.
"""


def _mlstm_chunkwise__parallel_fw_H(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    # these states must be all states up to the last chunk, i.e. :-1
    matC_states: torch.Tensor,  # (B, NH, NC * DHQK, DHHV)
    vecN_states: torch.Tensor,  # (B, NH, NC * DHQK)
    scaMinter_states: torch.Tensor,  # (B, NH, NC)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    qk_scale: float = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:  # matH_out (B, NH, S, DHHV), vecN_out (B, NH, S)
    """This function defines the grid and block sizes for the kernel launch and calls the kernel."""
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[-1]

    NC = NUM_CHUNKS
    L = CHUNK_SIZE

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1
    num_warps = 4 if siz_b_DHQK == 64 else 2

    # TODO make these empty
    matH_out = torch.empty(B, NH, S, DHHV, device=matQ.device, dtype=matQ.dtype)
    vecN_out = torch.empty(B, NH, S, device=matQ.device, dtype=matQ.dtype)
    vecM_out = torch.empty(B, NH, S, device=matQ.device, dtype=matQ.dtype)

    grid = (num_b_DHHV, NC, B * NH)
    _mlstm_chunkwise_parallel_fw_H_kernel[grid](
        matQ=matQ,
        matK=matK,
        matV=matV,
        matC_states=matC_states,
        vecN_states=vecN_states,
        scaMinter_states=scaMinter_states,
        vecI=vecI,
        vecB=vecB,
        matHout=matH_out,
        vecNout=vecN_out,
        vecMout=vecM_out,
        qk_scale=qk_scale,
        str_matQK_B_NH=matQ.stride(1),
        str_matQK_S=matQ.stride(2),
        str_matQK_DHQK=matQ.stride(3),
        str_matHV_B_NH=matV.stride(1),
        str_matHV_S=matV.stride(2),
        str_matHV_DHHV=matV.stride(3),
        str_matCstates_B_NH=matC_states.stride(1),
        str_matCstates_NCDHQK=matC_states.stride(2),
        str_matCstates_DHHV=matC_states.stride(3),
        str_vecNstates_B_NH=vecN_states.stride(1),
        str_vecNstates_NCDHQK=vecN_states.stride(2),
        str_scaMinterstates_B_NH=scaMinter_states.stride(1),
        str_vecBI_B_NH=vecB.stride(1),
        str_vecBI_NC=vecB.stride(2),
        str_vecBI_L=vecB.stride(3),
        str_vecMN_B_NH=vecN_out.stride(1),
        str_vecMN_S=vecN_out.stride(2),
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        DTYPE=torch2triton_dtype(matQ.dtype),
        EPS=EPS,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return matH_out, vecN_out, vecM_out


@contiguous_noctx
def _mlstm_chunkwise_fw(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH)
    qk_scale: float = None,
    return_last_states: bool = False,
    return_all_states: bool = False,
    CHUNK_SIZE: int = 64,
    EPS: float = 1e-6,
) -> tuple[
    torch.Tensor,  # matH_out (B, NH, S, DHV)
    torch.Tensor,  # vecN_out (B, NH, S)
    torch.Tensor,  # vecM_out (B, NH, S)
    None
    | (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ),  # last_states (matC_states (B, NH, DHQK, DHV), vecN_states (B, NH, DHQK), scaMinter_states (B, NH))
    None
    | (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ),  # all_states (matC_states (B, NH, (NC+1) * DHQK, DHV), vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1)))
]:
    B, NH, S, DHQK = matQ.shape
    DHV = matV.shape[-1]
    assert S % CHUNK_SIZE == 0, f"Sequence length {S} is not divisible by chunk size {CHUNK_SIZE}."
    NC = S // CHUNK_SIZE

    vecI = rearrange(vecI, "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE)
    vecF = rearrange(vecF, "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE)

    # compute the gates, the g and the a and b vectors
    vecF_logsig = F.logsigmoid(vecF)
    vecB = vecF_logsig.cumsum(-1)

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    #! materialize the  C_k, n_k, m_k states for each chunk
    matC_k_states, vecN_k_states, scaMinter_k_states = _mlstm_chunkwise__recurrent_fw_C(
        matK=matK,
        matV=matV,
        vecB=vecB,
        vecI=vecI,
        matC_initial=matC_initial,
        vecN_initial=vecN_initial,
        scaMinter_initial=scaM_initial,
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
    )

    #! compute the outputs within each chunk
    matH_out, vecN_out, vecM_out = _mlstm_chunkwise__parallel_fw_H(
        matQ=matQ,
        matK=matK,
        matV=matV,
        matC_states=matC_k_states[:, :, :-DHQK, :],
        vecN_states=vecN_k_states[:, :, :-DHQK],
        scaMinter_states=scaMinter_k_states[:, :, :-1],
        vecI=vecI,
        vecB=vecB,
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )

    ret_tuple = (
        matH_out,
        vecN_out,
        vecM_out,
    )
    if return_last_states:
        ret_tuple += (
            (
                matC_k_states[:, :, -DHQK:, :],
                vecN_k_states[:, :, -DHQK:],
                scaMinter_k_states[:, :, -1],
            ),
        )
    else:
        ret_tuple += (None,)

    if return_all_states:
        ret_tuple += ((matC_k_states, vecN_k_states, scaMinter_k_states),)
    else:
        ret_tuple += (None,)

    return ret_tuple  # (matH_out, vecN_out, vecM_out, optional(last_states), optional(all_states))
