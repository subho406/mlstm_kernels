# Copyright JKU Linz 2024
# Author: Maximilian Beck
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from ...kernel_utils import contiguous

"""
Triton.

This module contains the recurrent step of the mLSTM in triton.

We want to compare this to the torch implementation in mlstm_kernels/mlstm/recurrent/torch_fw.py.

# TODO we cannot use this fully parallel work partitioning for the step
# we have to sum over the qk dimension for the outputs..

# either have two kernels or one kernel with a loop over the qk dimension
"""

@triton.jit
def _recurrent_step_fw_kernel_C(
    matC_old_val,  # (B, NH, DHQK, DHV)
    vecN_old,  # (B, NH, DHQK)
    scaM_old,  # (B, NH, 1)
    vecK,  # (B, NH, DHQK)
    vecV,  # (B, NH, DHV)
    scaI,  # (B, NH, 1)
    scaF,  # (B, NH, 1)
    matC_new,  # (B, NH, DHQK, DHV)
    vecN_new,  # (B, NH, DHQK)
    scaM_new,  # (B, NH, 1)
    qk_scale,
    s_matC_b,
    s_matC_nh,
    s_matC_dhqk,
    s_matC_dhv,
    s_vecN_b,
    s_vecN_nh,
    s_vecN_dhqk,
    s_scaM_b,
    s_scaM_nh,
    s_vecQK_b,
    s_vecQK_nh,
    s_vecQK_dhqk,
    s_vecVH_b,
    s_vecVH_nh,
    s_vecVH_dhv,
    s_scaIF_b,
    s_scaIF_nh,
    B,
    NH,
    DHQK: tl.constexpr,
    DHV: tl.constexpr,
    BLOCK_DQK: tl.constexpr,  # DHQK = BLOCK_DQK * NUM_BLOCKS_DQK
    BLOCK_DV: tl.constexpr,  # DHV = BLOCK_DV * NUM_BLOCKS_DV
    EPS: tl.constexpr = 1e-6,
):
    i_dhqk, i_dhv, i_bnh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    # ? Define pointers
    matC_old_bptr = tl.make_block_ptr(
        base=matC_old_val + i_bnh * s_matC_nh,
        shape=(DHQK, DHV),
        strides=(s_matC_dhqk, s_matC_dhv),
        offsets=(i_dhqk * BLOCK_DQK, i_dhv * BLOCK_DV),
        block_shape=(BLOCK_DQK, BLOCK_DV),
        order=(0, 1),
    )
    matC_new_bptr = tl.make_block_ptr(
        base=matC_new + i_bnh * s_matC_nh,
        shape=(DHQK, DHV),
        strides=(s_matC_dhqk, s_matC_dhv),
        offsets=(i_dhqk * BLOCK_DQK, i_dhv * BLOCK_DV),
        block_shape=(BLOCK_DQK, BLOCK_DV),
        order=(0, 1),
    )

    vecN_old_ptr = (
        vecN_old
        + i_bnh * s_vecN_nh
        + i_dhqk * BLOCK_DQK * s_vecN_dhqk
        + tl.arange(0, BLOCK_DQK)
    )
    vecN_new_ptr = (
        vecN_new
        + i_bnh * s_vecN_nh
        + i_dhqk * BLOCK_DQK * s_vecN_dhqk
        + tl.arange(0, BLOCK_DQK)
    )

    scaM_old_ptr = scaM_old + i_bnh * s_scaM_nh
    scaM_new_ptr = scaM_new + i_bnh * s_scaM_nh

    vecK_ptr = (
        vecK
        + i_bnh * s_vecQK_nh
        + i_dhqk * BLOCK_DQK * s_vecQK_dhqk
        + tl.arange(0, BLOCK_DQK)
    )
    vecV_ptr = (
        vecV
        + i_bnh * s_vecVH_nh
        + i_dhv * BLOCK_DV * s_vecVH_dhv
        + tl.arange(0, BLOCK_DV)
    )

    scaI_ptr = scaI + i_bnh * s_scaIF_nh
    scaF_ptr = scaF + i_bnh * s_scaIF_nh

    # ? Load data
    # gates
    scaF_val = tl.load(scaF_ptr)
    scaI_val = tl.load(scaI_ptr)

    scaFlog_val = tl.log(tl.sigmoid(scaF_val))

    # update rule
    scaM_new_val = tl.maximum(scaFlog_val + tl.load(scaM_old_ptr), scaI_val)

    scaF_act = tl.exp(scaFlog_val + tl.load(scaM_old_ptr) - scaM_new_val)
    scaI_act = tl.exp(scaI_val - scaM_new_val)

    # TODO add masking to avoid out of bound access
    vecK_val_scaled = tl.load(vecK_ptr) * qk_scale
    vecV_val = tl.load(vecV_ptr)

    matC_old_val = tl.load(matC_old_bptr, boundary_check=(0, 1), padding_option="zero")

    matC_new_val = scaF_act * matC_old_val + scaI_act * (
        vecK_val_scaled[:, None] * vecV_val[None, :]
    )

    vecN_new_val = scaF_act * tl.load(vecN_old_ptr) + scaI_act * vecK_val_scaled

    # ? Store data
    tl.store(matC_new_bptr, matC_new_val, boundary_check=(0, 1))
    tl.store(
        vecN_new_ptr, vecN_new_val
    )  # TODO add masking to avoid out of bound access
    tl.store(scaM_new_ptr, scaM_new_val)

@triton.jit
def _recurrent_step_fw_kernel_h(
    vecQ,  # (B, NH, DHQK)
    vecH,  # (B, NH, DHV)
    matC_new,  # (B, NH, DHQK, DHV)
    vecN_new,  # (B, NH, DHQK)
    scaM_new,  # (B, NH, 1)
    s_matC_b,
    s_matC_nh,
    s_matC_dhqk,
    s_matC_dhv,
    s_vecN_b,
    s_vecN_nh,
    s_vecN_dhqk,
    s_scaM_b,
    s_scaM_nh,
    s_vecQK_b,
    s_vecQK_nh,
    s_vecQK_dhqk,
    s_vecVH_b,
    s_vecVH_nh,
    s_vecVH_dhv,
    s_scaIF_b,
    s_scaIF_nh,
    B,
    NH,
    DHQK: tl.constexpr,
    DHV: tl.constexpr,
    BLOCK_DQK: tl.constexpr,  # DHQK = BLOCK_DQK * NUM_BLOCKS_DQK
    BLOCK_DV: tl.constexpr,  # DHV = BLOCK_DV * NUM_BLOCKS_DV
    EPS: tl.constexpr = 1e-6,
):
    i_dhv, i_bnh = tl.program_id(1), tl.program_id(2)

    # ? Define pointers
    matC_new_bptr = tl.make_block_ptr(
        base=matC_new + i_bnh * s_matC_nh,
        shape=(DHQK, DHV),
        strides=(s_matC_dhqk, s_matC_dhv),
        offsets=(0, i_dhv * BLOCK_DV),
        block_shape=(BLOCK_DQK, BLOCK_DV),
        order=(0, 1),
    )
    scaM_new_ptr = scaM_new + i_bnh * s_scaM_nh
    scaM_new_val = tl.load(scaM_new_ptr)
    vecH_ptr = (
        vecH
        + i_bnh * s_vecVH_nh
        + i_dhv * BLOCK_DV * s_vecVH_dhv
        + tl.arange(0, BLOCK_DV)
    )

    h_num = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    qn_dotproduct = tl.zeros((1,), dtype=tl.float32)
    
    NUM_BLOCKS_DQK = triton.cdiv(DHQK, BLOCK_DQK)

    for i_dhqk in range(NUM_BLOCKS_DQK):

        vecN_new_ptr = (
            vecN_new
            + i_bnh * s_vecN_nh
            + i_dhqk * BLOCK_DQK * s_vecN_dhqk
            + tl.arange(0, BLOCK_DQK)
        )

        vecQ_ptr = (
            vecQ
            + i_bnh * s_vecQK_nh
            + i_dhqk * BLOCK_DQK * s_vecQK_dhqk
            + tl.arange(0, BLOCK_DQK)
        )

        # ? Load data
        matC_new_val = tl.load(matC_new_bptr, boundary_check=(0, 1), padding_option="zero")
        vecN_new_val = tl.load(vecN_new_ptr)

        vecQ_val = tl.load(vecQ_ptr)  # TODO add masking to avoid out of bound access

        # outputs
        h_num_temp = vecQ_val[:, None] * matC_new_val
        h_num += tl.sum(h_num_temp, axis=0)

        qn_dotproduct += tl.sum(vecQ_val * vecN_new_val)
        matC_new_bptr = tl.advance(matC_new_bptr, (BLOCK_DQK,0))

    max_val = tl.exp(-scaM_new_val)
    h_denom = tl.maximum(tl.abs(qn_dotproduct), max_val) + EPS

    h = tl.fdiv(h_num, h_denom)


    # ? Store data
    tl.store(vecH_ptr, h)

@contiguous
def recurrent_step_fw(
    matC_old: torch.Tensor,  # (B, NH, DHQK, DHV)
    vecN_old: torch.Tensor,  # (B, NH, DHQK)
    scaM_old: torch.Tensor,  # (B, NH, 1)
    vecQ: torch.Tensor,  # (B, NH, DHQK)
    vecK: torch.Tensor,  # (B, NH, DHQK)
    vecV: torch.Tensor,  # (B, NH, DHV)
    scaI: torch.Tensor,  # (B, NH, 1)
    scaF: torch.Tensor,  # (B, NH, 1)
    matC_new: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecN_new: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_new: torch.Tensor = None,  # (B, NH, 1)
    qk_scale: float = None,
    DTYPE_GATE: torch.dtype = torch.float32,
    DTYPE_STATE: torch.dtype = torch.float32,
    DTPYE_QKV: torch.dtype = torch.float32,
    EPS: float = 1e-6,
    BLOCK_DQK: int = 16,
    BLOCK_DV: int = 16,
    BLOCK_DQK_H: int = 16,
    BLOCK_DV_H: int = 16,
):
    B, NH, DHQK, DHV = matC_old.shape

    # cast inputs
    matC_old = matC_old.to(DTYPE_STATE)
    vecN_old = vecN_old.to(DTYPE_STATE)
    scaM_old = scaM_old.to(DTYPE_STATE)

    vecQ = vecQ.to(DTPYE_QKV)
    vecK = vecK.to(DTPYE_QKV)
    vecV = vecV.to(DTPYE_QKV)

    scaI = scaI.to(DTYPE_GATE)
    scaF = scaF.to(DTYPE_GATE)

    if qk_scale is None:
        qk_scale = 1 / math.sqrt(DHQK)

    if matC_new is None:
        assert (
            vecN_new is None and scaM_new is None
        ), "Initial states must be provided together."
        matC_new = torch.ones(
            (B, NH, DHQK, DHV), dtype=DTYPE_STATE, device=matC_old.device
        )
        vecN_new = torch.ones((B, NH, DHQK), dtype=DTYPE_STATE, device=matC_old.device)
        scaM_new = torch.ones((B, NH, 1), dtype=DTYPE_STATE, device=matC_old.device)

    def grid_fn_C(*args):
        NUM_BLOCKS_DQK = triton.cdiv(DHQK, BLOCK_DQK)
        NUM_BLOCKS_DV = triton.cdiv(DHV, BLOCK_DV)
        NUM_BATCH_HEAD = B * NH
        grid = (NUM_BLOCKS_DQK, NUM_BLOCKS_DV, NUM_BATCH_HEAD)
        print(grid)
        return grid

    grid_C = grid_fn_C

    # create output tensors
    vecH = torch.ones_like(vecV)

    _recurrent_step_fw_kernel_C[grid_C](
        matC_old,
        vecN_old,
        scaM_old,
        vecK,
        vecV,
        scaI,
        scaF,
        matC_new,
        vecN_new,
        scaM_new,
        qk_scale,
        matC_old.stride(0),
        matC_old.stride(1),
        matC_old.stride(2),
        matC_old.stride(3),
        vecN_old.stride(0),
        vecN_old.stride(1),
        vecN_old.stride(2),
        scaM_old.stride(0),
        scaM_old.stride(1),
        vecQ.stride(0),
        vecQ.stride(1),
        vecQ.stride(2),
        vecV.stride(0),
        vecV.stride(1),
        vecV.stride(2),
        scaI.stride(0),
        scaI.stride(1),
        B,
        NH,
        DHQK,
        DHV,
        BLOCK_DQK,
        BLOCK_DV,
        EPS,
    )

    def grid_fn_h(*args):
        NUM_BLOCKS_DV_H = triton.cdiv(DHV, BLOCK_DV_H)
        NUM_BATCH_HEAD = B * NH
        grid = (1, NUM_BLOCKS_DV_H, NUM_BATCH_HEAD)
        print(grid)
        return grid
    
    grid_h = grid_fn_h

    _recurrent_step_fw_kernel_h[grid_C](
        vecQ,
        vecH,
        matC_new,
        vecN_new,
        scaM_new,
        matC_old.stride(0),
        matC_old.stride(1),
        matC_old.stride(2),
        matC_old.stride(3),
        vecN_old.stride(0),
        vecN_old.stride(1),
        vecN_old.stride(2),
        scaM_old.stride(0),
        scaM_old.stride(1),
        vecQ.stride(0),
        vecQ.stride(1),
        vecQ.stride(2),
        vecV.stride(0),
        vecV.stride(1),
        vecV.stride(2),
        scaI.stride(0),
        scaI.stride(1),
        B,
        NH,
        DHQK,
        DHV,
        BLOCK_DQK_H,
        BLOCK_DV_H,
        EPS,
    )

    return vecH, (matC_new, vecN_new, scaM_new)
