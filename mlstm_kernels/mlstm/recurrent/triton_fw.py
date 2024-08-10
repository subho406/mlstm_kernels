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
"""


@triton.jit
def _recurrent_step_fw_kernel(
    matC_old,  # (B, NH, DHQK, DHV)
    vecN_old,  # (B, NH, DHQK)
    scaM_old,  # (B, NH, 1)
    vecQ,  # (B, NH, DHQK)
    vecK,  # (B, NH, DHQK)
    vecV,  # (B, NH, DHV)
    scaI,  # (B, NH, 1)
    scaF,  # (B, NH, 1)
    vecH,  # (B, NH, DHV)
    matC_new,  # (B, NH, DHQK, DHV)
    vecN_new,  # (B, NH, DHQK)
    scaM_new,  # (B, NH, 1)
    s_matC_b, 
    s_matC_nh,
    s_matC_dhqk,
    s_vecN_b,
    s_vecN_nh,
    s_scaM_b,
    s_scaM_nh,
    s_vecQK_b,
    s_vecQK_nh,
    s_vecVH_b,
    s_vecVH_nh,
    B,
    NH,
    DHQK: tl.constexpr,
    DHV: tl.constexpr,
    BLOCK_DQK: tl.constexpr,  # DHQK = BLOCK_DQK * NUM_BLOCKS_DQK
    BLOCK_DV: tl.constexpr,  # DHV = BLOCK_DV * NUM_BLOCKS_DV
    EPS: tl.constexpr = 1e-6,
):
    pass


BLOCK_DQK = 16
BLOCK_DV = 16


@contiguous
def recurrent_step_fw(
    matC_old: torch.Tensor,  # (B, NH, DHQK, DHV)
    vecN_old: torch.Tensor,  # (B, NH, DHQK)
    scaM_old: torch.Tensor,  # (B, NH)
    vecQ: torch.Tensor,  # (B, NH, DHQK)
    vecK: torch.Tensor,  # (B, NH, DHQK)
    vecV: torch.Tensor,  # (B, NH, DHV)
    scaI: torch.Tensor,  # (B, NH, 1)
    scaF: torch.Tensor,  # (B, NH, 1)
    matC_new: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecN_new: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_new: torch.Tensor = None,  # (B, NH, 1)
    DTYPE_GATE: torch.dtype = torch.float32,
    DTYPE_STATE: torch.dtype = torch.float32,
    DTPYE_QKV: torch.dtype = torch.float32,
    EPS: float = 1e-6,
    BLOCK_DQK: int = 16,
    BLOCK_DV: int = 16,
):
    B, NH, DHQK, DHV = matC_old.shape

    if matC_new is None:
        assert (
            vecN_new is None and scaM_new is None
        ), "Initial states must be provided together."
        matC_new = torch.zeros(
            (B, NH, DHQK, DHV), dtype=DTYPE_STATE, device=matC_old.device
        )
        vecN_new = torch.zeros((B, NH, DHQK), dtype=DTYPE_STATE, device=matC_old.device)
        scaM_new = torch.zeros((B, NH, 1), dtype=DTYPE_STATE, device=matC_old.device)

    def grid_fn(*args):
        NUM_BLOCKS_DQK = triton.cdiv(DHQK, BLOCK_DQK)
        NUM_BLOCKS_DV = triton.cdiv(DHV, BLOCK_DV)
        NUM_BATCH_HEAD = B * NH
        return (NUM_BLOCKS_DQK, NUM_BLOCKS_DV, NUM_BATCH_HEAD)

    grid = grid_fn

    # create output tensors
    vecH = torch.zeros_like(vecV)

    _recurrent_step_fw_kernel[grid](
        matC_old,
        vecN_old,
        scaM_old,
        vecQ,
        vecK,
        vecV,
        scaI,
        scaF,
        vecH,
        matC_new,
        vecN_new,
        scaM_new,
        matC_old.stride(0),
        matC_old.stride(1),
        matC_old.stride(2),
        vecN_old.stride(0),
        vecN_old.stride(1),
        scaM_old.stride(0),
        scaM_old.stride(1),
        vecQ.stride(0),
        vecQ.stride(1),
        vecV.stride(0),
        vecV.stride(1),
        B,
        NH,
        DHQK,
        DHV,
        BLOCK_DQK,
        BLOCK_DV,
        EPS,
    )

    return vecH, (matC_new, vecN_new, scaM_new)
