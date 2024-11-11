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

from ...kernel_utils import contiguous_noctx, torch2triton_dtype
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
) -> (
    torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
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


ENABLE_AUTOTUNING = True

# TODO find better heuristic
# add more block sizes if the DHQK and DHV are not equal.

if ENABLE_AUTOTUNING:
    configs = [
        triton.Config({"BLOCK_DQK": BQ, "BLOCK_DV": BV}, num_stages=s, num_warps=w)
        for BQ, BV, w in [
            (256, 256, 8),
            (256, 256, 16),
            (128, 128, 4),
            (128, 128, 8),
            (128, 128, 16),
            (64, 64, 2),
            (64, 64, 4),
            (64, 64, 8),
            (32, 32, 1),
            (32, 32, 2),
            (32, 32, 4),
            (16, 16, 1),
        ]
        for s in [1]
    ]
else:
    configs = [
        triton.Config({"BLOCK_DQK": BQ, "BLOCK_DV": BV}, num_stages=s, num_warps=w)
        for BQ, BV, w in [
            (32, 32, 4),
        ]
        for s in [1]
    ]


def keep(conf):
    BQ = conf.kwargs["BLOCK_DQK"]
    BV = conf.kwargs["BLOCK_DV"]
    if BQ * BV < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["DHQK", "DHV"])
@triton.jit
def _recurrent_step_fw_kernel(
    matC_old,  # (B, NH, DHQK, DHV)
    vecN_old,  # (B, NH, DHQK)
    scaM_old,  # (B, NH, 1)
    vecQ,  # (B, NH, DHQK)
    vecK,  # (B, NH, DHQK)
    vecV,  # (B, NH, DHV)
    vecH,  # (B, NH, DHV) # TODO organize outputs to the end of argument list
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
    DTYPE: tl.constexpr = tl.float16,
    # num_warps: tl.constexpr = 4,
):
    i_dhv, i_bnh = tl.program_id(1), tl.program_id(2)

    # ? Define pointers
    matC_old_bptr = tl.make_block_ptr(
        base=matC_old + i_bnh * s_matC_nh,
        shape=(DHQK, DHV),
        strides=(s_matC_dhqk, s_matC_dhv),
        offsets=(0, i_dhv * BLOCK_DV),
        block_shape=(BLOCK_DQK, BLOCK_DV),
        order=(0, 1),
    )
    matC_new_bptr = tl.make_block_ptr(
        base=matC_new + i_bnh * s_matC_nh,
        shape=(DHQK, DHV),
        strides=(s_matC_dhqk, s_matC_dhv),
        offsets=(0, i_dhv * BLOCK_DV),
        block_shape=(BLOCK_DQK, BLOCK_DV),
        order=(0, 1),
    )
    vecH_ptr = (
        vecH
        + i_bnh * s_vecVH_nh
        + i_dhv * BLOCK_DV * s_vecVH_dhv
        + tl.arange(0, BLOCK_DV)
    )

    scaI_ptr = scaI + i_bnh * s_scaIF_nh
    scaF_ptr = scaF + i_bnh * s_scaIF_nh

    scaM_old_ptr = scaM_old + i_bnh * s_scaM_nh
    scaM_new_ptr = scaM_new + i_bnh * s_scaM_nh

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
    scaF_act = tl.exp2(
        (scaFlog_val + scaM_old_val - scaM_new_val) * 1.4426950408889634
    ).to(DTYPE)
    scaI_act = tl.exp2((scaI_val - scaM_new_val) * 1.4426950408889634).to(DTYPE)
    # tl.static_print("scaF_act", scaF_act)
    # ? init accumulators
    h_num = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    qn_dotproduct = tl.zeros((1,), dtype=tl.float32)

    NUM_BLOCKS_DQK = triton.cdiv(DHQK, BLOCK_DQK)

    for i_dhqk in range(NUM_BLOCKS_DQK):
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

        vecQ_ptr = (
            vecQ
            + i_bnh * s_vecQK_nh
            + i_dhqk * BLOCK_DQK * s_vecQK_dhqk
            + tl.arange(0, BLOCK_DQK)
        )
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

        # update rule
        # TODO add masking to avoid out of bound access
        vecK_val = tl.load(vecK_ptr)
        vecV_val = tl.load(vecV_ptr)
        # tl.static_print("vecK_val_scaled", vecK_val_scaled)
        # tl.static_print("vecV_val", vecV_val)
        matC_old_val = tl.load(
            matC_old_bptr, boundary_check=(0, 1), padding_option="zero"
        )

        matC_new_val = scaF_act * matC_old_val + scaI_act * (
            vecK_val[:, None] * vecV_val[None, :]
        )

        vecN_new_val = scaF_act * tl.load(vecN_old_ptr) + scaI_act * vecK_val
        # tl.static_print("vecN_new_val", vecN_new_val)
        # tl.static_print("matC_new_val", matC_new_val)
        # tl.static_print("matC_old_val", matC_old_val)
        # ? Store data
        tl.store(
            matC_new_bptr,
            matC_new_val.to(matC_new.type.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            vecN_new_ptr, vecN_new_val.to(vecN_new.type.element_ty)
        )  # TODO add masking to avoid out of bound access

        # ? advance pointers
        matC_old_bptr = tl.advance(matC_old_bptr, (BLOCK_DQK, 0))
        matC_new_bptr = tl.advance(matC_new_bptr, (BLOCK_DQK, 0))

        # ? accumulate h_num & qn_dotproduct
        vecQ_val = tl.load(vecQ_ptr) * qk_scale
        # outputs
        h_num_temp = vecQ_val[:, None] * matC_new_val
        # tl.static_print("h_num_temp", h_num_temp)
        # we keep h_num and qn_dotproduct in float32 as they are accumulated
        h_num += tl.sum(h_num_temp, axis=0)
        qn_dotproduct += tl.sum(vecQ_val * vecN_new_val)
        # tl.static_print(
        #     "tl.sum(vecQ_val * vecN_new_val)", tl.sum(vecQ_val * vecN_new_val)
        # )

    # we compute h in float32 and then cast to DTYPE
    h_denom = tl.maximum(tl.abs(qn_dotproduct), max_val) + EPS
    # tl.static_print("h_denom", h_denom)
    # tl.static_print("h_num", h_num)
    h = tl.fdiv(h_num, h_denom)
    # tl.static_print("h", h)
    # tl.static_print("vecH_ptr", vecH_ptr)

    # ? Store data
    tl.store(vecH_ptr, h.to(DTYPE))


@contiguous_noctx
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
    EPS: float = 1e-6,
    # BLOCK_DQK: int = 16,
    # BLOCK_DV: int = 16,
):
    B, NH, DHQK, DHV = matC_old.shape

    DTYPE = matC_old.dtype

    if qk_scale is None:
        qk_scale = 1 / math.sqrt(DHQK)

    if matC_new is None:
        assert (
            vecN_new is None and scaM_new is None
        ), "Initial states must be provided together."
        matC_new = torch.empty_like(matC_old)
        vecN_new = torch.empty_like(vecN_old)
        scaM_new = torch.empty_like(scaM_old)

    def grid_fn_C(args):
        # NUM_BLOCKS_DQK = triton.cdiv(DHQK, args["BLOCK_DQK"])
        NUM_BLOCKS_DV = triton.cdiv(DHV, args["BLOCK_DV"])
        NUM_BATCH_HEAD = B * NH
        grid = (1, NUM_BLOCKS_DV, NUM_BATCH_HEAD)
        return grid

    # DEBUG ONLY
    # def grid_fn_C(*args):
    #     NUM_BLOCKS_DQK = triton.cdiv(DHQK, BLOCK_DQK)
    #     NUM_BLOCKS_DV = triton.cdiv(DHV, BLOCK_DV)
    #     NUM_BATCH_HEAD = B * NH
    #     grid = (NUM_BLOCKS_DQK, NUM_BLOCKS_DV, NUM_BATCH_HEAD)
    #     print(grid)
    #     return grid

    grid_C = grid_fn_C

    # create output tensors
    vecH = torch.empty_like(vecV)

    _recurrent_step_fw_kernel[grid_C](
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
        s_matC_b=matC_old.stride(0),
        s_matC_nh=matC_old.stride(1),
        s_matC_dhqk=matC_old.stride(2),
        s_matC_dhv=matC_old.stride(3),
        s_vecN_b=vecN_old.stride(0),
        s_vecN_nh=vecN_old.stride(1),
        s_vecN_dhqk=vecN_old.stride(2),
        s_scaM_b=scaM_old.stride(0),
        s_scaM_nh=scaM_old.stride(1),
        s_vecQK_b=vecQ.stride(0),
        s_vecQK_nh=vecQ.stride(1),
        s_vecQK_dhqk=vecQ.stride(2),
        s_vecVH_b=vecV.stride(0),
        s_vecVH_nh=vecV.stride(1),
        s_vecVH_dhv=vecV.stride(2),
        s_scaIF_b=scaI.stride(0),
        s_scaIF_nh=scaI.stride(1),
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHV=DHV,
        # BLOCK_DQK=BLOCK_DQK,
        # BLOCK_DV=BLOCK_DV,
        EPS=EPS,
        DTYPE=torch2triton_dtype(DTYPE),
    )

    return vecH, (matC_new, vecN_new, scaM_new)
