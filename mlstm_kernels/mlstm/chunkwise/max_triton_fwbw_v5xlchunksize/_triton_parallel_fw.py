# Author: Maximilian Beck
"""This file contains the parallel part of the forward pass of the mLSTM chunkwise formulation.
It is adapted such that it allows for arbitrary large chunk sizes AND head dimensions.
"""

import torch
import triton
import triton.language as tl

from ....kernel_utils import is_power_of_2, torch2triton_dtype
from ._torch_chunkwise_gates import compute_chunkwise_log_gates_vecB_vecA


@triton.jit
def _mlstm_chunkwise__parallel_fw_Hintra_kernel(
    matQ,  # (B, NH, S, DHQK)
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    # these are all the states at every chunk, (we only use NC states up to the last chunk, i.e. :-1)
    matC_states,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecN_states,  # (B, NH, (NC+1) * DHQK)
    scaMinter_states,  # (B, NH, (NC+1))
    vecI,  # (B, NH, NC, L)
    vecB,  # (B, NH, NC, L)
    matHout,  # (B, NH, S, DHHV)
    vecNout,  # (B, NH, S)
    vecMout,  # (B, NH, S)
    qk_scale: tl.constexpr,
    str_matQK_B_NH: tl.constexpr,
    str_matQK_S: tl.constexpr,
    str_matQK_DHQK: tl.constexpr,
    str_matHV_B_NH: tl.constexpr,
    str_matHV_S: tl.constexpr,
    str_matHV_DHHV: tl.constexpr,
    str_matCstates_B_NH: tl.constexpr,
    str_matCstates_NCDHQK: tl.constexpr,
    str_matCstates_DHHV: tl.constexpr,
    str_vecNstates_B_NH: tl.constexpr,
    str_vecNstates_NCDHQK: tl.constexpr,
    str_scaMinterstates_B_NH: tl.constexpr,
    str_vecBI_B_NH: tl.constexpr,
    str_vecBI_NC: tl.constexpr,
    str_vecBI_L: tl.constexpr,
    str_vecMN_B_NH: tl.constexpr,
    str_vecMN_S: tl.constexpr,
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    siz_b_LQ: tl.constexpr,
    siz_b_LKV: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
    OUTPUT_DTYPE: tl.constexpr = tl.float32,
    EPS: tl.constexpr = 0.0,
    MINIMUM_MAX_VAL: tl.constexpr = -10.0,
):
    # our grid has 4 dimensions: (num_b_DHHV, num_b_LQ, (NC, B * NH))
    idx_b_DHHV, idx_b_LQ, idx_b_NC_BNH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    idx_b_NC = idx_b_NC_BNH % NC
    idx_b_BNH = idx_b_NC_BNH // NC

    # inititalize  vecM states
    vecM_old_val = tl.zeros([siz_b_LQ], dtype=tl.float32) - float("inf")
    vecM_new_val = tl.zeros([siz_b_LQ], dtype=tl.float32) - float("inf")

    # gate pointers for the current thread block
    vecB_ptr = vecB + idx_b_BNH * str_vecBI_B_NH + idx_b_NC * str_vecBI_NC
    vecI_ptr = vecI + idx_b_BNH * str_vecBI_B_NH + idx_b_NC * str_vecBI_NC

    # load vecB_LQ (siz_b_LQ,)
    vecB_LQ_ptr = vecB_ptr + idx_b_LQ * siz_b_LQ + tl.arange(0, siz_b_LQ)
    vecB_LQ_val = tl.load(vecB_LQ_ptr).to(tl.float32)

    # for causal masking
    b_q_offset = idx_b_LQ * siz_b_LQ
    b_q_idxes = b_q_offset + tl.arange(0, siz_b_LQ)

    # ? compute the intra chunk contribution
    # loop over b_LKV blocks
    # initialize accumulators
    # matH accumulator (siz_b_LQ, siz_b_DHHV_threadblock)
    matH_intra_acc = tl.zeros([siz_b_LQ, siz_b_DHHV], dtype=tl.float32)
    # vecN accumulator (siz_b_LQ,)
    vecN_intra_acc = tl.zeros([siz_b_LQ], dtype=tl.float32)
    # only compute the lower triangular part
    idx_b_LKV_end = ((idx_b_LQ + 1) * siz_b_LQ) // siz_b_LKV
    for idx_b_LKV in range(idx_b_LKV_end):
        # compute matG = matQ @ matK^T
        # matG accumulator (siz_b_LQ, siz_b_LKV)
        matG = tl.zeros([siz_b_LQ, siz_b_LKV], dtype=tl.float32)
        ## loop over DHQK blocks
        for idx_b_DHQK in range(tl.cdiv(DHQK, siz_b_DHQK)):
            # load matQ block (siz_b_LQ, siz_b_DHQK)
            matQ_ptr = tl.make_block_ptr(
                base=matQ + idx_b_BNH * str_matQK_B_NH,
                shape=(S, DHQK),
                strides=(str_matQK_S, str_matQK_DHQK),
                offsets=(idx_b_NC * L + idx_b_LQ * siz_b_LQ, idx_b_DHQK * siz_b_DHQK),
                block_shape=(siz_b_LQ, siz_b_DHQK),
                order=(1, 0),
            )
            matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1)).to(DTYPE)
            # load matK transposed block (siz_b_DHQK, siz_b_LKV)
            matK_ptr = tl.make_block_ptr(
                base=matK + idx_b_BNH * str_matQK_B_NH,
                shape=(DHQK, S),
                strides=(str_matQK_DHQK, str_matQK_S),
                offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_NC * L + idx_b_LKV * siz_b_LKV),
                block_shape=(siz_b_DHQK, siz_b_LKV),
                order=(0, 1),
            )
            matK_val = tl.load(matK_ptr, boundary_check=(0, 1)).to(DTYPE)

            # accumulate in matG (siz_b_LQ, siz_b_LKV)
            matG += tl.dot(matQ_val, matK_val)

        # load vecB_LKV (siz_B_LKV,)
        vecB_LKV_ptr = vecB_ptr + idx_b_LKV * siz_b_LKV + tl.arange(0, siz_b_LKV)
        vecB_LKV = tl.load(vecB_LKV_ptr).to(tl.float32)

        # load vecI_LKV (siz_B_LKV,)
        vecI_LKV_ptr = vecI_ptr + idx_b_LKV * siz_b_LKV + tl.arange(0, siz_b_LKV)
        vecI_LKV = tl.load(vecI_LKV_ptr).to(tl.float32)

        # construct gate matrix matDtilde (siz_b_LQ, siz_b_LKV)
        matDtilde_val = vecB_LQ_val[:, None] - vecB_LKV[None, :] + vecI_LKV[None, :]

        b_kv_offset = idx_b_LKV * siz_b_LKV
        # causal masking if on the diagonal
        if b_kv_offset >= b_q_offset:
            b_kv_idxes = b_kv_offset + tl.arange(0, siz_b_LKV)
            mask = b_q_idxes[:, None] >= b_kv_idxes[None, :]
            matDtilde_val = tl.where(mask, matDtilde_val, -float("inf"))

        # compute vecM_new (siz_b_LQ,)
        vecM_new_val = tl.max(
            matDtilde_val, axis=1
        )  # (siz_b_LQ,) # row-wise max along siz_b_LKV
        vecM_new_val = tl.maximum(
            vecM_new_val, MINIMUM_MAX_VAL
        )  # (siz_b_LQ,) # element-wise max

        vecM_new_val = tl.maximum(vecM_old_val, vecM_new_val)
        vecM_ratio = tl.exp(vecM_old_val - vecM_new_val)

        # compute matD (siz_b_LQ, siz_b_LKV)
        matD_val = tl.exp(matDtilde_val - vecM_new_val[:, None])
        # tl.device_print("matD_val", matD_val)
        # compute matS (siz_b_LQ, siz_b_LKV)
        matS = matG * qk_scale * matD_val

        # compute vecN (siz_b_LQ,)
        vecN_intra_acc = vecM_ratio * vecN_intra_acc + tl.sum(matS, axis=1)

        # load matV (siz_b_LKV, siz_b_DHHV)
        matV_ptr = tl.make_block_ptr(
            base=matV + idx_b_BNH * str_matHV_B_NH,
            shape=(S, DHHV),
            strides=(str_matHV_S, str_matHV_DHHV),
            offsets=(idx_b_NC * L + idx_b_LKV * siz_b_LKV, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_LKV, siz_b_DHHV),
            order=(1, 0),
        )
        matV_val = tl.load(matV_ptr, boundary_check=(0, 1)).to(DTYPE)

        # accumulate matH (siz_b_LQ, siz_b_DHHV)
        matH_cur = tl.dot(matS.to(DTYPE), matV_val)
        matH_intra_acc = vecM_ratio[:, None] * matH_intra_acc + matH_cur

        # update max state for next iteration
        vecM_old_val = vecM_new_val

    ##? compute the inter chunk contribution
    # compute vecM_combine (siz_b_LQ,)
    # load scaM_inter (1,)
    scaM_inter_km1_ptr = (
        scaMinter_states + idx_b_BNH * str_scaMinterstates_B_NH + idx_b_NC
    )
    scaM_inter_km1_val = tl.load(scaM_inter_km1_ptr).to(tl.float32)
    # vecM_intra = vecM_new_val
    vecM_combine_val = tl.maximum(vecB_LQ_val + scaM_inter_km1_val, vecM_new_val)

    vecBbar_val = tl.exp(vecB_LQ_val + scaM_inter_km1_val - vecM_combine_val)

    ## loop over DHQK blocks
    # Note: this loop is the same as the inner one above!
    # we cannot merge this loop into the one above as we need the vecM_combine_val,
    # which is computed in the loop above and is necessary for vecBbar_val computation
    # The cost is that we load matQ twice, but this is necessary for the correct computation of vecBbar_val
    matH_inter_acc = tl.zeros([siz_b_LQ, siz_b_DHHV], dtype=tl.float32)
    vecN_inter_acc = tl.zeros([siz_b_LQ], dtype=tl.float32)
    for idx_b_DHQK in range(tl.cdiv(DHQK, siz_b_DHQK)):
        matQ_ptr = tl.make_block_ptr(
            base=matQ + idx_b_BNH * str_matQK_B_NH,
            shape=(S, DHQK),
            strides=(str_matQK_S, str_matQK_DHQK),
            offsets=(idx_b_NC * L + idx_b_LQ * siz_b_LQ, idx_b_DHQK * siz_b_DHQK),
            block_shape=(siz_b_LQ, siz_b_DHQK),
            order=(1, 0),
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

        # load matQ block (siz_b_LQ, siz_b_DHQK)
        matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1)).to(tl.float32)
        matQbar_val = (matQ_val * vecBbar_val[:, None] * qk_scale).to(DTYPE)

        # load matC_km1 (siz_b_DHQK, siz_b_DHHV)
        matC_km1_val = tl.load(matC_km1_ptr, boundary_check=(0, 1)).to(DTYPE)

        # acccumulate matH_inter (siz_b_LQ, siz_b_DHHV)
        matH_inter_acc += tl.dot(matQbar_val, matC_km1_val)

        # load vecN_km1 (siz_b_DHQK,)
        vecN_km1_val = tl.load(vecN_km1_ptr).to(tl.float32)

        # accumulate vecN_inter (siz_b_LQ,1) = matQbar (siz_b_LQ, siz_b_DHQK) @ vecN_km1 (siz_b_DHQK,1)
        vecN_inter_acc += tl.sum(matQbar_val * vecN_km1_val[None, :], axis=1)

    # ? combine the intra and inter chunk contributions

    # compute the vecM_comb_ratio (siz_b_LQ,)
    vecM_comb_ratio = tl.exp(vecM_new_val - vecM_combine_val)

    # compute the matH_comb_num (siz_b_LQ, siz_b_DHHV)
    matH_comb_num_val = matH_inter_acc + vecM_comb_ratio[:, None] * matH_intra_acc

    # compute the vecN_comb_denom (siz_b_LQ,)
    vecN_comb_denom_val = tl.maximum(
        tl.abs(vecN_inter_acc + vecM_comb_ratio * vecN_intra_acc),
        tl.exp(-vecM_combine_val),
    )

    # compute the matH_comb_out_val (siz_b_LQ, siz_b_DHHV)
    matH_comb_out_val = matH_comb_num_val / (vecN_comb_denom_val[:, None] + EPS)

    # store matHout (size_b_LQ, siz_b_DHHV)
    matHout_ptr = tl.make_block_ptr(
        base=matHout + idx_b_BNH * str_matHV_B_NH,
        shape=(S, DHHV),
        strides=(str_matHV_S, str_matHV_DHHV),
        offsets=(idx_b_NC * L + idx_b_LQ * siz_b_LQ, idx_b_DHHV * siz_b_DHHV),
        block_shape=(siz_b_LQ, siz_b_DHHV),
        order=(1, 0),
    )
    tl.store(matHout_ptr, matH_comb_out_val.to(DTYPE), boundary_check=(0, 1))

    # the different thread blocks for different value head dimensions
    # compute the same vecN and vecM
    if idx_b_DHHV == 0:
        # store vecNout (siz_b_LQ,)
        vecNout_ptr = (
            vecNout
            + idx_b_BNH * str_vecMN_B_NH
            + idx_b_NC * L
            + idx_b_LQ * siz_b_LQ
            + tl.arange(0, siz_b_LQ)
        )
        tl.store(vecNout_ptr, vecN_comb_denom_val.to(OUTPUT_DTYPE))
        # store vecMout (size_b_LQ,)
        vecMout_ptr = (
            vecMout
            + idx_b_BNH * str_vecMN_B_NH
            + idx_b_NC * L
            + idx_b_LQ * siz_b_LQ
            + tl.arange(0, siz_b_LQ)
        )
        tl.store(vecMout_ptr, vecM_combine_val.to(OUTPUT_DTYPE))


def mlstm_chunkwise__parallel_fw_Hintra(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecI: torch.Tensor,  # (B, NH, NC * L) = (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, NC * L) = (B, NH, S)
    # these are all the states at every chunk, (we only use NC states up to the last chunk, i.e. :-1)
    matC_states: torch.Tensor,  # (B, NH, (NC+1) * DHQK, DHHV)
    vecN_states: torch.Tensor,  # (B, NH, (NC+1) * DHQK)
    scaMinter_states: torch.Tensor,  # (B, NH, (NC+1))
    qk_scale: float = None,
    chunk_size: int = 64,
    siz_b_LQ: int = 32,
    siz_b_LKV: int = 32,
    siz_b_DHQK: int | None = None,
    siz_b_DHHV: int | None = None,  # DHHV blocksize for each thread block
    num_warps: int | None = None,
    num_stages: int | None = None,
    eps: float = 1e-6,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:  # matH_out (B, NH, S, DHHV), vecN_out (B, NH, S), vecM_out (B, NH, S)
    """This function defines the grid and block sizes for the kernel launch and calls the kernel.
    chunk parallel size:        siz_b_LQ
    chunk loop size:            siz_b_LKV
    head dim parallel size:     siz_b_DHHV
    head dim loop size:         siz_b_DHQK
    """
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[-1]

    assert (
        S % chunk_size == 0
    ), f"Sequence length {S} must be divisible by chunk size {chunk_size}"
    NC = S // chunk_size
    L = chunk_size

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    siz_b_DHQK = (
        min(64, triton.next_power_of_2(DHQK)) if siz_b_DHQK is None else siz_b_DHQK
    )

    if siz_b_DHHV is None:
        siz_b_DHHV = min(128, triton.next_power_of_2(DHHV))
    else:
        siz_b_DHHV = siz_b_DHHV

    assert siz_b_LQ <= L, "siz_b_LQ must be less than or equal to chunk size L"
    assert siz_b_LKV <= L, "siz_b_LKV must be less than or equal to chunk size L"
    assert siz_b_LKV <= siz_b_LQ, "siz_b_LKV must be less than or equal to siz_b_LQ"
    assert siz_b_LQ % siz_b_LKV == 0, "siz_b_LQ must be divisible by siz_b_LKV"
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)
    num_b_LQ = triton.cdiv(L, siz_b_LQ)

    num_stages = 1 if num_stages is None else num_stages
    if num_warps is None:
        num_warps = 4 if siz_b_DHQK >= 64 else 2

    matH_out = torch.empty(B, NH, S, DHHV, device=matQ.device, dtype=matQ.dtype)
    vecN_out = torch.empty(B, NH, S, device=matQ.device, dtype=output_dtype)
    vecM_out = torch.empty(B, NH, S, device=matQ.device, dtype=output_dtype)

    vecB = compute_chunkwise_log_gates_vecB_vecA(
        vecI=vecI, vecF=vecF, chunk_size=chunk_size, return_vecB_only=True
    )

    grid = (num_b_DHHV, num_b_LQ, NC * B * NH)
    # print("grid(num_b_DHHV, num_b_LQ, NC*B*NH)", grid)
    _mlstm_chunkwise__parallel_fw_Hintra_kernel[grid](
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
        siz_b_LQ=siz_b_LQ,
        siz_b_LKV=siz_b_LKV,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        DTYPE=torch2triton_dtype(matQ.dtype),
        OUTPUT_DTYPE=torch2triton_dtype(output_dtype),
        MINIMUM_MAX_VAL=-10.0,
        EPS=eps,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return matH_out, vecN_out, vecM_out
