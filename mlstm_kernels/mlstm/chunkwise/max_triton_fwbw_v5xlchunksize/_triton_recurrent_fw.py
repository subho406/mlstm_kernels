# Copyright JKU Linz 2024
# Author: Maximilian Beck
"""This file contains the recurrent part of forward pass of the mLSTM chunkwise formulation.
It is adapted such that it allows to write out the C states at every NC-th chunk.
"""

import torch
import triton
import triton.language as tl

from ....kernel_utils import is_power_of_2, torch2triton_dtype


# TODO use the strides in the pointers for generic use case (even if they are 1 normally)
# Note: we only pass stride for the head dimension (we do not access individual batch elements directly)
@triton.jit
def _mlstm_chunkwise__recurrent_fw_C_kernel(
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    vecF,  # (B, NH, NC * L) = (B, NH, S)
    vecI,  # (B, NH, NC * L) = (B, NH, S)
    matC_initial,  # (B, NH, DHQK, DHHV)
    vecN_initial,  # (B, NH, DHQK)
    scaMinter_initial,  # (B, NH)
    matC_states,  # (B, NH, (NC + 1) * DHQK, DHHV)
    vecN_states,  # (B, NH, (NC + 1) * DHQK)
    scaMinter_states,  # (B, NH, (NC + 1))
    str_matK_B_NH: tl.constexpr,
    str_matK_S: tl.constexpr,
    str_matK_DHQK: tl.constexpr,
    str_matV_B_NH: tl.constexpr,
    str_matV_S: tl.constexpr,
    str_matV_DHHV: tl.constexpr,
    str_vecFI_B_NH: tl.constexpr,
    str_matCstates_B_NH: tl.constexpr,
    str_matCstates_NCDHQK: tl.constexpr,
    str_matCstates_DHHV: tl.constexpr,
    str_vecNstates_B_NH: tl.constexpr,
    str_vecNstates_NCDHQK: tl.constexpr,
    str_scaMinterstates_B_NH: tl.constexpr,
    str_scaMinterstates_NC: tl.constexpr,
    str_matCinitial_B_NH: tl.constexpr,
    str_matCinitial_DHQK: tl.constexpr,
    str_matCinitial_DHHV: tl.constexpr,
    str_vecNinitial_B_NH: tl.constexpr,
    str_vecNinitial_DHQK: tl.constexpr,
    str_scaMinterinitial_B_NH: tl.constexpr,
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    save_states_every_nth_chunk: tl.constexpr,
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
        #! store only if k is a multiple of save_states_every_nth_chunk
        if k % save_states_every_nth_chunk == 0:
            idx_k_save = k // save_states_every_nth_chunk
            # each thread block stores a (siz_b_DHQK, siz_b_DHHV) block to matC_states_k
            # create pointer for matCstates_k, vecNstates_k, scaMinterstates_k
            matCstates_k_ptr = tl.make_block_ptr(
                base=matC_states + idx_b_BNH * str_matCstates_B_NH + idx_k_save * DHQK * DHHV,
                shape=(DHQK, DHHV),
                strides=(str_matCstates_NCDHQK, str_matCstates_DHHV),
                offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
                block_shape=(siz_b_DHQK, siz_b_DHHV),
                order=(1, 0),
            )
            vecNstates_k_ptr = (
                vecN_states
                + idx_b_BNH * str_vecNstates_B_NH
                + idx_k_save * DHQK
                + idx_b_DHQK * siz_b_DHQK
                + tl.arange(0, siz_b_DHQK)
            )
            scaMinterstates_k_ptr = scaMinter_states + idx_b_BNH * str_scaMinterstates_B_NH + idx_k_save

            # store the states from the previous iteration
            tl.store(matCstates_k_ptr, matC_k_val.to(dtype=tl.float32), boundary_check=(0, 1))
            if idx_b_DHHV == 0:
                tl.store(vecNstates_k_ptr, vecN_k_val.to(dtype=tl.float32))
            if (idx_b_DHQK == 0) and (idx_b_DHHV == 0):
                tl.store(scaMinterstates_k_ptr, scaMinter_k_val.to(dtype=tl.float32))

        # load / compute vecA_k, scaG_k
        # last element of vecB in k-th chunk
        idx_L = tl.arange(0, L)
        vecF_k_val = tl.load(
            vecF + idx_b_BNH * str_vecFI_B_NH + k * L + idx_L + 1,
            mask=(idx_L < L - 1),
            other=0.0,
        ).to(tl.float32)

        vecFlogsig_k_val = tl.log(tl.sigmoid(vecF_k_val))
        vecFlogsig_masked = tl.where(idx_L < L - 1, vecFlogsig_k_val, 0.0).to(tl.float32)

        vecI_k_val = tl.load(vecI + idx_b_BNH * str_vecFI_B_NH + k * L + idx_L).to(tl.float32)

        vecA_k_val = tl.flip(tl.cumsum(tl.flip(vecFlogsig_masked), axis=0)) + vecI_k_val

        vecFfirst_k_val = tl.load(vecF + idx_b_BNH * str_vecFI_B_NH + k * L + 0).to(tl.float32)
        vecFfirstlogsig_k_val = tl.log(tl.sigmoid(vecFfirst_k_val))
        scaG_k_val = tl.sum(vecFlogsig_masked, axis=0) + vecFfirstlogsig_k_val

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

    k = NC
    # store the states from the last iteration
    if k % save_states_every_nth_chunk == 0:
        idx_k_save = k // save_states_every_nth_chunk
        matCstates_k_ptr = tl.make_block_ptr(
            base=matC_states + idx_b_BNH * str_matCstates_B_NH + idx_k_save * DHQK * DHHV,
            shape=(DHQK, DHHV),
            strides=(str_matCstates_NCDHQK, str_matCstates_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        vecNstates_k_ptr = (
            vecN_states
            + idx_b_BNH * str_vecNstates_B_NH
            + idx_k_save * DHQK
            + idx_b_DHQK * siz_b_DHQK
            + tl.arange(0, siz_b_DHQK)
        )
        scaMinterstates_k_ptr = scaMinter_states + idx_b_BNH * str_scaMinterstates_B_NH + idx_k_save
        tl.store(matCstates_k_ptr, matC_k_val.to(dtype=tl.float32), boundary_check=(0, 1))
        if idx_b_DHHV == 0:
            tl.store(vecNstates_k_ptr, vecN_k_val.to(dtype=tl.float32))
        if (idx_b_DHQK == 0) and (idx_b_DHHV == 0):
            tl.store(scaMinterstates_k_ptr, scaMinter_k_val.to(dtype=tl.float32))


def mlstm_chunkwise__recurrent_fw_C(
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecF: torch.Tensor,  # (B, NH, NC * L) = (B, NH, S)
    vecI: torch.Tensor,  # (B, NH, NC * L) = (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaMinter_initial: torch.Tensor = None,  # (B, NH)
    chunk_size: int = 64,
    num_stages: int | None = None,
    num_warps: int | None = None,
    save_states_every_nth_chunk: int = 1,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:  # matC_states (B, NH, (NC+1) * DHQK, DHHV), vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1))
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[-1]

    L = chunk_size
    assert S % L == 0, "Sequence length must be divisible by chunk size."
    NC = S // L

    assert save_states_every_nth_chunk > 0, "save_states_every_nth_chunk must be positive."
    assert save_states_every_nth_chunk <= NC, "save_states_every_nth_chunk must be <= NC."

    assert is_power_of_2(
        save_states_every_nth_chunk
    ), f"save_states_every_nth_chunk must be a power of 2. Got {save_states_every_nth_chunk}."

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1 if num_stages is None else num_stages
    if num_warps is None:
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

    num_chunks_saved = NC // save_states_every_nth_chunk

    matC_states = torch.empty(
        B,
        NH,
        (num_chunks_saved + 1) * DHQK,
        DHHV,
        device=matK.device,
        dtype=torch.float32,
    )
    vecN_states = torch.empty(
        B,
        NH,
        (num_chunks_saved + 1) * DHQK,
        device=matK.device,
        dtype=torch.float32,
    )
    scaMinter_states = torch.empty(B, NH, (num_chunks_saved + 1), device=matK.device, dtype=torch.float32)

    grid = (num_b_DHQK, num_b_DHHV, B * NH)
    _mlstm_chunkwise__recurrent_fw_C_kernel[grid](
        matK=matK,
        matV=matV,
        vecF=vecF,
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
        str_vecFI_B_NH=vecF.stride(1),
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
        save_states_every_nth_chunk=save_states_every_nth_chunk,
        USE_INITIAL_STATE=USE_INITIAL_STATE,
        DTYPE=torch2triton_dtype(matK.dtype),
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return matC_states, vecN_states, scaMinter_states
