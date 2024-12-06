# Author: Maximilian Beck
"""This file contains the recurrent part of the backward pass of the mLSTM chunkwise formulation.
It is adapted such that it allows to write out the deltaC states and optionally the deltaN states
at every NC-th chunk.
It can be configured, whether the deltaN states are computed or not.
"""

import torch
import triton
import triton.language as tl

from ....torch.utils import is_power_of_2, torch2triton_dtype


@triton.jit
def _mlstm_chunkwise__recurrent_bw_dC_kernel(
    matQ,  # (B, NH, S, DHQK)
    vecF,  # (B, NH, NC * L) = (B, NH, S)
    scaM_inter,  # (B, NH, NC+1)
    vecM_combine,  # (B, NH, S)
    matDeltaH,  # (B, NH, S, DHHV)
    vecN_out,  # (B, NH, S)
    matDeltaC_last,  # (B, NH, DHQK, DHHV)
    matDeltaC_states,  # (B, NH, (NC+1) * DHQK, DHHV)
    qk_scale: tl.constexpr,
    str_matQ_B_NH: tl.constexpr,
    str_matQ_S: tl.constexpr,
    str_matQ_DHQK: tl.constexpr,
    str_vecF_B_NH: tl.constexpr,
    str_scaM_inter_B_NH: tl.constexpr,
    str_scaM_inter_NC: tl.constexpr,
    str_vecM_combine_B_NH: tl.constexpr,
    str_vecM_combine_S: tl.constexpr,
    str_matDeltaH_B_NH: tl.constexpr,
    str_matDeltaH_S: tl.constexpr,
    str_matDeltaH_DHHV: tl.constexpr,
    str_vecN_out_B_NH: tl.constexpr,
    str_vecN_out_S: tl.constexpr,
    str_matDeltaC_last_B_NH: tl.constexpr,
    str_matDeltaC_last_DHQK: tl.constexpr,
    str_matDeltaC_last_DHHV: tl.constexpr,
    str_matDeltaC_states_B_NH: tl.constexpr,
    str_matDeltaC_states_NCDHQK: tl.constexpr,
    str_matDeltaC_states_DHHV: tl.constexpr,
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
    USE_LAST_STATE: tl.constexpr,
    DTYPE: tl.constexpr = tl.float32,
    EPS: tl.constexpr = 1e-6,
):
    idx_b_DHQK, idx_b_DHHV, idx_b_NH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )

    # create running deltaC error state in shared memory
    matDeltaC_k_val = tl.zeros((siz_b_DHQK, siz_b_DHHV), dtype=tl.float32)

    if USE_LAST_STATE:
        # each thread block loads a (siz_b_DHQK, siz_b_DHHV) tile of matDeltaC_last
        matDeltaC_last_ptr = tl.make_block_ptr(
            base=matDeltaC_last + idx_b_NH * str_matDeltaC_last_B_NH,
            shape=(DHQK, DHHV),
            strides=(str_matDeltaC_last_DHQK, str_matDeltaC_last_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        # load last state
        matDeltaC_k_val = tl.load(matDeltaC_last_ptr, boundary_check=(1, 0)).to(tl.float32)

    # iterate over chunks from last to first
    for k in range(NC, 0, -1):
        # ? define pointers
        # load matQ in transposed form
        matQ_k_ptr = tl.make_block_ptr(
            base=matQ + idx_b_NH * str_matQ_B_NH,
            shape=(DHQK, S),
            strides=(str_matQ_DHQK, str_matQ_S),
            offsets=(idx_b_DHQK * siz_b_DHQK, (k - 1) * L),
            block_shape=(siz_b_DHQK, L),
            order=(0, 1),
        )
        matDeltaH_ptr = tl.make_block_ptr(
            base=matDeltaH + idx_b_NH * str_matDeltaH_B_NH,
            shape=(S, DHHV),
            strides=(str_matDeltaH_S, str_matDeltaH_DHHV),
            offsets=((k - 1) * L, idx_b_DHHV * siz_b_DHHV),
            block_shape=(L, siz_b_DHHV),
            order=(1, 0),
        )
        # ? end pointers
        if k % save_states_every_nth_chunk == 0:
            idx_k_save = k // save_states_every_nth_chunk
            # * store matDeltaC_k_val from previous iteration in HBM
            matDeltaCstates_k_ptr = tl.make_block_ptr(
                base=matDeltaC_states + idx_b_NH * str_matDeltaC_states_B_NH + idx_k_save * DHQK * DHHV,
                shape=(DHQK, DHHV),
                strides=(str_matDeltaC_states_NCDHQK, str_matDeltaC_states_DHHV),
                offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
                block_shape=(siz_b_DHQK, siz_b_DHHV),
                order=(1, 0),
            )
            tl.store(
                matDeltaCstates_k_ptr,
                matDeltaC_k_val.to(tl.float32),
                boundary_check=(0, 1),
            )

        # * compute matDeltaC_km1_val
        # load scaG_k, vecB_k, scaM_inter_km1, scaM_inter_k, vecM_combine_k
        # load vecF
        vecF_val = tl.load(
            vecF + idx_b_NH * str_vecF_B_NH + (k - 1) * L + tl.arange(0, L),
        ).to(tl.float32)
        vecFlogsig_val = tl.log(tl.sigmoid(vecF_val))

        vecB_val = tl.cumsum(vecFlogsig_val, axis=0)  # (L,)
        # scaG_k_val is the sum of all forget gates in the current chunk
        scaG_k_val = tl.sum(vecFlogsig_val, axis=0)  # (1,)

        scaM_inter_km1_val = tl.load(scaM_inter + idx_b_NH * str_scaM_inter_B_NH + (k - 1)).to(tl.float32)
        scaM_inter_k_val = tl.load(scaM_inter + idx_b_NH * str_scaM_inter_B_NH + k).to(tl.float32)
        vecM_combine_k_val = tl.load(
            vecM_combine + idx_b_NH * str_vecM_combine_B_NH + (k - 1) * L + tl.arange(0, L)
        ).to(tl.float32)

        # compute scaGbar_k, vecBbar_k
        scaGbar_k_val = tl.exp(scaG_k_val + scaM_inter_km1_val - scaM_inter_k_val)
        vecBbar_k_val = tl.exp(vecB_val + scaM_inter_km1_val - vecM_combine_k_val)

        # compute matQbar_k (DHQK, L) (Note: matQ_k is transposed)
        matQ_k_val = tl.load(matQ_k_ptr, boundary_check=(0, 1)).to(tl.float32)
        matQbar_k_val = (matQ_k_val * vecBbar_k_val[None, :] * qk_scale).to(DTYPE)

        # load vecN_out_k, matDeltaH_k
        vecN_out_k_val = tl.load(vecN_out + idx_b_NH * str_vecN_out_B_NH + (k - 1) * L + tl.arange(0, L)).to(
            tl.float32
        )  # (L,)
        matDeltaH_k_val = tl.load(matDeltaH_ptr, boundary_check=(0, 1)).to(tl.float32)  # (L, DHHV)
        # compute matDeltaHinter_k
        matDeltaH_k_val = (matDeltaH_k_val / (vecN_out_k_val[:, None] + EPS)).to(DTYPE)

        # compute matDeltaC_km1
        matDeltaC_k_val = scaGbar_k_val * matDeltaC_k_val + tl.dot(matQbar_k_val, matDeltaH_k_val)

    # * store the first state from the last iteration
    matDeltaCstates_0_ptr = tl.make_block_ptr(
        base=matDeltaC_states + idx_b_NH * str_matDeltaC_states_B_NH + 0,
        shape=(DHQK, DHHV),
        strides=(str_matDeltaC_states_NCDHQK, str_matDeltaC_states_DHHV),
        offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
        block_shape=(siz_b_DHQK, siz_b_DHHV),
        order=(1, 0),
    )
    tl.store(matDeltaCstates_0_ptr, matDeltaC_k_val.to(tl.float32), boundary_check=(0, 1))


def mlstm_chunkwise__recurrent_bw_dC(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    vecF: torch.Tensor,  # (B, NH, NC * L) = (B, NH, S)
    scaM_inter: torch.Tensor,  # (B, NH, NC+1)
    vecM_combine: torch.Tensor,  # (B, NH, S)
    matDeltaH: torch.Tensor,  # (B, NH, S, DHHV)
    vecN_out: torch.Tensor,  # (B, NH, S)
    matDeltaC_last: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    qk_scale: float = None,
    chunk_size: int = 64,
    save_states_every_nth_chunk: int = 1,
    num_warps: int | None = None,
    num_stages: int | None = None,
    eps: float = 0.0,
) -> torch.Tensor:  # matDeltaC_states (B, NH, (NC+1) * DHQK, DHHV)
    """Computes only the deltaC gradients for the backward pass.
    The other gradients are computed in the other (kernel) function.
    We do not need to compute the gradients for the denominator, as it cancels out in the forward in the groupnorm.
    """
    B, NH, S, DHQK, DHHV = *matQ.shape, matDeltaH.shape[-1]
    _dtype, _device = matQ.dtype, matQ.device
    L = chunk_size
    assert is_power_of_2(L), "Chunk size must be a power of 2."
    assert S % L == 0, "S must be divisible by chunk_size."
    NC = S // L

    assert save_states_every_nth_chunk > 0, "save_states_every_nth_chunk must be positive."
    assert save_states_every_nth_chunk <= NC, "save_states_every_nth_chunk must be <= NC."

    assert is_power_of_2(
        save_states_every_nth_chunk
    ), f"save_states_every_nth_chunk must be a power of 2. Got {save_states_every_nth_chunk}."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    USE_LAST_STATE = matDeltaC_last is not None

    num_chunks_saved = NC // save_states_every_nth_chunk

    matDeltaC_states = torch.empty(
        (B, NH, (num_chunks_saved + 1) * DHQK, DHHV),
        dtype=torch.float32,
        device=_device,
    )

    siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1 if num_stages is None else num_stages
    if num_warps is None:
        num_warps = 4 if siz_b_DHQK == 64 else 2

    grid = (num_b_DHQK, num_b_DHHV, B * NH)
    _mlstm_chunkwise__recurrent_bw_dC_kernel[grid](
        matQ=matQ,
        vecF=vecF,
        scaM_inter=scaM_inter,
        vecM_combine=vecM_combine,
        matDeltaH=matDeltaH,
        vecN_out=vecN_out,
        matDeltaC_last=matDeltaC_last,
        matDeltaC_states=matDeltaC_states,
        qk_scale=qk_scale,
        str_matQ_B_NH=matQ.stride(1),
        str_matQ_S=matQ.stride(2),
        str_matQ_DHQK=matQ.stride(3),
        str_vecF_B_NH=vecF.stride(1),
        str_scaM_inter_B_NH=scaM_inter.stride(1),
        str_scaM_inter_NC=scaM_inter.stride(2),
        str_vecM_combine_B_NH=vecM_combine.stride(1),
        str_vecM_combine_S=vecM_combine.stride(2),
        str_matDeltaH_B_NH=matDeltaH.stride(1),
        str_matDeltaH_S=matDeltaH.stride(2),
        str_matDeltaH_DHHV=matDeltaH.stride(3),
        str_vecN_out_B_NH=vecN_out.stride(1),
        str_vecN_out_S=vecN_out.stride(2),
        str_matDeltaC_last_B_NH=matDeltaC_last.stride(1) if USE_LAST_STATE else 0,
        str_matDeltaC_last_DHQK=matDeltaC_last.stride(2) if USE_LAST_STATE else 0,
        str_matDeltaC_last_DHHV=matDeltaC_last.stride(3) if USE_LAST_STATE else 0,
        str_matDeltaC_states_B_NH=matDeltaC_states.stride(1),
        str_matDeltaC_states_NCDHQK=matDeltaC_states.stride(2),
        str_matDeltaC_states_DHHV=matDeltaC_states.stride(3),
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
        USE_LAST_STATE=USE_LAST_STATE,
        DTYPE=torch2triton_dtype(_dtype),
        EPS=eps,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return matDeltaC_states
