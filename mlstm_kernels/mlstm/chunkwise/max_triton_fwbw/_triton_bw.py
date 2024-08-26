import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional
from einops import rearrange
from ....kernel_utils import contiguous_noctx, is_power_of_2, torch2triton_dtype

"""Triton.

Backward pass of the mLSTM chunkwise formulation.

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


@triton.jit
def _mlstm_chunkwise__recurrent_bw_dC_kernel(
    matQ,  # (B, NH, S, DHQK)
    vecB,  # (B, NH, NC, L)
    scaM_inter,  # (B, NH, NC+1)
    vecM_combine,  # (B, NH, S)
    matDeltaH,  # (B, NH, S, DHHV)
    vecN_out,  # (B, NH, S)
    matDeltaC_last,  # (B, NH, DHQK, DHHV)
    matDeltaC_states,  # (B, NH, (NC+1) * DHQK, DHHV)
    qk_scale,
    str_matQ_B_NH,
    str_matQ_S,
    str_matQ_DHQK,
    str_vecB_B_NH,
    str_vecB_NC,
    str_vecB_L,
    str_scaM_inter_B_NH,
    str_scaM_inter_NC,
    str_vecM_combine_B_NH,
    str_vecM_combine_S,
    str_matDeltaH_B_NH,
    str_matDeltaH_S,
    str_matDeltaH_DHHV,
    str_vecN_out_B_NH,
    str_vecN_out_S,
    str_matDeltaC_last_B_NH,
    str_matDeltaC_last_DHQK,
    str_matDeltaC_last_DHHV,
    str_matDeltaC_states_B_NH,
    str_matDeltaC_states_NCDHQK,
    str_matDeltaC_states_DHHV,
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
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
        matDeltaC_k_val = tl.load(matDeltaC_last_ptr, boundary_check=(1, 0)).to(
            tl.float32
        )

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
        matDeltaCstates_k_ptr = tl.make_block_ptr(
            base=matDeltaC_states
            + idx_b_NH * str_matDeltaC_states_B_NH
            + k * DHQK * DHHV,
            shape=(DHQK, DHHV),
            strides=(str_matDeltaC_states_NCDHQK, str_matDeltaC_states_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        # ? end pointers

        # * store matDeltaC_k_val from previous iteration in HBM
        tl.store(
            matDeltaCstates_k_ptr, matDeltaC_k_val.to(DTYPE), boundary_check=(0, 1)
        )

        # * compute matDeltaC_km1_val
        # load scaG_k, vecB_k, scaM_inter_km1, scaM_inter_k, vecM_combine_k
        # scaG_k_val is the last val of vecB_k
        scaG_k_val = tl.load(
            vecB + idx_b_NH * str_vecB_B_NH + (k - 1) * str_vecB_NC + (L - 1)
        ).to(tl.float32)
        vecB_val = tl.load(
            vecB + idx_b_NH * str_vecB_B_NH + (k - 1) * str_vecB_NC + tl.arange(0, L),
        ).to(tl.float32)
        scaM_inter_km1_val = tl.load(
            scaM_inter + idx_b_NH * str_scaM_inter_B_NH + (k - 1)
        ).to(tl.float32)
        scaM_inter_k_val = tl.load(scaM_inter + idx_b_NH * str_scaM_inter_B_NH + k).to(
            tl.float32
        )
        vecM_combine_k_val = tl.load(
            vecM_combine
            + idx_b_NH * str_vecM_combine_B_NH
            + (k - 1) * L
            + tl.arange(0, L)
        ).to(tl.float32)

        # compute scaGbar_k, vecBbar_k
        scaGbar_k_val = tl.exp(scaG_k_val + scaM_inter_km1_val - scaM_inter_k_val)
        vecBbar_k_val = tl.exp(vecB_val + scaM_inter_km1_val - vecM_combine_k_val)

        # compute matQbar_k (DHQK, L) (Note: matQ_k is transposed)
        matQ_k_val = tl.load(matQ_k_ptr, boundary_check=(0, 1)).to(tl.float32)
        matQbar_k_val = (matQ_k_val * vecBbar_k_val[None, :] * qk_scale).to(DTYPE)

        # load vecN_out_k, matDeltaH_k
        vecN_out_k_val = tl.load(
            vecN_out + idx_b_NH * str_vecN_out_B_NH + (k - 1) * L + tl.arange(0, L)
        ).to(tl.float32)  # (L,)
        matDeltaH_k_val = tl.load(matDeltaH_ptr, boundary_check=(0, 1)).to(
            DTYPE
        )  # (L, DHHV)
        # compute matDeltaHinter_k
        matDeltaHinter_k_val = (matDeltaH_k_val / (vecN_out_k_val[:, None] + EPS)).to(
            DTYPE
        )

        # compute matDeltaC_km1
        matDeltaC_k_val = scaGbar_k_val * matDeltaC_k_val + tl.dot(
            matQbar_k_val, matDeltaHinter_k_val
        )

    # * store the first state from the last iteration
    matDeltaCstates_0_ptr = tl.make_block_ptr(
        base=matDeltaC_states + idx_b_NH * str_matDeltaC_states_B_NH + 0,
        shape=(DHQK, DHHV),
        strides=(str_matDeltaC_states_NCDHQK, str_matDeltaC_states_DHHV),
        offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
        block_shape=(siz_b_DHQK, siz_b_DHHV),
        order=(1, 0),
    )
    tl.store(matDeltaCstates_0_ptr, matDeltaC_k_val.to(DTYPE), boundary_check=(0, 1))


@contiguous_noctx
def _mlstm_chunkwise__recurrent_bw_dC(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    scaM_inter: torch.Tensor,  # (B, NH, NC+1)
    vecM_combine: torch.Tensor,  # (B, NH, S)
    matDeltaH: torch.Tensor,  # (B, NH, S, DHHV)
    vecN_out: torch.Tensor,  # (B, NH, S)
    matDeltaC_last: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    qk_scale: float = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> torch.Tensor:  # matDeltaC_states (B, NH, (NC+1) * DHQK, DHHV)
    """Computes only the deltaC gradients for the backward pass.
    The other gradients are computed in the other (kernel) function.
    We do not need to compute the gradients for the denominator, as it cancels out in the forward in the groupnorm.
    """
    B, NH, S, DHQK, DHHV = *matQ.shape, matDeltaH.shape[-1]
    NC = NUM_CHUNKS
    L = CHUNK_SIZE
    _dtype, _device = matQ.dtype, matQ.device

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    USE_LAST_STATE = matDeltaC_last is not None

    matDeltaC_states = torch.empty(
        (B, NH, (NC + 1) * DHQK, DHHV), dtype=_dtype, device=_device
    )

    siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1
    num_warps = 4 if siz_b_DHQK == 64 else 2

    grid = (num_b_DHQK, num_b_DHHV, B * NH)
    _mlstm_chunkwise__recurrent_bw_dC_kernel[grid](
        matQ=matQ,
        vecB=vecB,
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
        str_vecB_B_NH=vecB.stride(1),
        str_vecB_NC=vecB.stride(2),
        str_vecB_L=vecB.stride(3),
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
        USE_LAST_STATE=USE_LAST_STATE,
        DTYPE=torch2triton_dtype(_dtype),
        EPS=EPS,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return matDeltaC_states


@triton.jit
def _mlstm_chunkwise__parallel_bw_dQKV_kernel(
    matQ,  # (B, NH, S, DHQK)
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    vecB,  # (B, NH, NC, L)
    vecI,  # (B, NH, NC, L)
    vecM_combine,  # (B, NH, S)
    scaM_inter,  # (B, NH, NC+1)
    matC_states,  # (B, NH, NC * DHQK, DHHV)
    matDeltaH,  # (B, NH, S, DHHV)
    vecN_out,  # (B, NH, S)
    matDeltaC_states,  # (B, NH, NC * DHQK, DHHV)
    matDeltaQ,  # (B, NH, S, DHQK)
    matDeltaK,  # (B, NH, S, DHQK)
    matDeltaV,  # (num_b_DHQK, B, NH, S, DHHV)
    qk_scale,
    str_matQK_B_NH,  # shared with matQ, matDeltaQ, matK, matDeltaK
    str_matQK_S,
    str_matQK_DHQK,
    str_matDV_num_b_DHQK,
    str_matHV_B_NH,  # shared with matDeltaV, matDeltaH
    str_matHV_S,
    str_matHV_DHHV,
    str_vecBI_B_NH,
    str_vecBI_NC,
    str_vecBI_L,
    str_vecM_combine_B_NH,
    str_vecM_combine_S,
    str_scaM_inter_B_NH,
    str_scaM_inter_NC,
    str_matC_states_B_NH,
    str_matC_states_NCDHQK,
    str_matC_states_DHHV,
    str_vecN_out_B_NH,
    str_vecN_out_S,
    str_matDeltaC_states_B_NH,
    str_matDeltaC_states_NCDHQK,
    str_matDeltaC_states_DHHV,
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
    idx_b_DHQK, idx_b_NC, idx_b_BNH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )
    # num_cta_BNH = tl.num_programs(2)

    # [intra] recompute matDbar
    # load gates (L,)
    vecB_val = tl.load(
        vecB + idx_b_BNH * str_vecBI_B_NH + idx_b_NC * str_vecBI_NC + tl.arange(0, L)
    ).to(tl.float32)  # (L,)
    vecI_val = tl.load(
        vecI + idx_b_BNH * str_vecBI_B_NH + idx_b_NC * str_vecBI_NC + tl.arange(0, L)
    ).to(tl.float32)  # (L,)
    # load vecM_combine (L,)
    vecM_combine_val = tl.load(
        vecM_combine
        + idx_b_BNH * str_vecM_combine_B_NH
        + idx_b_NC * L
        + tl.arange(0, L)
    )

    # compute gate matrix matDbar (L, L)
    idx_mask = tl.arange(0, L)
    mask = idx_mask[:, None] >= idx_mask[None, :]
    matD_full_val = vecB_val[:, None] - vecB_val[None, :] + vecI_val[None, :]
    matD_val = tl.where(mask, matD_full_val, -float("inf"))
    matDbar_val = tl.exp(matD_val - vecM_combine_val[:, None])

    # [inter,intra] compute vecAbar, vecBbar
    # load scaM_inter_k, scaM_inter_km1
    scaM_inter_km1_val = tl.load(
        scaM_inter + idx_b_BNH * str_scaM_inter_B_NH + idx_b_NC * str_scaM_inter_NC
    )  # (1,)
    scaM_inter_k_val = tl.load(
        scaM_inter
        + idx_b_BNH * str_scaM_inter_B_NH
        + (idx_b_NC + 1) * str_scaM_inter_NC
    )  # (1,)

    # scaG is the last val of vecB
    scaG_val = tl.load(
        vecB + idx_b_BNH * str_vecBI_B_NH + idx_b_NC * str_vecBI_NC + L - 1
    )  # (1,)
    vecA_val = scaG_val - vecB_val + vecI_val  # (L,)

    vecAbar_val = tl.exp(vecA_val - scaM_inter_k_val)  # (L,)
    vecBbar_val = tl.exp(vecB_val + scaM_inter_km1_val - vecM_combine_val)  # (L, )
    # tl.device_print("vecAbar", vecAbar_val)
    # tl.device_print("vecBbar", vecBbar_val)

    # [intra] recompute matS, matSbar
    # NOTE: we compute only a part of matS as we do not sum over the full DHQK dim
    # we have to sum the matV after the kernel.
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
        shape=(S, DHQK),
        strides=(str_matQK_S, str_matQK_DHQK),
        offsets=(idx_b_NC * L, idx_b_DHQK * siz_b_DHQK),
        block_shape=(L, siz_b_DHQK),
        order=(1, 0),
    )
    matQ_val = tl.load(matQ_ptr, boundary_check=(0, 1)).to(DTYPE)  # (L, siz_b_DHQK)
    matK_val = tl.load(matK_ptr, boundary_check=(0, 1)).to(DTYPE)  # (siz_b_DHQK, L)
    matS_val = tl.dot(matQ_val, tl.trans(matK_val)) * qk_scale
    matSbar_val = (matS_val * matDbar_val).to(DTYPE)  # (L, L)

    matKbar_val = (matK_val.to(tl.float32) * vecAbar_val[:, None]).to(
        DTYPE
    )  # (L, siz_b_DHQK)
    tl.static_print("matKbar_val", matKbar_val)

    matDeltaQ_inter_val = tl.zeros((L, siz_b_DHQK), dtype=tl.float32)
    matDeltaK_inter_val = tl.zeros((L, siz_b_DHQK), dtype=tl.float32)
    matDeltaS_val = tl.zeros((L, L), dtype=tl.float32)
    # TODO for later: matDeltaDbar for gate delta errors
    for idx_b_DHHV in range(tl.cdiv(DHHV, siz_b_DHHV)):
        # ? pointers for iteration
        # load pointers:
        matV_ptr = tl.make_block_ptr(
            base=matV + idx_b_BNH * str_matHV_B_NH,
            shape=(S, DHHV),
            strides=(str_matHV_S, str_matHV_DHHV),
            offsets=(idx_b_NC * L, idx_b_DHHV * siz_b_DHHV),
            block_shape=(L, siz_b_DHHV),
            order=(1, 0),
        )
        matDeltaH_ptr = tl.make_block_ptr(
            base=matDeltaH + idx_b_BNH * str_matHV_B_NH,
            shape=(S, DHHV),
            strides=(str_matHV_S, str_matHV_DHHV),
            offsets=(idx_b_NC * L, idx_b_DHHV * siz_b_DHHV),
            block_shape=(L, siz_b_DHHV),
            order=(1, 0),
        )
        matC_km1_ptr = tl.make_block_ptr(
            base=matC_states
            + idx_b_BNH * str_matC_states_B_NH
            + idx_b_NC * DHQK * DHHV,
            shape=(DHHV, DHQK),  # transposed
            strides=(str_matC_states_DHHV, str_matC_states_NCDHQK),
            offsets=(idx_b_DHHV * siz_b_DHHV, idx_b_DHQK * siz_b_DHQK),
            block_shape=(siz_b_DHHV, siz_b_DHQK),
            order=(0, 1),
        )
        matDeltaC_k_ptr = tl.make_block_ptr(
            base=matDeltaC_states
            + idx_b_BNH * str_matDeltaC_states_B_NH
            + idx_b_NC * DHQK * DHHV,
            shape=(DHQK, DHHV),
            strides=(str_matDeltaC_states_NCDHQK, str_matDeltaC_states_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(1, 0),
        )
        vecN_out_ptr = (
            vecN_out + idx_b_BNH * str_vecN_out_B_NH + idx_b_NC * L + tl.arange(0, L)
        )
        # store pointers:
        matDeltaV_ptr = tl.make_block_ptr(
            base=matDeltaV
            + idx_b_DHQK * str_matDV_num_b_DHQK
            + idx_b_BNH * str_matHV_B_NH,
            shape=(S, DHHV),
            strides=(str_matHV_S, str_matHV_DHHV),
            offsets=(idx_b_NC * L, idx_b_DHHV * siz_b_DHHV),
            block_shape=(L, siz_b_DHHV),
            order=(1, 0),
        )
        # ? end pointers

        # [inter,intra] matDeltaV
        # matDeltaH = matDeltaH / vecN_out
        matDeltaH_val = tl.load(matDeltaH_ptr, boundary_check=(0, 1)).to(
            tl.float32
        )  # (L, siz_b_DHHV)
        vecN_out_val = tl.load(vecN_out_ptr).to(tl.float32)  # (L,)
        matDeltaH_val = matDeltaH_val / (vecN_out_val[:, None] + EPS)  # (L, siz_b_DHHV)

        # [inter] matDeltaK += (matV @ matDeltaC_k.transpose()) * vecAbar
        matV_val = tl.load(matV_ptr, boundary_check=(0, 1)).to(DTYPE)  # (L, siz_b_DHHV)
        matDeltaC_k_val = tl.load(matDeltaC_k_ptr, boundary_check=(0, 1)).to(
            DTYPE
        )  # (siz_b_DHQK, siz_b_DHHV)

        matDeltaK_inter_val += (
            tl.dot(matV_val, tl.trans(matDeltaC_k_val)) * vecAbar_val[:, None]
        )  # (L, siz_b_DHQK)

        # [inter] matDeltaQ += matDeltaH @ matC_km1.transpose() * vecBbar
        matC_km1_val = tl.load(matC_km1_ptr, boundary_check=(0, 1))  # (siz_b_DHHV, siz_b_DHQK)
        tl.static_print("matDeltaH_val", matDeltaH_val)
        tl.static_print("vecBbar_val", vecBbar_val)
        matDeltaQ_inter_val += tl.dot(
            (matDeltaH_val * vecBbar_val[:, None]).to(DTYPE), (matC_km1_val * qk_scale).to(DTYPE)
        )  # (L, siz_b_DHQK)

        # [intra] matDeltaS += (matDeltaH @ matV.transpose()) * matDbar
        matDeltaS_val += (
            tl.dot(matDeltaH_val.to(DTYPE), tl.trans(matV_val)) * matDbar_val
        )  # (L, L)

        # [inter, intra] matDeltaV = matSbar.transpose() @ matDeltaH + matKbar @ matDeltaC_k
        matDeltaV_val = tl.dot(tl.trans(matSbar_val), matDeltaH_val.to(DTYPE)) + tl.dot(
            matKbar_val, matDeltaC_k_val
        )
        # store matDeltaV (NOTE: each idx_b_DHQK stores its own contribution in HBM, it will be summed outside the kernel)
        tl.store(
            matDeltaV_ptr,
            matDeltaV_val.to(matDeltaV_ptr.dtype.element_ty),
            boundary_check=(0, 1),
        )

    matDeltaS_val = matDeltaS_val.to(DTYPE)
    # [intra] matDeltaK = matDeltaS.transpose() @ matQ * qk_scale
    matDeltaK_val = matDeltaK_inter_val + (
        tl.dot(tl.trans(matDeltaS_val), matQ_val) * qk_scale
    )
    # [intra] matDeltaQ = matDeltaS @ matK * qk_scale
    matDeltaQ_val = matDeltaQ_inter_val + (tl.dot(matDeltaS_val, matK_val) * qk_scale)

    # store matDeltaQ, matDeltaK
    matDeltaK_ptr = tl.make_block_ptr(
        base=matDeltaK + idx_b_BNH * str_matQK_B_NH,
        shape=(S, DHQK),
        strides=(str_matQK_S, str_matQK_DHQK),
        offsets=(idx_b_NC * L, idx_b_DHQK * siz_b_DHQK),
        block_shape=(L, siz_b_DHQK),
        order=(1, 0),
    )
    matDeltaQ_ptr = tl.make_block_ptr(
        base=matDeltaQ + idx_b_BNH * str_matQK_B_NH,
        shape=(S, DHQK),
        strides=(str_matQK_S, str_matQK_DHQK),
        offsets=(idx_b_NC * L, idx_b_DHQK * siz_b_DHQK),
        block_shape=(L, siz_b_DHQK),
        order=(1, 0),
    )
    tl.store(
        matDeltaK_ptr,
        matDeltaK_val.to(matDeltaK.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(
        matDeltaQ_ptr,
        matDeltaQ_val.to(matDeltaQ.dtype.element_ty),
        boundary_check=(0, 1),
    )


def _mlstm_chunkwise__parallel_bw_dQKV(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    vecM_combine: torch.Tensor,  # (B, NH, S) = (B, NH, NC * L)
    scaM_inter: torch.Tensor,  # (B, NH, NC+1)
    matC_states: torch.Tensor,  # (B, NH, NC * DHQK, DHHV)
    matDeltaH: torch.Tensor,  # (B, NH, S, DHHV)
    vecN_out: torch.Tensor,  # (B, NH, S)
    matDeltaC_states: torch.Tensor,  # (B, NH, NC * DHQK, DHHV)
    qk_scale: float = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:  # matDeltaQ (B,NH,S,DHQK), matDeltaK (B,NH,S,DHQK), matDeltaV (B,NH,S,DHHV)
    B, NH, S, DHQK, DHHV = *matQ.shape, matV.shape[-1]
    NC = NUM_CHUNKS
    L = CHUNK_SIZE
    _dtype, _device = matQ.dtype, matQ.device

    assert is_power_of_2(L), "Chunk size must be a power of 2."

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    siz_b_DHQK = min(
        32 if _dtype == torch.float32 else 64, triton.next_power_of_2(DHQK)
    )
    siz_b_DHHV = min(
        32 if _dtype == torch.float32 else 64, triton.next_power_of_2(DHHV)
    )

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1
    num_warps = 4 if siz_b_DHQK == 64 else 2

    # TODO make empty tensors later
    matDeltaQ = torch.ones((B, NH, S, DHQK), dtype=_dtype, device=_device)
    matDeltaK = torch.ones((B, NH, S, DHQK), dtype=_dtype, device=_device)
    # each b_DHQK thread block computes the contribution of its siz_b_DHQK block of matDeltaC
    # we need to sum them up to get the final result (we do this outside the kernel)
    matDeltaV = torch.ones((num_b_DHQK, B, NH, S, DHHV), dtype=_dtype, device=_device)

    grid = (num_b_DHQK, NC, B * NH)
    _mlstm_chunkwise__parallel_bw_dQKV_kernel[grid](
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecB=vecB,
        vecI=vecI,
        vecM_combine=vecM_combine,
        scaM_inter=scaM_inter,
        matC_states=matC_states,
        matDeltaH=matDeltaH,
        vecN_out=vecN_out,
        matDeltaC_states=matDeltaC_states,
        matDeltaQ=matDeltaQ,
        matDeltaK=matDeltaK,
        matDeltaV=matDeltaV,
        qk_scale=qk_scale,
        str_matQK_B_NH=matQ.stride(1),
        str_matQK_S=matQ.stride(2),
        str_matQK_DHQK=matQ.stride(3),
        str_matDV_num_b_DHQK=matV.stride(0),
        str_matHV_B_NH=matV.stride(1),
        str_matHV_S=matV.stride(2),
        str_matHV_DHHV=matV.stride(3),
        str_vecBI_B_NH=vecI.stride(1),
        str_vecBI_NC=vecI.stride(2),
        str_vecBI_L=vecI.stride(3),
        str_vecM_combine_B_NH=vecM_combine.stride(1),
        str_vecM_combine_S=vecM_combine.stride(2),
        str_scaM_inter_B_NH=scaM_inter.stride(1),
        str_scaM_inter_NC=scaM_inter.stride(2),
        str_matC_states_B_NH=matC_states.stride(1),
        str_matC_states_NCDHQK=matC_states.stride(2),
        str_matC_states_DHHV=matC_states.stride(3),
        str_vecN_out_B_NH=vecN_out.stride(1),
        str_vecN_out_S=vecN_out.stride(2),
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
        DTYPE=torch2triton_dtype(_dtype),
        EPS=EPS,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    # sum up the contributions of each siz_b_DHQK block
    matDeltaV = matDeltaV.sum(dim=0)  # (B, NH, S, DHHV)

    return matDeltaQ, matDeltaK, matDeltaV
