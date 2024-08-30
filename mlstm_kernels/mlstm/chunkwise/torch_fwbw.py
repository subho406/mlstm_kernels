# Copyright JKU Linz 2024
# Author: Maximilian Beck
import torch
from einops import rearrange
import torch.nn.functional as F
from typing import Optional
from torch.amp import custom_fwd, custom_bwd

from ...kernel_utils import contiguous


"""PyTorch.

Forward and backward pass of the mLSTM chunkwise formulation.

Notation:
Dimensions:
    B: batch size
    NH: number of heads
    S: sequence length
    DH: hidden dimension
    NC: number of chunks
    L: chunk size

Variables:
    vecA, a: forward gate contribution, contribution of forget gates from last chunk state C_{k-1} to current timestep t
    vecB, b: backward gate contribution, contribution of forget and input gates up to next chunk state C_k (form current timestep t)
    scaG, g: "go through" gate contribution, contribution of forget gates from C_{k-1} to C_k.
"""


def _mlstm_chunkwise__recurrent_fw_C(
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    vecB: torch.Tensor,  # (B, NH, NC, L) # cumsum(logsigmoid(f))
    vecI: torch.Tensor,  # (B, NH, NC, L)
    matC_states: torch.Tensor = None,  # (B, NH, (NC + 1) * DHQK, DHV)
    vecN_states: torch.Tensor = None,  # (B, NH, (NC + 1) * DHQK)
    scaMinter_states: torch.Tensor = None,  # (B, NH, (NC + 1)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaMinter_initial: torch.Tensor = None,  # (B, NH)
    qk_scale: float = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:  # matC_states (B, NH, (NC+1) * DHQK, DHV), vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1))
    B, NH, S, DHQK, DHV = *matK.shape, matV.shape[-1]
    NC = NUM_CHUNKS
    _dtype, _device = matK.dtype, matK.device

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    # initialize the states tensors
    if matC_states is None:
        matC_states = torch.zeros(
            (B, NH, (NC + 1) * DHQK, DHV), dtype=_dtype, device=_device
        )
    if vecN_states is None:
        vecN_states = torch.zeros(
            (B, NH, (NC + 1) * DHQK), dtype=_dtype, device=_device
        )
    if scaMinter_states is None:
        scaMinter_states = torch.zeros((B, NH, (NC + 1)), dtype=_dtype, device=_device)

    # assign the initial states to the running states
    matC_k = (
        torch.zeros((B, NH, DHQK, DHV), dtype=_dtype, device=_device)
        if matC_initial is None
        else matC_initial
    )
    vecN_k = (
        torch.zeros((B, NH, DHQK), dtype=_dtype, device=_device)
        if vecN_initial is None
        else vecN_initial
    )
    scaM_inter_k = (
        torch.zeros((B, NH), dtype=_dtype, device=_device)
        if scaMinter_initial is None
        else scaMinter_initial
    )
    vecA = (vecB[..., -1, None] - vecB) + vecI
    scaG = vecB[..., -1]
    scaA_max = vecA.max(-1).values

    for k in range(0, NUM_CHUNKS):
        # store the states from the previous iteration before updating them
        # in the first iteration, these are the initial states
        matC_states[:, :, k * DHQK : (k + 1) * DHQK, :] = matC_k
        vecN_states[:, :, k * DHQK : (k + 1) * DHQK] = vecN_k
        scaMinter_states[:, :, k] = scaM_inter_k

        # m_k update
        scaA_max_k = scaA_max[:, :, k]
        scaG_k = scaG[:, :, k]
        scaM_inter_k_next = torch.max(scaG_k + scaM_inter_k, scaA_max_k)
        # C_k update
        matK_chunk = matK[:, :, k * CHUNK_SIZE : (k + 1) * CHUNK_SIZE, :]  # * qk_scale
        matV_chunk = matV[:, :, k * CHUNK_SIZE : (k + 1) * CHUNK_SIZE, :]
        vecA_k = vecA[:, :, k, :]

        vecAbar_k = torch.exp(vecA_k - scaM_inter_k_next[..., None])[:, :, :, None]

        matK_chunk_gated = matK_chunk * vecAbar_k

        scaGbar_k = torch.exp(scaG_k + scaM_inter_k - scaM_inter_k_next)[:, :, None]

        # print(
        #     "fw_C: k",
        #     k,
        #     "scaGbar_k",
        #     scaGbar_k,
        # )
        # print(
        #     "fw_C: k",
        #     k,
        #     "vecAbar_k",
        #     vecAbar_k,
        # )

        # NOTE: no update in-place (i.e. +=) as this gives error for autograd backward
        matC_k_next = scaGbar_k[..., None] * matC_k + matK_chunk_gated.transpose(
            -2, -1
        ) @ (matV_chunk)

        # n_k update
        vecN_k_next = scaGbar_k * vecN_k + matK_chunk_gated.transpose(-2, -1).sum(-1)

        # move to the next iteration
        scaM_inter_k = scaM_inter_k_next
        matC_k = matC_k_next
        vecN_k = vecN_k_next

    # store the states from the last iteration
    matC_states[:, :, -DHQK:, :] = matC_k
    vecN_states[:, :, -DHQK:] = vecN_k
    scaMinter_states[:, :, -1] = scaM_inter_k

    return matC_states, vecN_states, scaMinter_states


def _mlstm_chunkwise__parallel_fw_H(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    # these states must be all states up to the last chunk, i.e. :-1
    matC_states: torch.Tensor,  # (B, NH, NC * DHQK, DHV)
    vecN_states: torch.Tensor,  # (B, NH, NC * DHQK)
    scaMinter_states: torch.Tensor,  # (B, NH, NC)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    qk_scale: float,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> tuple[
    torch.Tensor, torch.Tensor
]:  # matH_out (B, NH, S, DHV), vecN_out (B, NH, S), vecM_out (B, NH, S)
    _device = matQ.device
    NC, L = NUM_CHUNKS, CHUNK_SIZE
    matC_k_states = rearrange(
        matC_states, "b nh (nc dhqk) dhv -> b nh nc dhqk dhv", nc=NC
    )
    vecN_k_states = rearrange(vecN_states, "b nh (nc dhqk) -> b nh nc dhqk", nc=NC)
    scaMinter_k_states = scaMinter_states

    matQ = rearrange(matQ, "b nh (nc l) dh -> b nh nc l dh", l=L)
    matK = rearrange(matK, "b nh (nc l) dh -> b nh nc l dh", l=L)
    matV = rearrange(matV, "b nh (nc l) dh -> b nh nc l dh", l=L)

    ltr = torch.tril(
        torch.ones(
            (L, L),
            dtype=torch.bool,
            device=_device,
        )
    )

    #! compute the H_states in parallel

    # ? Compute intra chunk contribution: H_intra
    matF_logsig_chunk = vecB[:, :, :, :, None] - vecB[:, :, :, None, :]

    matF_logsig_mask_chunk = torch.where(ltr, matF_logsig_chunk, -float("inf"))

    matLogD_chunk = matF_logsig_mask_chunk + vecI[:, :, :, None, :]

    # max_state intra
    vecMintra_k = torch.max(
        matLogD_chunk, dim=-1, keepdim=False
    ).values  # (B, NH, NC, L)

    # max_state combined
    vecM_b_inter = vecB + scaMinter_k_states[:, :, :, None]  # (B, NH, NC, L)
    vecM_k_combine = torch.maximum(vecM_b_inter, vecMintra_k)  # (B, NH, NC, L)

    vecM_k_combine = vecM_k_combine[:, :, :, :, None]  # (B, NH, NC, L, 1)
    vecM_b_inter = vecM_b_inter[:, :, :, :, None]  # (B, NH, NC, L, 1)

    matLogD_stabilized_chunk = matLogD_chunk - vecM_k_combine
    matD_chunk = torch.exp(matLogD_stabilized_chunk)

    matS_chunk = (matQ @ matK.transpose(-2, -1)) * qk_scale

    matM_chunk = matS_chunk * matD_chunk

    # ? Combine H_intra with H_inter
    vecBbar = torch.exp(vecM_b_inter - vecM_k_combine)
    # print(f"p_fw, vecBbar: {vecBbar}, {vecBbar.shape}")
    matQ_chunk_gated = matQ * vecBbar * qk_scale

    matNumerator_common = (
        matQ_chunk_gated @ matC_k_states + matM_chunk @ matV
    )  # (B, NH, NC, L, DHV)

    vecDenom_l_common = matQ_chunk_gated @ vecN_k_states.unsqueeze(-1) + matM_chunk.sum(
        dim=-1, keepdim=True
    )  # (B, NH, NC, L, 1)

    vecDenom_max_common = torch.maximum(
        torch.abs(vecDenom_l_common), torch.exp(-vecM_k_combine)
    )

    matH_k_chunk = matNumerator_common / (vecDenom_max_common + EPS)

    matH_out = rearrange(matH_k_chunk, "b nh nc l dh -> b nh (nc l) dh")

    # we need the denominator and the overall max state for the backward pass
    vecN_out = rearrange(
        vecDenom_max_common, "b nh nc l 1 -> b nh (nc l)"
    )  # (B, NH, S)
    vecM_out = rearrange(vecM_k_combine, "b nh nc l 1 -> b nh (nc l)")  # (B, NH, S)
    return matH_out, vecN_out, vecM_out


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
    Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],  # last_states (matC_states (B, NH, DHQK, DHV), vecN_states (B, NH, DHQK), scaMinter_states (B, NH))
    Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],  # all_states (matC_states (B, NH, (NC+1) * DHQK, DHV), vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1)))
]:
    B, NH, S, DHQK = matQ.shape
    DHV = matV.shape[-1]
    assert (
        S % CHUNK_SIZE == 0
    ), f"Sequence length {S} is not divisible by chunk size {CHUNK_SIZE}."
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


def _mlstm_chunkwise__recurrent_bw_dC(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    scaM_inter: torch.Tensor,  # (B, NH, NC+1)
    vecM_combine: torch.Tensor,  # (B, NH, S)
    matDeltaH: torch.Tensor,  # (B, NH, S, DHV)
    vecN_out: torch.Tensor,  # (B, NH, S)
    matDeltaC_last: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    qk_scale: float = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> torch.Tensor:  # matDeltaC_states (B, NH, (NC+1) * DHQK, DHV)
    """Computes only the deltaC gradients for the backward pass.
    The other gradients are computed in the other (kernel) function.
    We do not need to compute the gradients for the denominator, as it cancels out in the forward in the groupnorm.
    """
    B, NH, S, DHQK, DHV = *matQ.shape, matDeltaH.shape[-1]
    NC = NUM_CHUNKS
    L = CHUNK_SIZE
    _dtype, _device = matQ.dtype, matQ.device

    matDeltaC_states = torch.zeros(
        (B, NH, (NC + 1) * DHQK, DHV), dtype=_dtype, device=_device
    )

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    if matDeltaC_last is not None:
        matDeltaC_k = matDeltaC_last
    else:
        matDeltaC_k = torch.zeros((B, NH, DHQK, DHV), dtype=_dtype, device=_device)

    scaG = vecB[..., -1]  # (B, NH, NC)

    for k in range(NC, 0, -1):  # goes until 1
        # store the matDeltaC_k from the previous iteration
        # in the first iteration, this is the delta error from the last chunk
        matDeltaC_states[:, :, k * DHQK : (k + 1) * DHQK, :] = matDeltaC_k.clone()

        # load
        scaG_k = scaG[:, :, (k - 1), None]
        scaM_inter_kminus1 = scaM_inter[:, :, (k - 1), None]
        scaM_inter_k = scaM_inter[:, :, k, None]
        scaGbar_k = torch.exp(scaG_k + scaM_inter_kminus1 - scaM_inter_k)[:, :, None]

        vecB_k = vecB[:, :, (k - 1), :]  # (B, NH, L)
        vecM_combine_k = vecM_combine[:, :, (k - 1) * L : k * L]  # (B, NH, L)
        vecBbar_k = torch.exp(vecB_k + scaM_inter_kminus1 - vecM_combine_k)[
            :, :, :, None
        ]  # (B, NH, L, 1)

        # print(
        #     "bw_dC, k:",
        #     k,
        #     "vecBbar_k",
        #     vecBbar_k,
        # )
        # print(
        #     "bw_dC, k:",
        #     k,
        #     "scaGbar_k",
        #     scaGbar_k,
        # )

        matQ_k = matQ[:, :, (k - 1) * L : k * L, :]  # (B, NH, L, DHQK)
        matQbar_k = matQ_k * vecBbar_k * qk_scale

        vecN_k = vecN_out[:, :, (k - 1) * L : k * L, None]  # (B, NH, L, 1)
        matDeltaH_k = matDeltaH[:, :, (k - 1) * L : k * L, :] / (
            vecN_k + EPS
        )  # (B, NH, L, DHV)

        # matDeltaC_k-1 update
        matDeltaC_kminus1 = (
            scaGbar_k * matDeltaC_k + matQbar_k.transpose(-2, -1) @ matDeltaH_k
        )  # (B, NH, DHQK, DHV)

        # move to the next iteration
        matDeltaC_k = matDeltaC_kminus1

    # store the matDeltaC_k from the last iteration
    matDeltaC_states[:, :, :DHQK, :] = matDeltaC_k

    return matDeltaC_states


# maybe also compute deltaI?
def _mlstm_chunkwise__parallel_bw_dQKV(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    vecM_combine: torch.Tensor,  # (B, NH, S) = (B, NH, NC * L)
    scaM_inter: torch.Tensor,  # (B, NH, NC+1)
    matC_states: torch.Tensor,  # (B, NH, NC * DHQK, DHV)
    matDeltaH: torch.Tensor,  # (B, NH, S, DHV)
    vecN_out: torch.Tensor,  # (B, NH, S)
    matDeltaC_states: torch.Tensor,  # (B, NH, NC * DHQK, DHV)
    qk_scale: float = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:  # matDeltaQ (B,NH,S,DHQK), matDeltaK (B,NH,S,DHQK), matDeltaV (B,NH,S,DHV)
    B, NH, S, DHQK, DHV = *matQ.shape, matV.shape[-1]
    NC = NUM_CHUNKS
    L = CHUNK_SIZE
    _dtype, _device = matQ.dtype, matQ.device

    #! intra chunk gradients
    # load / prepare the inputs
    matDeltaH = matDeltaH / (vecN_out[:, :, :, None] + EPS)

    matDeltaH = rearrange(matDeltaH, "b nh (nc l) dh -> b nh nc l dh", l=L)

    matQ = rearrange(matQ, "b nh (nc l) dh -> b nh nc l dh", l=L)
    matK = rearrange(matK, "b nh (nc l) dh -> b nh nc l dh", l=L)
    matV = rearrange(matV, "b nh (nc l) dh -> b nh nc l dh", l=L)

    vecM_combine = rearrange(vecM_combine, "b nh (nc l) -> b nh nc l", l=L)

    ltr = torch.tril(
        torch.ones(
            (L, L),
            dtype=torch.bool,
            device=_device,
        )
    )

    # recompute the gate matrix D
    matF = vecB[:, :, :, :, None] - vecB[:, :, :, None, :]
    matF_mask = torch.where(ltr, matF, -float("inf"))
    matDtilde = matF_mask + vecI[:, :, :, None, :]
    matDbar = torch.exp(matDtilde - vecM_combine[:, :, :, :, None])

    # recompute the S matrix
    matS = (matQ @ matK.transpose(-2, -1)) * qk_scale
    matSbar = matS * matDbar

    # compute the intra delta gradients
    matDeltaV_intra = matSbar.transpose(-2, -1) @ matDeltaH

    matDeltaSbar = matDeltaH @ matV.transpose(-2, -1)
    matDeltaS = matDeltaSbar * matDbar

    # TODO compute matDeltaDtilde for the deltaI gradients
    matDeltaQ_intra = (matDeltaS @ matK) * qk_scale
    matDeltaK_intra = ((matDeltaS).transpose(-2, -1) @ matQ) * qk_scale

    #! inter chunk gradients
    # load / prepare the inputs
    matDeltaC_states = rearrange(
        matDeltaC_states, "b nh (nc dhqk) dhv -> b nh nc dhqk dhv", nc=NC
    )
    matC_states = rearrange(
        matC_states, "b nh (nc dhqk) dhv -> b nh nc dhqk dhv", nc=NC
    )

    vecA = (vecB[..., -1, None] - vecB) + vecI  # (B, NH, NC, L)

    # compute the gates vecA, vecB
    scaM_inter_kminus1 = scaM_inter[:, :, :-1, None]
    scaM_inter_k = scaM_inter[:, :, 1:, None]
    vecBbar = torch.exp(vecB + scaM_inter_kminus1 - vecM_combine)[:, :, :, :, None]
    vecAbar = torch.exp(vecA - scaM_inter_k)[:, :, :, :, None]

    # compute the inter delta gradients
    matDeltaV_inter = (matK * vecAbar) @ matDeltaC_states

    matDeltaK_inter = (matV * vecAbar) @ (matDeltaC_states.transpose(-2, -1))
    matDeltaQ_inter = (matDeltaH * vecBbar) @ (matC_states * qk_scale).transpose(-2, -1)

    # combine the delta gradients
    matDeltaQ = matDeltaQ_intra + matDeltaQ_inter
    matDeltaK = matDeltaK_intra + matDeltaK_inter
    matDeltaV = matDeltaV_intra + matDeltaV_inter

    matDeltaQ = rearrange(matDeltaQ, "b nh nc l dh -> b nh (nc l) dh")
    matDeltaK = rearrange(matDeltaK, "b nh nc l dh -> b nh (nc l) dh")
    matDeltaV = rearrange(matDeltaV, "b nh nc l dh -> b nh (nc l) dh")
    return matDeltaQ, matDeltaK, matDeltaV


def _mlstm_chunkwise_bw(
    ## Forward arguments
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH)
    qk_scale: float = None,
    ## Backward arguments
    matC_all: torch.Tensor = None,  # (B, NH, NC * DHQK, DHV)
    vecN_all: torch.Tensor = None,  # (B, NH, NC * DHQK)
    scaM_all: torch.Tensor = None,  # (B, NH, NC)
    vecN_out: torch.Tensor = None,  # (B, NH, NC * L) = (B, NH, S)
    vecM_out: torch.Tensor = None,  # (B, NH, NC * L) = (B, NH, S)
    matDeltaH: torch.Tensor = None,  # (B, NH, S, DHV)
    matDeltaC_last: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecDeltaN_last: torch.Tensor = None,  # (B, NH, DHQK) # TODO not used, maybe leave out
    scaDeltaM_last: torch.Tensor = None,  # (B, NH) # TODO not used, maybe leave out
    ## Common arguments
    CHUNK_SIZE: int = 64,
    EPS: float = 1e-6,
):
    B, NH, S, DHQK = matQ.shape
    DHV = matV.shape[-1]

    assert (
        S % CHUNK_SIZE == 0
    ), f"Sequence length {S} is not divisible by chunk size {CHUNK_SIZE}."

    NC = S // CHUNK_SIZE

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    vecI = rearrange(vecI, "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE)
    vecF = rearrange(vecF, "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE)

    # compute the gates, the g and the a and b vectors
    vecF_logsig = F.logsigmoid(vecF)
    vecB = vecF_logsig.cumsum(-1)

    #! recompute the "all" states if needed
    if matC_all is None:
        assert (
            (matC_all is None) and (vecN_all is None) and (scaM_all is None)
        ), "Either all or none of the states must be provided."
        matC_all, vecN_all, scaM_all = _mlstm_chunkwise__recurrent_fw_C(
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

    # print("scaM_all", scaM_all)
    # print("vecM_out", vecM_out)
    # print("vecN_out", vecN_out)
    # print("matQ", matQ)
    # print("vecB", vecB)
    # print("matDeltaH", matDeltaH)

    # save inputs
    inp_dict_bw_dC = dict(
        matQ=matQ,
        vecB=vecB,
        scaM_inter=scaM_all,
        vecM_combine=vecM_out,
        matDeltaH=matDeltaH,
        vecN_out=vecN_out,
        matDeltaC_last=matDeltaC_last,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )
    # torch.save(inp_dict_bw_dC, "./inputs_bw_dC")

    #! recurrent backward: compute the deltaC gradients
    matDeltaC_states = _mlstm_chunkwise__recurrent_bw_dC(
        matQ=matQ,  # (B, NH, S, DHQK)
        vecB=vecB,  # (B, NH, NC, L)
        scaM_inter=scaM_all,  # (B, NH, NC+1)
        vecM_combine=vecM_out,  # (B, NH, S)
        matDeltaH=matDeltaH,  # (B, NH, S, DHV)
        vecN_out=vecN_out,  # (B, NH, S)
        matDeltaC_last=matDeltaC_last,  # (B, NH, DHQK, DHV)
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )  # (B, NH, NC * DHQK, DHV)

    # print("matC_states", matC_all, matC_all.shape)
    # print("matDeltaC_states", matDeltaC_states, matDeltaC_states.shape)
    # print("scaM_all", scaM_all, scaM_all.shape)
    #! parallel backward: compute the deltaQ, deltaK, deltaV, deltaI gradients
    # scaM_inter_k_states = scaM_all[:, :, 1:]  # take the last NC states
    matC_k_states = matC_all[:, :, :-DHQK, :]  # take the first NC states

    matDeltaC_k_states = matDeltaC_states[:, :, DHQK:, :]  # take the last NC states

    # print("matC_k_states", matC_k_states, matC_k_states.shape)
    # print("matDeltaC_k_states", matDeltaC_k_states, matDeltaC_k_states.shape)
    # print("scaM_inter_k_states", scaM_inter_k_states, scaM_inter_k_states.shape)

    inp_dict_bw_dQKV = dict(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecB=vecB,
        vecI=vecI,
        vecM_combine=vecM_out,
        scaM_inter=scaM_all,  # (B, NH, NC)
        matC_states=matC_k_states,  # (B, NH, NC * DHQK, DHV)
        matDeltaH=matDeltaH,  # (B, NH, S, DHV)
        vecN_out=vecN_out,  # (B, NH, S)
        matDeltaC_states=matDeltaC_k_states,  # (B, NH, NC * DHQK, DHV)
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )

    # torch.save(inp_dict_bw_dQKV, "./inputs_bw_dQKV")

    matDeltaQ, matDeltaK, matDeltaV = _mlstm_chunkwise__parallel_bw_dQKV(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecB=vecB,
        vecI=vecI,
        vecM_combine=vecM_out,
        scaM_inter=scaM_all,  # (B, NH, NC)
        matC_states=matC_k_states,  # (B, NH, NC * DHQK, DHV)
        matDeltaH=matDeltaH,  # (B, NH, S, DHV)
        vecN_out=vecN_out,  # (B, NH, S)
        matDeltaC_states=matDeltaC_k_states,  # (B, NH, NC * DHQK, DHV)
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )

    #! postprocessing: compute deltaF and deltaI gradients
    ## ? postprocessing
    vecF = rearrange(vecF, "b nh nc l -> b nh (nc l)")
    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    vecDeltaFbar_acc = (matQ * matDeltaQ - matK * matDeltaK).sum(-1)
    vecDeltaFbar = vecDeltaFbar_acc.flip(-1).cumsum(-1).flip(-1)
    vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)
    ## ? end postprocessing
    # compute deltaI
    # both are equivalent:
    vecDeltaI = (matV * matDeltaV).sum(-1)
    # vecDeltaI = (matK * matDeltaK).sum(-1)

    matDeltaC_initial = (
        matDeltaC_states[:, :, :DHQK, :] if matC_initial is not None else None
    )
    vecDeltaN_initial = (
        torch.zeros_like(vecN_initial) if vecN_initial is not None else None
    )
    scaDeltaM_initial = (
        torch.zeros_like(scaM_initial) if scaM_initial is not None else None
    )

    return (
        matDeltaQ,
        matDeltaK,
        matDeltaV,
        vecDeltaI,
        vecDeltaF,
        matDeltaC_initial,
        vecDeltaN_initial,
        scaDeltaM_initial,
    )


def mlstm_chunkwise_fw(
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
    CHUNK_SIZE: int = 64,
    EPS: float = 1e-6,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    matH_out, _, _, last_states, _ = _mlstm_chunkwise_fw(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecI=vecI,
        vecF=vecF,
        matC_initial=matC_initial,
        vecN_initial=vecN_initial,
        scaM_initial=scaM_initial,
        qk_scale=qk_scale,
        return_last_states=return_last_states,
        return_all_states=False,
        EPS=EPS,
        CHUNK_SIZE=CHUNK_SIZE,
    )
    if return_last_states:
        return matH_out, last_states
    else:
        return matH_out


def mlstm_chunkwise_fwbw(
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
    RECOMPUTE_STATES_IN_BW: bool = True,
    CHUNK_SIZE: int = 64,
    EPS: float = 1e-6,
):
    matH, matC_last, vecN_last, scaM_last = _mlstm_chunkwise_fwbw.apply(
        matQ,
        matK,
        matV,
        vecI,
        vecF,
        matC_initial,
        vecN_initial,
        scaM_initial,
        qk_scale,
        return_last_states,
        RECOMPUTE_STATES_IN_BW,
        CHUNK_SIZE,
        EPS,
    )
    if return_last_states:
        return matH, (matC_last, vecN_last, scaM_last)
    else:
        return matH


## PyTorch Autograd Function - Boilerplate
class _mlstm_chunkwise_fwbw(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    @contiguous
    def forward(
        ctx,
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
        RECOMPUTE_STATES_IN_BW: bool = True,
        CHUNK_SIZE: int = 64,
        EPS: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, NH, S, DHQK = matQ.shape
        if qk_scale is None:
            qk_scale = DHQK**-0.5

        matH_out, vecN_out, vecM_out, last_states, all_states = _mlstm_chunkwise_fw(
            matQ=matQ,
            matK=matK,
            matV=matV,
            vecI=vecI,
            vecF=vecF,
            matC_initial=matC_initial,
            vecN_initial=vecN_initial,
            scaM_initial=scaM_initial,
            qk_scale=qk_scale,
            return_last_states=return_last_states,
            return_all_states=(not RECOMPUTE_STATES_IN_BW),
            EPS=EPS,
            CHUNK_SIZE=CHUNK_SIZE,
        )

        if return_last_states:
            (matC_last, vecN_last, scaM_last) = last_states
        else:
            (matC_last, vecN_last, scaM_last) = (None, None, None)

        if all_states is not None:
            matC_all, vecN_all, scaM_all = all_states
        else:
            matC_all, vecN_all, scaM_all = (None, None, None)

        ctx.save_for_backward(
            matQ,
            matK,
            matV,
            vecI,
            vecF,
            matC_initial,
            vecN_initial,
            scaM_initial,
            matC_all,
            vecN_all,
            scaM_all,
            vecN_out,
            vecM_out,
            torch.tensor(CHUNK_SIZE),
            torch.tensor(EPS),
        )
        return matH_out, matC_last, vecN_last, scaM_last

    @staticmethod
    @custom_bwd(device_type="cuda")
    @contiguous
    def backward(ctx, matDeltaH, matDeltaC_last, vecDeltaN_last, scaDeltaM_last):
        (
            matQ,
            matK,
            matV,
            vecI,
            vecF,
            matC_initial,
            vecN_initial,
            scaM_initial,
            matC_all,
            vecN_all,
            scaM_all,
            vecN_out,
            vecM_out,
            CHUNK_SIZE,
            EPS,
        ) = ctx.saved_tensors
        B, NH, S, DHQK = matQ.shape
        DHV = matV.shape[-1]

        (
            matDeltaQ,
            matDeltaK,
            matDeltaV,
            vecDeltaI,
            vecDeltaF,
            matDeltaC_initial,
            vecDeltaN_initial,
            scaDeltaM_initial,
        ) = _mlstm_chunkwise_bw(
            matQ=matQ,
            matK=matK,
            matV=matV,
            vecI=vecI,
            vecF=vecF,
            matC_initial=matC_initial,
            vecN_initial=vecN_initial,
            scaM_initial=scaM_initial,
            matC_all=matC_all,
            vecN_all=vecN_all,
            scaM_all=scaM_all,
            vecN_out=vecN_out,
            vecM_out=vecM_out,
            matDeltaH=matDeltaH,
            matDeltaC_last=matDeltaC_last,
            vecDeltaN_last=vecDeltaN_last,
            scaDeltaM_last=scaDeltaM_last,
            CHUNK_SIZE=int(CHUNK_SIZE),
            EPS=float(EPS),
        )

        return (
            matDeltaQ,
            matDeltaK,
            matDeltaV,
            vecDeltaI,
            vecDeltaF,
            matDeltaC_initial,
            vecDeltaN_initial,
            scaDeltaM_initial,
            None,
            None,
            None,
            None,
            None,
        )
