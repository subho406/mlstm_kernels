# Copyright JKU Linz 2024
# Author: Maximilian Beck
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.amp import custom_bwd, custom_fwd

from ....torch.utils import contiguous

# PyTorch.

# Forward and backward pass of the mLSTM chunkwise formulation.

# Notation:
# Dimensions:
#     B: batch size
#     NH: number of heads
#     S: sequence length
#     DH: hidden dimension
#     NC: number of chunks
#     L: chunk size

# Variables:
#     vecA, a: forward gate contribution, contribution of forget gates from last chunk state C_{k-1} to current timestep t
#     vecB, b: backward gate contribution, contribution of forget and input gates up to next chunk state C_k (form current timestep t)
#     scaG, g: "go through" gate contribution, contribution of forget gates from C_{k-1} to C_k.


def _mlstm_chunkwise__recurrent_fw_C(
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    vecF: torch.Tensor,  # (B, NH, NC, L)
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
        matC_states = torch.zeros((B, NH, (NC + 1) * DHQK, DHV), dtype=_dtype, device=_device)
    if vecN_states is None:
        vecN_states = torch.zeros((B, NH, (NC + 1) * DHQK), dtype=_dtype, device=_device)
    if scaMinter_states is None:
        scaMinter_states = torch.zeros((B, NH, (NC + 1)), dtype=_dtype, device=_device)

    # assign the initial states to the running states
    matC_k = torch.zeros((B, NH, DHQK, DHV), dtype=_dtype, device=_device) if matC_initial is None else matC_initial
    vecN_k = torch.zeros((B, NH, DHQK), dtype=_dtype, device=_device) if vecN_initial is None else vecN_initial
    scaM_inter_k = (
        torch.zeros((B, NH), dtype=_dtype, device=_device) if scaMinter_initial is None else scaMinter_initial
    )
    vecFlogsig = F.logsigmoid(vecF)

    # unstable vecA computation:
    # vecA = (vecB[..., -1, None] - vecB) + vecI  # (B, NH, NC, L)
    # stable vecA computation:
    vecA = (
        torch.cat(
            [
                vecFlogsig[..., 1:].flip(-1).cumsum(-1).flip(-1),
                torch.zeros((B, NH, NC, 1), device=_device, dtype=_dtype),
            ],
            dim=-1,
        )
        + vecI
    )  # (B, NH, NC, L)
    scaG = vecFlogsig.sum(-1)
    scaA_max = vecA.max(-1).values
    # print(f"fw_C: vecA: {vecA}, {vecA.shape}")
    # print(f"fw_C: scaG: {scaG}, {scaG.shape}")

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
        matC_k_next = scaGbar_k[..., None] * matC_k + matK_chunk_gated.transpose(-2, -1) @ (matV_chunk)

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
    vecF: torch.Tensor,  # (B, NH, NC, L)
    qk_scale: float,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:  # matH_out (B, NH, S, DHV), vecN_out (B, NH, S), vecM_out (B, NH, S)
    _device = matQ.device
    NC, L = NUM_CHUNKS, CHUNK_SIZE
    matC_k_states = rearrange(matC_states, "b nh (nc dhqk) dhv -> b nh nc dhqk dhv", nc=NC)
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

    vecFlogsig = F.logsigmoid(vecF)

    vecB = vecFlogsig.cumsum(-1)

    #! compute the H_states in parallel

    # ? Compute intra chunk contribution: H_intra
    # compute the gate matrix D
    # unsable way
    # matF = vecB[:, :, :, :, None] - vecB[:, :, :, None, :]
    # matF_mask = torch.where(ltr, matF, -float("inf"))
    # matDtilde = matF_mask + vecI[:, :, :, None, :]
    # stable way
    matFlogsig_tril = vecFlogsig[:, :, :, :, None].repeat(1, 1, 1, 1, L).tril(-1)
    matFlogsig_cum = matFlogsig_tril.cumsum(-2)
    matFlogsig_mask = torch.where(ltr, matFlogsig_cum, -float("inf"))
    matD = matFlogsig_mask + vecI[:, :, :, None, :]

    # max_state intra
    vecMintra_k = torch.max(matD, dim=-1, keepdim=False).values  # (B, NH, NC, L)

    # max_state combined
    vecM_b_inter = vecB + scaMinter_k_states[:, :, :, None]  # (B, NH, NC, L)
    vecM_k_combine = torch.maximum(vecM_b_inter, vecMintra_k)  # (B, NH, NC, L)

    vecM_k_combine = vecM_k_combine[:, :, :, :, None]  # (B, NH, NC, L, 1)
    vecM_b_inter = vecM_b_inter[:, :, :, :, None]  # (B, NH, NC, L, 1)

    matDbar = torch.exp(matD - vecMintra_k[..., :, None])

    matS_chunk = (matQ @ matK.transpose(-2, -1)) * qk_scale

    matM_chunk = matS_chunk * matDbar

    # ? Combine H_intra with H_inter
    vecBbar = torch.exp(vecM_b_inter - vecM_k_combine)
    # print(f"p_fw, vecBbar: {vecBbar}, {vecBbar.shape}")
    matQ_chunk_gated = matQ * vecBbar * qk_scale

    matNumerator_common = matQ_chunk_gated @ matC_k_states + matM_chunk @ matV  # (B, NH, NC, L, DHV)

    vecDenom_l_common = matQ_chunk_gated @ vecN_k_states.unsqueeze(-1) + matM_chunk.sum(
        dim=-1, keepdim=True
    )  # (B, NH, NC, L, 1)

    vecDenom_max_common = torch.maximum(torch.abs(vecDenom_l_common), torch.exp(-vecM_k_combine))

    matH_k_chunk = matNumerator_common / (vecDenom_max_common + EPS)

    matH_out = rearrange(matH_k_chunk, "b nh nc l dh -> b nh (nc l) dh")

    # we need the denominator and the overall max state for the backward pass
    vecN_out = rearrange(vecDenom_max_common, "b nh nc l 1 -> b nh (nc l)")  # (B, NH, S)
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

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    #! materialize the  C_k, n_k, m_k states for each chunk
    matC_k_states, vecN_k_states, scaMinter_k_states = _mlstm_chunkwise__recurrent_fw_C(
        matK=matK,
        matV=matV,
        vecF=vecF,
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
        vecF=vecF,
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
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
