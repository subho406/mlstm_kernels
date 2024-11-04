# Copyright JKU Linz 2024
# Author: Maximilian Beck
import torch
from einops import rearrange
import torch.nn.functional as F
from typing import Optional
from collections.abc import Callable
from torch.amp import custom_fwd, custom_bwd

from ....kernel_utils import contiguous

from ._torch_fw import _mlstm_chunkwise__recurrent_fw_C

# PyTorch.

# Forward and backward pass of the mLSTM chunkwise formulation.

# Notation:
# Dimensions:
# B: batch size
# NH: number of heads
# S: sequence length
# DH: hidden dimension
# NC: number of chunks
# L: chunk size

# Variables:
# vecA, a: forward gate contribution, contribution of forget gates from last chunk state C_{k-1} to current timestep t
# vecB, b: backward gate contribution, contribution of forget and input gates up to next chunk state C_k (form current timestep t)
# scaG, g: "go through" gate contribution, contribution of forget gates from C_{k-1} to C_k.


def _mlstm_chunkwise__recurrent_bw_dC(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    vecF: torch.Tensor,  # (B, NH, NC, L)
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

    vecFlogsig = F.logsigmoid(vecF)
    vecB = vecFlogsig.cumsum(-1)  # (B, NH, NC, L)

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
    vecF: torch.Tensor,  # (B, NH, NC, L)
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

    vecFlogsig = F.logsigmoid(vecF)
    vecB = vecFlogsig.cumsum(-1)  # (B, NH, NC, L)

    # recompute the gate matrix D
    # matF = vecB[:, :, :, :, None] - vecB[:, :, :, None, :]
    # matF_mask = torch.where(ltr, matF, -float("inf"))
    # matDtilde = matF_mask + vecI[:, :, :, None, :]
    # stable way
    matFlogsig_tril = vecFlogsig[:, :, :, :, None].repeat(1, 1, 1, 1, L).tril(-1)
    matFlogsig_cum = matFlogsig_tril.cumsum(-2)
    matFlogsig_mask = torch.where(ltr, matFlogsig_cum, -float("inf"))
    matD = matFlogsig_mask + vecI[:, :, :, None, :]

    matDbar = torch.exp(matD - vecM_combine[:, :, :, :, None])

    # recompute the S matrix
    matS = (matQ @ matK.transpose(-2, -1)) * qk_scale
    matSbar = matS * matDbar

    # compute the intra delta gradients
    matDeltaV_intra = matSbar.transpose(-2, -1) @ matDeltaH

    matDeltaSbar = matDeltaH @ matV.transpose(-2, -1)
    matDeltaS = matDeltaSbar * matDbar

    matDeltaDbar = matDeltaSbar * matS
    matDeltaD = matDeltaDbar * matDbar  # (B, NH, NC, L, L)

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

    scaG = vecB[..., -1, None]  # (B, NH, NC)
    # print("scaG\n", scaG, scaG.shape)
    print("vecA\n", vecA, vecA.shape)
    print("vecB\n", vecB, vecB.shape)

    # compute the gates vecA, vecB
    scaM_inter_kminus1 = scaM_inter[:, :, :-1, None]
    scaM_inter_k = scaM_inter[:, :, 1:, None]
    vecBbar = torch.exp(vecB + scaM_inter_kminus1 - vecM_combine)[
        :, :, :, :, None
    ]  # (B, NH, NC, L, 1)
    vecAbar = torch.exp(vecA - scaM_inter_k)[:, :, :, :, None]
    scaGbar = torch.exp(scaG + scaM_inter_kminus1 - scaM_inter_k)  # (B, NH, NC, 1)

    # compute the inter delta gradients
    matDeltaV_inter = (matK * vecAbar) @ matDeltaC_states

    matDeltaKbar = matV @ (matDeltaC_states.transpose(-2, -1))
    matDeltaK_inter = (
        matDeltaKbar * vecAbar
    )  # (matV * vecAbar) @ (matDeltaC_states.transpose(-2, -1))

    matDeltaQbar = matDeltaH @ (matC_states * qk_scale).transpose(-2, -1)
    # matDeltaQbar2 = matDeltaH @ (matC_states).transpose(-2, -1)
    matDeltaQ_inter = (
        matDeltaQbar * vecBbar
    )  # (matDeltaH * vecBbar) @ (matC_states * qk_scale).transpose(-2, -1)

    # combine the delta gradients
    matDeltaQ = matDeltaQ_intra + matDeltaQ_inter
    matDeltaK = matDeltaK_intra + matDeltaK_inter
    matDeltaV = matDeltaV_intra + matDeltaV_inter

    matDeltaQ = rearrange(matDeltaQ, "b nh nc l dh -> b nh (nc l) dh")
    matDeltaK = rearrange(matDeltaK, "b nh nc l dh -> b nh (nc l) dh")
    matDeltaV = rearrange(matDeltaV, "b nh nc l dh -> b nh (nc l) dh")

    # inter gate contributions
    # fgate grads + igate grads:
    vecDeltaA = (matDeltaKbar * matK).sum(
        -1, keepdim=True
    ) * vecAbar  # (B, NH, NC, L, 1)
    vecDeltaA = vecDeltaA.squeeze(-1)
    # fgate grads only:
    scaDeltaGbar = matC_states * matDeltaC_states  # (B, NH, NC, DHQK, DHV)
    scaDeltaG = scaDeltaGbar.sum(-1).sum(-1, keepdim=True) * scaGbar  # (B, NH, NC, 1)
    vecDeltaB = (matDeltaQbar * matQ).sum(
        -1, keepdim=True
    ) * vecBbar  # (B, NH, NC, L, 1)
    vecDeltaB = vecDeltaB.squeeze(-1)
    print("vecDeltaB\n", vecDeltaB, vecDeltaB.shape)

    # intra forgetgate contributions from matDeltaD
    vecDeltaF_f2onw_intra = (
        matDeltaD.cumsum(-1).tril(-1).sum(-2)[..., :-1]
    )  # (B, NH, NC, L-1) # only from f2 onwards

    # inter forgetgate contributions from matDeltaD
    # vecDeltaG is added to all
    # TODO here error?
    vecDeltaF_vecBcontr_inter = (
        vecDeltaB.flip(-1).cumsum(-1).flip(-1)  # .squeeze(-1)
    )  # (B, NH, NC, L)
    # TODO here error?
    vecDeltaF_vecAcontr_f2onw_inter = vecDeltaA.cumsum(-1)[..., :-1]  # (B, NH, NC, L-1)

    print(
        "vecDeltaF_vecBcontr_inter\n",
        vecDeltaF_vecBcontr_inter,
        vecDeltaF_vecBcontr_inter.shape,
    )
    print(
        "vecDeltaF_vecAcontr_f2onw_inter\n",
        vecDeltaF_vecAcontr_f2onw_inter,
        vecDeltaF_vecAcontr_f2onw_inter.shape,
    )
    print("vecDeltaG\n", scaDeltaG, scaDeltaG.shape)
    print("vecDeltaF_f2onw_intra\n", vecDeltaF_f2onw_intra, vecDeltaF_f2onw_intra.shape)

    # sum up the forgetgate contributions
    vecDeltaF = vecDeltaF_vecBcontr_inter
    # vecDeltaF = torch.zeros_like(vecDeltaF_vecBcontr_inter)

    vecDeltaF[..., 1:] += vecDeltaF_vecAcontr_f2onw_inter
    vecDeltaF[..., 1:] += vecDeltaF_f2onw_intra
    vecDeltaF += scaDeltaG  # broadcast to all forgetgates

    vecDeltaF = vecDeltaF * torch.sigmoid(-vecF)

    vecDeltaF = rearrange(vecDeltaF, "b nh nc l -> b nh (nc l)")

    vecDeltaI = (rearrange(matK, "b nh nc l dh -> b nh (nc l) dh") * matDeltaK).sum(-1)
    # vDv_temp = (matQ * matDeltaQ - matK * matDeltaK).sum(-1)
    # vecDeltaFbar = vDv_temp.flip(-1).cumsum(-1).flip(-1)
    # vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)
    # vecDeltaI = torch.zeros((B, NH, NC * L), device=_device, dtype=_dtype)
    # vecDeltaF = torch.zeros((B, NH, NC * L), device=_device, dtype=_dtype)
    return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF


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

    #! recompute the "all" states if needed
    if matC_all is None:
        assert (
            (matC_all is None) and (vecN_all is None) and (scaM_all is None)
        ), "Either all or none of the states must be provided."
        matC_all, vecN_all, scaM_all = _mlstm_chunkwise__recurrent_fw_C(
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

    # print("scaM_all", scaM_all)
    # print("vecM_out", vecM_out)
    # print("vecN_out", vecN_out)
    # print("matQ", matQ)
    # print("vecB", vecB)
    # print("matDeltaH", matDeltaH)

    # save inputs
    inp_dict_bw_dC = dict(
        matQ=matQ,
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
        vecF=vecF,  # (B, NH, NC, L)
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

    matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF = (
        _mlstm_chunkwise__parallel_bw_dQKV(
            matQ=matQ,
            matK=matK,
            matV=matV,
            vecF=vecF,
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
    )

    # #! postprocessing: compute deltaF and deltaI gradients
    # ## ? postprocessing
    # vecF = rearrange(vecF, "b nh nc l -> b nh (nc l)")
    # # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    # vecDeltaFbar_acc = (matQ * matDeltaQ - matK * matDeltaK).sum(-1)
    # vecDeltaFbar = vecDeltaFbar_acc.flip(-1).cumsum(-1).flip(-1)
    # vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)
    # ## ? end postprocessing
    # # compute deltaI
    # # both are equivalent:
    # vecDeltaI = (matV * matDeltaV).sum(-1)
    # # vecDeltaI = (matK * matDeltaK).sum(-1)

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
