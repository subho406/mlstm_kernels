# Copyright JKU Linz 2024
# Author: Maximilian Beck
import math
from typing import Optional
import torch
import torch.nn.functional as F

"""
PyTorch.

This module contains the recurrent implementation of the mLSTM.
"""


def mlstm_recurrent_sequence_torch_autograd(
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


def recurrent_sequence_fw(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    vecI: torch.Tensor,  # (B, NH, S, 1)
    vecF: torch.Tensor,  # (B, NH, S, 1)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH)
    return_last_states: bool = False,
    return_all_states: bool = False,
    EPS: float = 1e-6,
    **kwargs,
) -> tuple[
    torch.Tensor,  # (B, NH, S, DHV)
    torch.Tensor,  # (B, NH, S, DHQK)
    torch.Tensor,  # (B, NH, S)
    Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],  # (matC_state_last (B, NH, DHQK, DHV), vecN_state_last (B, NH, DHQK), vecM_state_last (B, NH, 1))
    Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],  # (matC_states (B, NH, S, DHQK, DHV), vecN_states (B, NH, S, DHQK), vecM_states (B, NH, S))
]:
    B, NH, S, DHQK = matQ.shape
    DHV = matV.shape[-1]
    device = matQ.device
    dtype = matQ.dtype

    if matC_initial is not None:
        assert (
            vecN_initial is not None and scaM_initial is not None
        ), "Initial states must be provided together."
        assert scaM_initial.dim() == 2, "Initial states must be 2D."
        matC_state, vecN_state, vecM_state = (
            matC_initial,
            vecN_initial,
            scaM_initial[:, :, None],
        )
    else:
        # memory state
        matC_state = torch.zeros((B, NH, DHQK, DHV), dtype=dtype, device=device)
        # normalizer state
        vecN_state = torch.zeros((B, NH, DHQK), dtype=dtype, device=device)
        # max state
        vecM_state = torch.zeros((B, NH, 1), dtype=dtype, device=device)

    if return_all_states:
        matC_list = []
        matC_list.append(matC_state)

    vecH_list = []
    vecN_list, vecM_list = [], []
    vecN_list.append(vecN_state)
    vecM_list.append(vecM_state)
    for t in range(S):
        # gates
        vecF_t, vecI_t = vecF[:, :, t], vecI[:, :, t]  # (B, NH, 1)

        # projections
        vecQ_t, vecK_t, vecV_t = (
            matQ[:, :, t, :],  # (B, NH, DHQK)
            matK[:, :, t, :],  # (B, NH, DHQK)
            matV[:, :, t, :],  # (B, NH, DHV)
        )

        # step
        vecH, (matC_state, vecN_state, vecM_state) = recurrent_step_fw(
            matC_old=matC_state,
            vecN_old=vecN_state,
            scaM_old=vecM_state,
            vecQ=vecQ_t,
            vecK=vecK_t,
            vecV=vecV_t,
            scaI=vecI_t,
            scaF=vecF_t,
            EPS=EPS,
        )
        vecH_list.append(vecH)
        vecN_list.append(vecN_state)
        vecM_list.append(vecM_state)

        if return_all_states:
            matC_list.append(matC_state)

    matH = torch.stack(vecH_list, dim=-2)  # (B, NH, S, DHV)
    vecN_states = torch.stack(vecN_list, dim=-2)  # (B, NH, S, DHQK)
    vecM_states = torch.cat(vecM_list, dim=-1)  # (B, NH, S)

    ret_tuple = (matH, vecN_states, vecM_states)

    if return_last_states:
        ret_tuple += ((matC_state, vecN_state, vecM_state),)
    else:
        ret_tuple += (None,)

    if return_all_states:
        matC_states = torch.stack(matC_list, dim=-3)  # (B, NH, S, DHQK, DHV)
        matC_states.retain_grad()
        ret_tuple += ((matC_states, vecN_states, vecM_states),)
    else:
        ret_tuple += (None,)

    return ret_tuple


def recurrent_step_fw(
    matC_old: torch.Tensor,  # (B, NH, DHQK, DHV)
    vecN_old: torch.Tensor,  # (B, NH, DHQK)
    scaM_old: torch.Tensor,  # (B, NH, 1)
    vecQ: torch.Tensor,  # (B, NH, DHQK)
    vecK: torch.Tensor,  # (B, NH, DHQK)
    vecV: torch.Tensor,  # (B, NH, DHV)
    scaI: torch.Tensor,  # (B, NH, 1)
    scaF: torch.Tensor,  # (B, NH, 1)
    EPS: float = 1e-6,
    **kwargs,
) -> tuple[
    torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:  # vecH, (matC_state_new (B, NH, DHQK, DHV), vecN_state_new (B, NH, DHQK), vecM_state_new (B, NH, 1))
    """This is a single step of the mLSTM operation in recurrent form.

    Args:
        matC_old (torch.Tensor): (B, NH, DHQK, DHV)
        vecN_old (torch.Tensor): (B, NH, DHQK)
        scaM_old (torch.Tensor): (B, NH, 1)
        vecQ (torch.Tensor): (B, NH, DHQK)
        vecK (torch.Tensor): (B, NH, DHQK)
        vecV (torch.Tensor): (B, NH, DHV)
        scaI (torch.Tensor): (B, NH, 1)
        scaF (torch.Tensor): (B, NH, 1)
        eps (float, optional): Used for building the forgetgate matrix. Defaults to 1e-6.

    Returns:
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            (hidden_state [B, NH, DHV], (c_state_new [B, NH, DHQK, DHV], n_state_new [B, NH, DHQK]], m_state_new [B, NH, 1]))
    """
    B, NH, DHQK = vecQ.shape

    # gates
    scaF_log = torch.nn.functional.logsigmoid(scaF)

    # update rule
    scaM_state_new = torch.max(scaF_log + scaM_old, scaI)  # (B, NH, 1)

    scaF_act = torch.exp(scaF_log + scaM_old - scaM_state_new)  # (B, NH, 1)
    scaI_act = torch.exp(scaI - scaM_state_new)  # (B, NH, 1)

    vecQ_scaled = vecQ / math.sqrt(DHQK)  # (B, NH, DHQK)

    matC_state_new = scaF_act[:, :, :, None] * matC_old + scaI_act[:, :, :, None] * (
        vecK[:, :, :, None] @ vecV[:, :, None, :]
    )  # (B, NH, DHQK, DHV)
    vecN_state_new = scaF_act * vecN_old + scaI_act * vecK  # (B, NH, DHQK)

    h_num = vecQ_scaled[:, :, None, :] @ matC_state_new  # (B, NH, 1, DHV)
    h_num = h_num.squeeze(2)  # (B, NH, DHV)

    qn_dotproduct = (
        vecQ_scaled[:, :, None, :] @ vecN_state_new[:, :, :, None]
    )  # (B, NH, 1, 1)
    qn_dotproduct = qn_dotproduct.squeeze(2)  # (B, NH, 1)
    max_val = torch.exp(-scaM_state_new)  # (B, NH, 1)
    h_denom = torch.maximum(qn_dotproduct.abs(), max_val) + EPS  # (B, NH, 1)
    h = h_num / h_denom  # (B, NH, DHV) / (B, NH, 1) = (B, NH, DHV)

    return h, (matC_state_new, vecN_state_new, scaM_state_new)
