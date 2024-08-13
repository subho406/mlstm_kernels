# Copyright JKU Linz 2024
# Author: Maximilian Beck
import math

import torch
import torch.nn.functional as F

"""
PyTorch.

This module contains the recurrent implementation of the mLSTM.
"""


def recurrent_sequence_fw(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    initial_matC: torch.Tensor = None,
    initial_vecN: torch.Tensor = None,
    initial_scaM: torch.Tensor = None,
    eps: float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    """This is the core mLSTM operation in stabilized recurrent form. It operates on a full
    input sequence of length S. It is stabilized by adding a third "max" state.
    This is analog to [1].

    [1] Milakov, Maxim, and Natalia Gimelshein. “Online Normalizer Calculation for Softmax.” arXiv, July 28, 2018.
        http://arxiv.org/abs/1805.02867.


    Args:
        matQ (torch.Tensor): (B, NH, S, DHQK)
        matK (torch.Tensor): (B, NH, S, DHQK)
        matV (torch.Tensor): (B, NH, S, DHV)
        vecI (torch.Tensor): (B, NH, S)
        vecF (torch.Tensor): (B, NH, S)
        initial_matC (torch.Tensor, optional): (B, NH, DHQK, DHV). Defaults to None.
        initial_vecN (torch.Tensor, optional): (B, NH, DHQK). Defaults to None.
        initial_scaM (torch.Tensor, optional): (B, NH). Defaults to None.
        eps (float, optional): Used for building the forgetgate matrix. Defaults

    Returns:
        torch.Tensor: (B, NH, S, DHV), hidden states
    """

    B, NH, S, DHQK = matQ.shape
    DHV = matV.shape[-1]
    device = matQ.device
    dtype = matQ.dtype

    if initial_matC is not None:
        assert (
            initial_vecN is not None and initial_scaM is not None
        ), "Initial states must be provided together."
        matC_state, vecN_state, vecM_state = initial_matC, initial_vecN, initial_scaM
    else:
        # memory state
        matC_state = torch.zeros((B, NH, DHQK, DHV), dtype=dtype, device=device)
        # normalizer state
        vecN_state = torch.zeros((B, NH, DHQK), dtype=dtype, device=device)
        # max state
        vecM_state = torch.zeros((B, NH, 1), dtype=dtype, device=device)

    hidden_states = []
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
        h, (matC_state, vecN_state, vecM_state) = recurrent_step_fw(
            matC_old=matC_state,
            vecN_old=vecN_state,
            scaM_old=vecM_state,
            vecQ=vecQ_t,
            vecK=vecK_t,
            vecV=vecV_t,
            scaI=vecI_t,
            scaF=vecF_t,
            EPS=eps,
        )
        hidden_states.append(h)

    hidden_states = torch.stack(hidden_states, dim=-2)  # (B, NH, S, DHV)
    return hidden_states


def recurrent_step_fw(
    matC_old: torch.Tensor,
    vecN_old: torch.Tensor,
    scaM_old: torch.Tensor,
    vecQ: torch.Tensor,
    vecK: torch.Tensor,
    vecV: torch.Tensor,
    scaI: torch.Tensor,
    scaF: torch.Tensor,
    EPS: float = 1e-6,
    **kwargs,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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

    vecK_scaled = vecK / math.sqrt(DHQK)  # (B, NH, DHQK)

    matC_state_new = scaF_act[:, :, :, None] * matC_old + scaI_act[:, :, :, None] * (
        (vecK_scaled[:, :, :, None] @ vecV[:, :, None, :])
    )  # (B, NH, DHQK, DHV)
    vecN_state_new = scaF_act * vecN_old + scaI_act * vecK_scaled  # (B, NH, DHQK)

    h_num = vecQ[:, :, None, :] @ matC_state_new  # (B, NH, 1, DHV)
    h_num = h_num.squeeze(2)  # (B, NH, DHV)

    qn_dotproduct = vecQ[:, :, None, :] @ vecN_state_new[:, :, :, None]  # (B, NH, 1, 1)
    qn_dotproduct = qn_dotproduct.squeeze(2) # (B, NH, 1)
    max_val = torch.exp(-scaM_state_new)  # (B, NH, 1)
    h_denom = torch.maximum(qn_dotproduct.abs(), max_val) + EPS # (B, NH, 1)
    h = h_num / h_denom  # (B, NH, DHV) / (B, NH, 1) = (B, NH, DHV)

    return h, (matC_state_new, vecN_state_new, scaM_state_new)
