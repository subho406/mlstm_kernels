from collections.abc import Callable

import torch

from .native_step import mlstm_recurrent_step__native_fw
from .triton_step import mlstm_recurrent_step__triton_fw
from .triton_step_fused import mlstm_recurrent_step__triton_fused_fw


def _mlstm_recurrent_sequence_loop_fw(
    mlstm_step_fn: Callable,
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHV)
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH)
    return_last_states: bool = False,
    return_all_states: bool = False,
    eps: float = 1e-6,
    **kwargs,
) -> tuple[
    torch.Tensor,  # (B, NH, S, DHV)
    torch.Tensor,  # (B, NH, S, DHQK)
    torch.Tensor,  # (B, NH, S)
    None
    | (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ),  # (matC_state_last (B, NH, DHQK, DHV), vecN_state_last (B, NH, DHQK), vecM_state_last (B, NH, 1))
    None
    | (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ),  # (matC_states (B, NH, S, DHQK, DHV), vecN_states (B, NH, S, DHQK), vecM_states (B, NH, S))
]:
    B, NH, S, DHQK = matQ.shape
    DHV = matV.shape[-1]
    device = matQ.device
    dtype = matQ.dtype

    if matC_initial is not None:
        assert vecN_initial is not None and scaM_initial is not None, "Initial states must be provided together."
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
        vecF_t, vecI_t = vecF[:, :, t, None], vecI[:, :, t, None]  # (B, NH, 1)

        # projections
        vecQ_t, vecK_t, vecV_t = (
            matQ[:, :, t, :],  # (B, NH, DHQK)
            matK[:, :, t, :],  # (B, NH, DHQK)
            matV[:, :, t, :],  # (B, NH, DHV)
        )

        # step
        vecH, (matC_state, vecN_state, vecM_state) = mlstm_step_fn(
            matC_old=matC_state,
            vecN_old=vecN_state,
            scaM_old=vecM_state,
            vecQ=vecQ_t,
            vecK=vecK_t,
            vecV=vecV_t,
            scaI=vecI_t,
            scaF=vecF_t,
            eps=eps,
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
        ret_tuple += ((matC_states, vecN_states, vecM_states),)
    else:
        ret_tuple += (None,)

    return ret_tuple


def mlstm_recurrent_sequence__native_fw(
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
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    ret_tuple = _mlstm_recurrent_sequence_loop_fw(
        mlstm_step_fn=mlstm_recurrent_step__native_fw,
        matQ=q,
        matK=k,
        matV=v,
        vecI=i,
        vecF=f,
        matC_initial=c_initial,
        vecN_initial=n_initial,
        scaM_initial=m_initial,
        return_last_states=return_last_states,
        eps=eps,
        return_all_states=False,
    )
    if return_last_states:
        return ret_tuple[0], ret_tuple[3]
    else:
        return ret_tuple[0]


def mlstm_recurrent_sequence__triton_step_fw(
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
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    ret_tuple = _mlstm_recurrent_sequence_loop_fw(
        mlstm_step_fn=mlstm_recurrent_step__triton_fw,
        matQ=q,
        matK=k,
        matV=v,
        vecI=i,
        vecF=f,
        matC_initial=c_initial,
        vecN_initial=n_initial,
        scaM_initial=m_initial,
        return_last_states=return_last_states,
        eps=eps,
        return_all_states=False,
    )
    if return_last_states:
        return ret_tuple[0], ret_tuple[3]
    else:
        return ret_tuple[0]


def mlstm_recurrent_sequence__triton_step_fused_fw(
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
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    ret_tuple = _mlstm_recurrent_sequence_loop_fw(
        mlstm_step_fn=mlstm_recurrent_step__triton_fused_fw,
        matQ=q,
        matK=k,
        matV=v,
        vecI=i,
        vecF=f,
        matC_initial=c_initial,
        vecN_initial=n_initial,
        scaM_initial=m_initial,
        return_last_states=return_last_states,
        eps=eps,
        return_all_states=False,
    )
    if return_last_states:
        return ret_tuple[0], ret_tuple[3]
    else:
        return ret_tuple[0]
