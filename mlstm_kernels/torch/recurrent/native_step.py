# Maximilian Beck
import torch

"""
PyTorch.

This module contains the recurrent step implementation of the mLSTM.
"""


def mlstm_recurrent_step__native_fw(
    matC_old: torch.Tensor,  # (B, NH, DHQK, DHV)
    vecN_old: torch.Tensor,  # (B, NH, DHQK)
    scaM_old: torch.Tensor,  # (B, NH, 1)
    vecQ: torch.Tensor,  # (B, NH, DHQK)
    vecK: torch.Tensor,  # (B, NH, DHQK)
    vecV: torch.Tensor,  # (B, NH, DHV)
    scaI: torch.Tensor,  # (B, NH, 1)
    scaF: torch.Tensor,  # (B, NH, 1)
    eps: float = 1e-6,
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

    vecQ_scaled = vecQ * (DHQK ** (-0.5))  # (B, NH, DHQK)

    matC_state_new = scaF_act[:, :, :, None] * matC_old + scaI_act[:, :, :, None] * (
        vecK[:, :, :, None] @ vecV[:, :, None, :]
    )  # (B, NH, DHQK, DHV)
    vecN_state_new = scaF_act * vecN_old + scaI_act * vecK  # (B, NH, DHQK)

    h_num = vecQ_scaled[:, :, None, :] @ matC_state_new  # (B, NH, 1, DHV)
    h_num = h_num.squeeze(2)  # (B, NH, DHV)

    qn_dotproduct = vecQ_scaled[:, :, None, :] @ vecN_state_new[:, :, :, None]  # (B, NH, 1, 1)
    qn_dotproduct = qn_dotproduct.squeeze(2)  # (B, NH, 1)
    max_val = torch.exp(-scaM_state_new)  # (B, NH, 1)
    h_denom = torch.maximum(qn_dotproduct.abs(), max_val) + eps  # (B, NH, 1)
    h = h_num / h_denom  # (B, NH, DHV) / (B, NH, 1) = (B, NH, DHV)

    return h, (matC_state_new, vecN_state_new, scaM_state_new)
