"""
PyTorch

mLSTM forward and backward pass. Parallel formulation.
"""

# Copyright JKU Linz 2024
# Author: Maximilian Beck
import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.amp import custom_bwd, custom_fwd

from ...kernel_utils import contiguous


def _mlstm_fw(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    _dtype, _device = matQ.dtype, matQ.device

    vecLogSigF = F.logsigmoid(vecF)  # (B, NH, S)
    vecLogSigF_cumsum = vecLogSigF.cumsum(-1)

    matLogSigF = vecLogSigF_cumsum[:, :, :, None] - vecLogSigF_cumsum[:, :, None, :]

    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )

    matLogSigF_mask = torch.where(ltr, matLogSigF, -float("inf"))

    matLogD = matLogSigF_mask + vecI[:, :, None, :]

    vecM, _ = torch.max(matLogD, dim=-1, keepdim=True)  # (B, NH, S, 1)
    matLogD_stabilized = matLogD - vecM

    matD = torch.exp(matLogD_stabilized)  # (B, NH, S, S)

    matS = (matQ @ matK.transpose(-2, -1)) / math.sqrt(DHQK)  # (B, NH, S, S)

    matCtilde = matS * matD  # (B, NH, S, S)
    vecN = torch.maximum(
        matCtilde.sum(dim=-1, keepdim=True).abs(), torch.exp(-vecM)
    )  # (B, NH, S, 1)
    # (B, NH, S, S)
    matC = matCtilde / (vecN + eps)

    matH = matC @ matV  # (B, NH, S, DH)

    vecN = vecN.squeeze(-1)
    vecM = vecM.squeeze(-1)

    return (matH, vecN, vecM)


def _mlstm_bw(
    matDeltaHtilde: torch.Tensor,
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    vecM: torch.Tensor,
    vecN: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, ...]:
    B, NH, S, DHQK = matQ.shape
    assert matK.shape == (B, NH, S, DHQK)
    assert vecI.shape == (B, NH, S)
    assert vecF.shape == (B, NH, S)

    _dtype, _device = matQ.dtype, matQ.device

    vecLogSigF = F.logsigmoid(vecF)  # (B, NH, S)
    vecLogSigF_cumsum = vecLogSigF.cumsum(-1)

    matLogSigF = vecLogSigF_cumsum[:, :, :, None] - vecLogSigF_cumsum[:, :, None, :]

    ltr = torch.tril(
        torch.ones(
            (S, S),
            dtype=torch.bool,
            device=_device,
        )
    )

    matLogSigF_mask = torch.where(ltr, matLogSigF, -float("inf"))

    matLogD = matLogSigF_mask + vecI[:, :, None, :]

    matLogD_stabilized = matLogD - vecM[:, :, :, None]

    matD = torch.exp(matLogD_stabilized)  # (B, NH, S, S)

    # intermediate delta-errors
    matDeltaC = matDeltaHtilde @ matV.transpose(-2, -1) / (vecN[:, :, :, None] + eps)

    matS = (matQ @ matK.transpose(-2, -1)) / math.sqrt(DHQK)

    matDeltaDtilde = matDeltaC * matD * matS

    vecDeltaI = torch.sum(matDeltaDtilde, dim=-2)

    # output delta-errors / gradients
    matP = matDeltaC * matD

    matDeltaQ = (matP @ matK) / math.sqrt(DHQK)
    matDeltaK = (matP.transpose(-2, -1) @ matQ) / math.sqrt(DHQK)

    matCtilde = matS * matD
    matDeltaV = matCtilde.transpose(-2, -1) @ (
        matDeltaHtilde / (vecN[:, :, :, None] + eps)
    )

    # EFFICIENT LINEAR ATTENTION TRICK
    # compute the vecDeltaFbar values with dfbar = rev_cumsum((q*dq - k*dk).sum(-1))
    vecDeltaFbar_acc = (matQ * matDeltaQ - matK * matDeltaK).sum(-1)
    vecDeltaFbar = vecDeltaFbar_acc.flip(-1).cumsum(-1).flip(-1)
    vecDeltaF = vecDeltaFbar * torch.sigmoid(-vecF)

    return (
        matDeltaQ,
        matDeltaK,
        matDeltaV,
        vecDeltaI,
        vecDeltaF,
    )


def _mlstm_parallel_fwbw_generator(autocast_kernel_dtype=torch.float32) -> Callable:
    class _mlstm_parallel_fwbw(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cuda", cast_inputs=autocast_kernel_dtype)
        @contiguous
        def forward(
            ctx,
            matQ: torch.Tensor,
            matK: torch.Tensor,
            matV: torch.Tensor,
            vecI: torch.Tensor,
            vecF: torch.Tensor,
            eps: float = 1e-6,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            matH, vecN, vecM = _mlstm_fw(
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                eps=eps,
            )
            ctx.save_for_backward(matQ, matK, matV, vecI, vecF, vecM, vecN)
            return matH, vecM, vecN

        @staticmethod
        @custom_bwd(device_type="cuda")
        @contiguous
        def backward(
            ctx,
            matDeltaHtilde: torch.Tensor,
            vecDeltaN_unused: torch.Tensor,
            vecDeltaM_unused: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            (matQ, matK, matV, vecI, vecF, vecM, vecN) = ctx.saved_tensors
            matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF = _mlstm_bw(
                matDeltaHtilde=matDeltaHtilde,
                matQ=matQ,
                matK=matK,
                matV=matV,
                vecI=vecI,
                vecF=vecF,
                vecM=vecM,
                vecN=vecN,
            )
            return matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF, None

    return _mlstm_parallel_fwbw


_mlstm_parallel_fwbw_float32 = _mlstm_parallel_fwbw_generator(
    autocast_kernel_dtype=torch.float32
)
_mlstm_parallel_fwbw_float16 = _mlstm_parallel_fwbw_generator(
    autocast_kernel_dtype=torch.float16
)
_mlstm_parallel_fwbw_bfloat16 = _mlstm_parallel_fwbw_generator(
    autocast_kernel_dtype=torch.bfloat16
)


def _get_parallel_fwbw_kernel(autocast_kernel_dtype: torch.dtype) -> Callable:
    if autocast_kernel_dtype == torch.float32:
        return _mlstm_parallel_fwbw_float32
    elif autocast_kernel_dtype == torch.float16:
        return _mlstm_parallel_fwbw_float16
    elif autocast_kernel_dtype == torch.bfloat16:
        return _mlstm_parallel_fwbw_bfloat16
    else:
        raise ValueError(f"Unsupported autocast_kernel_dtype: {autocast_kernel_dtype}")


def mlstm_parallel_torch_autograd(
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
) -> torch.Tensor:
    assert c_initial is None, "c_initial is not supported"
    assert n_initial is None, "n_initial is not supported"
    assert m_initial is None, "m_initial is not supported"
    assert not return_last_states, "return_last_states is not supported"

    matH, _, _ = _mlstm_fw(
        matQ=q,
        matK=k,
        matV=v,
        vecI=i,
        vecF=f,
        eps=eps,
    )
    return matH


def mlstm_parallel_torch_ownbw(
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
    autocast_kernel_dtype: torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    assert c_initial is None, "c_initial is not supported"
    assert n_initial is None, "n_initial is not supported"
    assert m_initial is None, "m_initial is not supported"
    assert return_last_states is False, "return_last_states is not supported"

    _mlstm_parallel_fwbw = _get_parallel_fwbw_kernel(
        autocast_kernel_dtype=autocast_kernel_dtype
    )

    matH, _, _ = _mlstm_parallel_fwbw.apply(q, k, v, i, f, eps)
    return matH
