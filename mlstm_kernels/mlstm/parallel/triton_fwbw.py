import torch

from ._triton_bw import mlstm_bw
from ._triton_fw import mlstm_fw

from torch.amp import custom_fwd, custom_bwd

from ..utils import contiguous

def mlstm_fwbw(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matH, _, _ = _mlstm_fwbw.apply(matQ, matK, matV, vecI, vecF, eps)
    return matH


class _mlstm_fwbw(torch.autograd.Function):

    @staticmethod
    @custom_fwd(device_type="cuda")
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
        matH, vecM, vecN = mlstm_fw(
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
        vecDeltaM_unused: torch.Tensor,
        vecDeltaN_unused: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        (matQ, matK, matV, vecI, vecF, vecM, vecN) = ctx.saved_tensors
        matDeltaQ, matDeltaK, matDeltaV, vecDeltaI, vecDeltaF = mlstm_bw(
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
