import torch
from torch.amp import custom_fwd, custom_bwd
from ....kernel_utils import contiguous

"""Triton.

Forward and backward pass of the mLSTM chunkwise formulation.

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
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
