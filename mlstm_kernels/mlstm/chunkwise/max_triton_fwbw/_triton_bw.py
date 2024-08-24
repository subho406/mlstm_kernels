import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional
from einops import rearrange
from ....kernel_utils import contiguous_noctx, is_power_of_2, torch2triton_dtype

"""Triton.

Backward pass of the mLSTM chunkwise formulation.

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

@triton.jit
def _mlstm_chunkwise__recurrent_bw_dC_kernel():
    pass



@contiguous_noctx
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