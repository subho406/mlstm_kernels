# Author: Maximilian Beck
"""In this file we compute the chunkwise or cumulative gates (i.e. vecA and vecB)
for the forward and backward pass of the mLSTM.
We use the stable formulations, i.e. we avoid subtraction of forget gates.
"""

import torch
from einops import rearrange
from torch.nn.functional import logsigmoid

# TODO write a triton kernel for this


def compute_chunkwise_log_gates_vecB_vecA(
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    chunk_size: int,
    return_vecB_only: bool = False,
):
    B, NH, S = vecI.shape
    assert S % chunk_size == 0, f"S={S} is not divisible by chunk_size={chunk_size}"
    _device = vecI.device
    NC = S // chunk_size
    L = chunk_size

    # compute vecB
    vecF_logsig = logsigmoid(vecF.to(dtype=torch.float32))
    vecF_logsig_chunked = rearrange(vecF_logsig, "b nh (nc l) -> b nh nc l", nc=NC, l=L)
    vecB = vecF_logsig_chunked.cumsum(dim=-1)

    if return_vecB_only:
        return vecB
    else:
        # compute vecA
        vecI_chunked = rearrange(vecI, "b nh (nc l) -> b nh nc l", nc=NC, l=L)
        # unstable vecA computation:
        # vecA = (vecB[..., -1, None] - vecB) + vecI  # (B, NH, NC, L)
        # stable vecA computation:
        vecA = (
            torch.cat(
                [
                    vecF_logsig_chunked[..., 1:].flip(-1).cumsum(-1).flip(-1),
                    torch.zeros((B, NH, NC, 1), device=_device, dtype=torch.float32),
                ],
                dim=-1,
            )
            + vecI_chunked
        )  # (B, NH, NC, L)
        return vecB, vecA
