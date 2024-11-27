from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
import torch

from mlstm_kernels.torch.chunkwise import (
    mlstm_chunkwise__limit_chunk,
    mlstm_chunkwise__xl_chunk,
)
from mlstm_kernels.torch.kernel_wrappers import wrap_chunkwise__pad_zeros
from mlstm_kernels.torch.parallel.native_stablef import (
    mlstm_parallel__native_stablef_autograd,
)


def template_test_pad_zeros(
    B: int,
    NH: int,
    S: int,
    DHQK: int,
    DHHV: int,
    chunk_size: int,
    chunkwise_kernel: Callable,
    parallel_baseline: Callable,
    device: str,
    dtype: str,
    atol: float,
    rtol: float,
    eps: float,
):
    torch.manual_seed(42)
    q = torch.randn(B, NH, S, DHQK)
    k = torch.randn(B, NH, S, DHQK)
    v = torch.randn(B, NH, S, DHHV)
    i = torch.randn(B, NH, S)
    f = 3.0 + torch.randn(B, NH, S)

    dtype = getattr(torch, dtype)
    device = torch.device(device)
    (q, k, v, i, f) = tuple(
        map(lambda x: x.to(dtype=dtype, device=device), (q, k, v, i, f))
    )

    # run the parallel baseline
    h_parallel_bl = parallel_baseline(q, k, v, i, f, eps=eps)

    # create the padded kernel function
    padded_chunkwise_kernel = partial(
        wrap_chunkwise__pad_zeros, mlstm_chunkwise_kernel=chunkwise_kernel
    )

    # run the padded kernel
    h_padded = padded_chunkwise_kernel(
        q=q,
        k=k,
        v=v,
        i=i,
        f=f,
        eps=eps,
        return_last_states=False,
        chunk_size=chunk_size,
    )

    # compare the results
    h_parallel_bl = h_parallel_bl.detach().cpu().to(dtype=torch.float64).numpy()
    h_padded = h_padded.detach().cpu().to(dtype=torch.float64).numpy()

    np.testing.assert_allclose(h_padded, h_parallel_bl, atol=atol, rtol=rtol)


@pytest.mark.parametrize("S, chunk_size", [[128, 32], [65, 64], [1, 16]])
@pytest.mark.parametrize("chunkwise_kernel", [mlstm_chunkwise__limit_chunk])
@pytest.mark.parametrize("parallel_baseline", [mlstm_parallel__native_stablef_autograd])
def test_pad_zeros__limit_chunk_kernels(
    S: int, chunk_size: int, chunkwise_kernel: Callable, parallel_baseline: Callable
):
    template_test_pad_zeros(
        B=1,
        NH=1,
        S=S,
        DHQK=16,
        DHHV=32,
        chunk_size=chunk_size,
        chunkwise_kernel=chunkwise_kernel,
        parallel_baseline=parallel_baseline,
        device="cuda",
        dtype="float32",
        atol=1e-2,
        rtol=1e-2,
        eps=1e-6,
    )


@pytest.mark.parametrize("S, chunk_size", [[128, 32], [65, 64], [1, 16], [400, 128]])
@pytest.mark.parametrize("chunkwise_kernel", [mlstm_chunkwise__xl_chunk])
@pytest.mark.parametrize("parallel_baseline", [mlstm_parallel__native_stablef_autograd])
def test_pad_zeros__xl_chunk_kernels(
    S: int, chunk_size: int, chunkwise_kernel: Callable, parallel_baseline: Callable
):
    template_test_pad_zeros(
        B=1,
        NH=1,
        S=S,
        DHQK=16,
        DHHV=32,
        chunk_size=chunk_size,
        chunkwise_kernel=chunkwise_kernel,
        parallel_baseline=parallel_baseline,
        device="cuda",
        dtype="float32",
        atol=1e-2,
        rtol=1e-2,
        eps=1e-6,
    )
