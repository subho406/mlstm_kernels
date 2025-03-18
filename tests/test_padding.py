#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import sys
import unittest

import torch

print(sys.path)

from mlstm_kernels import get_kernel, get_whole_registry


class TestPadding(unittest.TestCase):
    def test_padding(self):
        B, N, S, H = 1, 1, 63, 128
        dtype = torch.bfloat16
        device = torch.device("cuda")
        q, k, v = (
            torch.zeros([B, N, S, H], device=device, dtype=dtype),
            torch.zeros([B, N, S, H], device=device, dtype=dtype),
            torch.zeros([B, N, S, H], device=device, dtype=dtype),
        )
        i, f = (
            torch.zeros([B, N, S], device=device, dtype=dtype),
            torch.zeros([B, N, S], device=device, dtype=dtype),
        )
        kernel = get_kernel("mlstm_chunkwise--triton", padded_chunk_size=64)
        h = kernel(q, k, v, i, f)
        assert h.shape == v.shape

        B, N, S, H = 1, 1, 128, 128
        dtype = torch.bfloat16
        device = torch.device("cuda")
        q, k, v = (
            torch.zeros([B, N, S, H], device=device, dtype=dtype),
            torch.zeros([B, N, S, H], device=device, dtype=dtype),
            torch.zeros([B, N, S, H], device=device, dtype=dtype),
        )
        i, f = (
            torch.zeros([B, N, S], device=device, dtype=dtype),
            torch.zeros([B, N, S], device=device, dtype=dtype),
        )
        kernel = get_kernel("mlstm_chunkwise--triton", padded_chunk_size=64)
        h = kernel(q, k, v, i, f)
        assert h.shape == v.shape

    def test_padding_whole_registry(self):
        B, N, S, H = 1, 1, 63, 128
        dtype = torch.bfloat16
        device = torch.device("cuda")
        q, k, v = (
            torch.zeros([B, N, S, H], device=device, dtype=dtype),
            torch.zeros([B, N, S, H], device=device, dtype=dtype),
            torch.zeros([B, N, S, H], device=device, dtype=dtype),
        )
        i, f = (
            torch.zeros([B, N, S], device=device, dtype=dtype),
            torch.zeros([B, N, S], device=device, dtype=dtype),
        )
        kernel = get_whole_registry(padded_chunk_size=64)["mlstm_chunkwise--triton"]
        h = kernel(q, k, v, i, f)
        assert h.shape == v.shape

        B, N, S, H = 1, 1, 128, 128
        dtype = torch.bfloat16
        device = torch.device("cuda")
        q, k, v = (
            torch.zeros([B, N, S, H], device=device, dtype=dtype),
            torch.zeros([B, N, S, H], device=device, dtype=dtype),
            torch.zeros([B, N, S, H], device=device, dtype=dtype),
        )
        i, f = (
            torch.zeros([B, N, S], device=device, dtype=dtype),
            torch.zeros([B, N, S], device=device, dtype=dtype),
        )
        kernel = get_whole_registry(padded_chunk_size=64)["mlstm_chunkwise--triton"]
        h = kernel(q, k, v, i, f)
        assert h.shape == v.shape


# print("Hello")
