#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import numpy as np
import pytest
import torch


@pytest.fixture
def state_passing_qkvif() -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    torch.manual_seed(42)
    B, NH, S, DHQK, DHHV = 2, 4, 256, 16, 32
    q = torch.randn(B, NH, S, DHQK)
    k = torch.randn(B, NH, S, DHQK)
    v = torch.randn(B, NH, S, DHHV)
    igate_preact = torch.randn(B, NH, S)
    fgate_preact = torch.randn(B, NH, S)
    return q, k, v, igate_preact, fgate_preact


@pytest.fixture
def mlstm_state_passing_test() -> callable:
    def _mlstm_state_passing_test(
        kernel_fn: callable,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        num_chunks: int = 4,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        device: str = "cuda",
    ) -> torch.Tensor:
        ctx_len = q.shape[2]
        B, NH, S, DHQK = q.shape
        DHHV = v.shape[-1]

        input_arrays = (q, k, v, igate_preact, fgate_preact)
        input_arrays = tuple(map(lambda x: x.to(device), input_arrays))

        h_full_solo = kernel_fn(*input_arrays, return_last_states=False)
        h_full_states, (c_full, n_full, m_full) = kernel_fn(
            *input_arrays, return_last_states=True
        )

        assert h_full_states.shape == (
            B,
            NH,
            S,
            DHHV,
        ), f"Expected shape {(B, NH, S, DHHV)}, got {h_full_states.shape}."
        assert c_full.shape == (
            B,
            NH,
            DHQK,
            DHHV,
        ), f"Expected shape {(B, NH, DHQK, DHHV)}, got {c_full.shape}."
        assert n_full.shape == (
            B,
            NH,
            DHQK,
        ), f"Expected shape {(B, NH, DHQK)}, got {n_full.shape}."
        assert m_full.shape == (
            B,
            NH,
            1,
        ), f"Expected shape {(B, NH, 1)}, got {m_full.shape}."

        h_chunked = []
        c_chunked, n_chunked, m_chunked = None, None, None
        chunk_size = ctx_len // num_chunks
        for i in range(num_chunks):
            input_chunk = map(
                lambda x: x[:, :, i * chunk_size : (i + 1) * chunk_size], input_arrays
            )
            h_chunked_i, (c_chunked, n_chunked, m_chunked) = kernel_fn(
                *input_chunk,
                c_initial=c_chunked,
                n_initial=n_chunked,
                m_initial=m_chunked,
                return_last_states=True,
            )
            h_chunked.append(h_chunked_i)
            assert h_chunked_i.shape == (
                B,
                NH,
                chunk_size,
                DHHV,
            ), f"Expected shape {(B, NH, chunk_size, DHHV)}, got {h_chunked_i.shape}."
            assert c_chunked.shape == (
                B,
                NH,
                DHQK,
                DHHV,
            ), f"Expected shape {(B, NH, DHQK, DHHV)}, got {c_chunked.shape}."
            assert n_chunked.shape == (
                B,
                NH,
                DHQK,
            ), f"Expected shape {(B, NH, DHQK)}, got {n_chunked.shape}."
            assert m_chunked.shape == (
                B,
                NH,
                1,
            ), f"Expected shape {(B, NH, 1)}, got {m_chunked.shape}."

        h_chunked = torch.concatenate(h_chunked, axis=2)

        h_full_solo = h_full_solo.cpu().detach().numpy()
        h_full_states = h_full_states.cpu().detach().numpy()
        h_chunked = h_chunked.cpu().detach().numpy()

        np.testing.assert_allclose(
            h_full_solo,
            h_full_states,
            rtol=rtol,
            atol=atol,
            err_msg="H state with return_last_states=False vs True do not match.",
        )
        np.testing.assert_allclose(
            h_full_states[:, :, :chunk_size],
            h_chunked[:, :, :chunk_size],
            rtol=rtol,
            atol=atol,
            err_msg="H state with single forward vs chunked do not match in the first chunk, ie without state passing.",
        )
        np.testing.assert_allclose(
            h_full_states[:, :, chunk_size:],
            h_chunked[:, :, chunk_size:],
            rtol=rtol,
            atol=atol,
            err_msg="H state with single forward vs chunked do not match after the first chunk, ie with state passing.",
        )

        c_full, n_full, m_full = map(
            lambda x: x.cpu().detach().numpy(), (c_full, n_full, m_full)
        )
        c_chunked, n_chunked, m_chunked = map(
            lambda x: x.cpu().detach().numpy(), (c_chunked, n_chunked, m_chunked)
        )

        np.testing.assert_allclose(
            c_full,
            c_chunked,
            rtol=rtol,
            atol=atol,
            err_msg="C state with single forward vs chunked do not match.",
        )
        np.testing.assert_allclose(
            n_full,
            n_chunked,
            rtol=rtol,
            atol=atol,
            err_msg="N state with single forward vs chunked do not match.",
        )
        np.testing.assert_allclose(
            m_full,
            m_chunked,
            rtol=rtol,
            atol=atol,
            err_msg="M state with single forward vs chunked do not match.",
        )

    return _mlstm_state_passing_test
