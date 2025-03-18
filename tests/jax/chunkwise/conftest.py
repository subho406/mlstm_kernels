#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def mlstm_state_passing_test() -> callable:
    def _mlstm_state_passing_test(
        kernel_fn: callable,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        igate_preact: jax.Array,
        fgate_preact: jax.Array,
        num_chunks: int = 4,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> jax.Array:
        ctx_len = q.shape[2]
        input_arrays = (q, k, v, igate_preact, fgate_preact)
        h_full_solo = kernel_fn(*input_arrays, return_last_states=False)
        h_full_states, (c_full, n_full, m_full) = kernel_fn(
            *input_arrays, return_last_states=True
        )
        h_chunked = []
        c_chunked, n_chunked, m_chunked = None, None, None
        chunk_size = ctx_len // num_chunks
        for i in range(num_chunks):
            input_chunk = jax.tree.map(
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
        h_chunked = jnp.concatenate(h_chunked, axis=2)

        h_full_solo = jax.device_get(h_full_solo)
        h_full_states = jax.device_get(h_full_states)
        h_chunked = jax.device_get(h_chunked)

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

        c_full, n_full, m_full = jax.device_get((c_full, n_full, m_full))
        c_chunked, n_chunked, m_chunked = jax.device_get(
            (c_chunked, n_chunked, m_chunked)
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
