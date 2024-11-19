from mlstm_kernels.jax.parallel.native import mlstm_parallel__native_autograd
from mlstm_kernels.jax.parallel.native_stablef import mlstm_parallel__native_stablef_autograd

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_mlstm_parallel_jax_vs_torch(
    torch_parallel_stablef_vs_unstablef_test_data
):
    test_data = torch_parallel_stablef_vs_unstablef_test_data
    matQ = jnp.array(test_data["matQ"])
    matK = jnp.array(test_data["matK"])
    matV = jnp.array(test_data["matV"])
    vecI = jnp.array(test_data["vecI"])
    vecF = jnp.array(test_data["vecF"])
    print(torch_parallel_stablef_vs_unstablef_test_data.keys())
    matH_torch_unstable = test_data["matH_target"]
    matH_torch_stable = test_data["matH_baseline"]

    matH_jax_unstable = mlstm_parallel__native_autograd(matQ, matK, matV, vecI, vecF)
    matH_jax_unstable = jax.device_get(matH_jax_unstable)

    np.testing.assert_allclose(matH_torch_unstable, matH_jax_unstable, atol=3e-3, rtol=6e-2)

    matH_jax_stable = mlstm_parallel__native_stablef_autograd(matQ, matK, matV, vecI, vecF)
    matH_jax_stable = jax.device_get(matH_jax_stable)

    np.testing.assert_allclose(matH_torch_stable, matH_jax_stable, atol=3e-3, rtol=6e-2)