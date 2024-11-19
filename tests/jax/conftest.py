import os
from pathlib import Path

import numpy as np
import pytest

from .template_test_parallel_interface import template_test_parallel_interface

# Select CPU or GPU devices.
if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
elif "JAX_PLATFORMS" not in os.environ:
    # Don't override JAX_PLATFORMS if it has been set.

    # If JAX_PLATFORMS has not been set, set it to auto-configure (i.e., to "").
    # Setting it this way will let jax automatically configure the platform, which
    # defaults to GPU when GPUs are available (which they should be when CUDA_VISIBLE_DEVICES
    # is set).
    os.environ["JAX_PLATFORMS"] = ""

# Select number of devices. On CPU, we simulate 8 devices.
if os.environ["JAX_PLATFORMS"] == "cpu":
    NUM_DEVICES = 8
    # The following line has to be imported here to avoid jax being initialized before
    # setting JAX_PLATFORMS to "cpu".
    from mlstm_kernels.jax.xla_utils import simulate_CPU_devices

    simulate_CPU_devices(NUM_DEVICES)
else:
    NUM_DEVICES = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    # Set XLA flags for deterministic operations on GPU. We do not use this flag for training runs
    # as it can slow down the training significantly.
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"


# Check if triton is available.
try:
    import jax_triton

    jt_version = jax_triton.__version__

    # If we run on GPU environments with jax triton installed, but with JAX_PLATFORMS
    # set to CPU, we need to disable the triton tests.
    TRITON_AVAILABLE = os.environ.get("JAX_PLATFORMS", "") != "cpu"
except ImportError:
    TRITON_AVAILABLE = False

import jax
import jax.numpy as jnp


@pytest.fixture
def mlstm_parallel_interface_test() -> callable:
    return template_test_parallel_interface


# Share environment variables with pytest.
def pytest_configure():
    pytest.triton_available = TRITON_AVAILABLE


@pytest.fixture
def torch_parallel_stablef_vs_unstablef_test_data(test_output_folder: Path) -> dict[str, np.ndarray]:
    torch_stablef_vs_unstablef_test_file = (
        test_output_folder
        / "parallel-torch-native_parallel_stablef_custbw-vs-parallel_unstable_custbw_S256B1NH2DHQK64DHHV128.npz"
    )

    assert (
        torch_stablef_vs_unstablef_test_file.exists()
    ), f"File {torch_stablef_vs_unstablef_test_file} does not exist. Please run pytorch tests first."

    data = np.load(torch_stablef_vs_unstablef_test_file)
    return data

@pytest.fixture
def default_qkvif() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    B = 1
    NH = 2
    S = 128
    DH = 64
    rng = np.random.default_rng(2)
    q = rng.normal(size=(B, NH, S, DH)).astype(np.float32)
    k = rng.normal(size=(B, NH, S, DH)).astype(np.float32)
    v = rng.normal(size=(B, NH, S, DH)).astype(np.float32)
    igate_preact = rng.normal(size=(B, NH, S)).astype(np.float32)
    fgate_preact = rng.normal(size=(B, NH, S)).astype(np.float32) + 4.5

    q = jnp.array(q)
    k = jnp.array(k)
    v = jnp.array(v)
    igate_preact = jnp.array(igate_preact)
    fgate_preact = jnp.array(fgate_preact)

    return q, k, v, igate_preact, fgate_preact
