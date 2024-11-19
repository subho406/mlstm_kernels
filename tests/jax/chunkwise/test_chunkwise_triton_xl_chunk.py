import logging

from mlstm_kernels.jax.chunkwise.triton_xl_chunk import mlstm_chunkwise__xl_chunk
from mlstm_kernels.jax.parallel.native_stablef import mlstm_parallel__native_stablef_autograd

import jax
import jax.numpy as jnp
import pytest

from ...conftest import final_combinations
from ..template_test_against_pytorch import check_jax_against_pytorch_reference

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "chunkwise-jax_xl_chunk"


@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_jax_native_chunkwise_vs_triton_xl_chunk_fp32(
    test_session_folder, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_autograd,
        target_fn=mlstm_chunkwise__xl_chunk,
        baseline_name="native_parallel_stablef_autograd",
        target_name="triton_xl_chunk",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=jnp.float32,
        atol_fw=3e-3,
        rtol_fw=5e-2,
        atol_fwbw=2e-1, # we need those high tolerances for the forget gate gradient Max absolute difference: 0.2168259
        rtol_fwbw=1e-2,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        use_jit=True,
    )


def test_vs_torch_xl_chunk(test_output_folder):
    test_data_file = (
        test_output_folder
        / "chunkwise-triton_xl_chunk_triton_chunkwise_xl_chunk-vs-native_parallel_stablef_custbw_S256B1NH2DHQK64DHHV128.npz"
    )

    check_jax_against_pytorch_reference(
        torch_test_data_file=test_data_file,
        jax_mlstm_parallel_fn=mlstm_chunkwise__xl_chunk,
        atol_fw=1e-4,
        rtol_fw=1e-2,
        atol_fwbw=2e-2,
        rtol_fwbw=5e-2,
    )


@pytest.mark.skipif(not pytest.triton_available, reason="Triton is not available.")
@pytest.mark.parametrize("mlstm_kernel", [mlstm_chunkwise__xl_chunk])
def test_mlstm_chunkwise_state_passing(
    default_qkvif: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    mlstm_state_passing_test: callable,
    mlstm_kernel: callable,
):
    """Compare single forward vs chunked one with states passed between steps."""
    # Repeat the inputs to have longer sequence length.
    default_qkvif = jax.tree.map(lambda x: jnp.repeat(x, 2, axis=2), default_qkvif)
    mlstm_state_passing_test(mlstm_kernel, *default_qkvif, num_chunks=4, rtol=5e-2, atol=5e-3)
