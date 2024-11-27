import logging

import jax
import jax.numpy as jnp
import pytest

from mlstm_kernels.jax.parallel.native_stablef import (
    mlstm_parallel__native_stablef_autograd,
    mlstm_parallel__native_stablef_custbw,
)

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "parallel-jax-native"


@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_parallel_stablef_native_autograd_vs_native_custbw_fp32(
    test_session_folder,
    test_output_folder,
    mlstm_parallel_interface_test,
    S,
    B,
    NH,
    DHQK,
    DHHV,
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_autograd,
        target_fn=mlstm_parallel__native_stablef_custbw,
        baseline_name="parallel_stable_autograd",
        target_name="parallel_sable_custbw",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=jnp.float32,
        atol_fw=1e-3,
        rtol_fw=1e-2,
        atol_fwbw=0.15,  # 3.2e-2,  # matQgrad as high errors: Max absolute difference: 0.33696747
        rtol_fwbw=0.1,  # 9e-2,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        run_backward=True,
        # save_output_tensors_dir=str(test_output_folder / "test_data"),
        use_jit=False,
    )
