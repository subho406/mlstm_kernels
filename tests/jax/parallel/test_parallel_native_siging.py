#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging

import jax
import jax.numpy as jnp
import pytest

from mlstm_kernels.jax.parallel.native_siging import (
    mlstm_siging_parallel__native_autograd,
    mlstm_siging_parallel__native_custbw,
)

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "parallel-jax-native-siging"


@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("stable_fgate", [True, False])
def test_parallel_native_siging_autograd_vs_native_custbw_fp32(
    test_session_folder,
    test_output_folder,
    mlstm_parallel_interface_test,
    S,
    B,
    NH,
    DHQK,
    DHHV,
    normalize,
    stable_fgate,
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_siging_parallel__native_autograd,
        target_fn=mlstm_siging_parallel__native_custbw,
        baseline_name=f"parallel_siging_stablef_{stable_fgate}_norm{normalize}_autograd",
        target_name=f"parallel_siging_stablef_{stable_fgate}_norm{normalize}_custbw",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=jnp.float32,
        atol_fw=1e-3,
        rtol_fw=0.01,
        atol_fwbw=0.2,  # vecFgrad as high errors: Max absolute difference: 0.27992487
        rtol_fwbw=0.1,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        run_backward=True,
        # save_output_tensors_dir=str(test_output_folder / "test_data"),
        use_jit=False,
    )
