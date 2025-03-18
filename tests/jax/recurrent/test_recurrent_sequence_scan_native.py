#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
from pathlib import Path

from mlstm_kernels.jax.parallel.native_stablef import (
    mlstm_parallel__native_stablef_autograd,
)
from mlstm_kernels.jax.recurrent.native_sequence_scan import (
    mlstm_recurrent_sequence__native_fw,
    mlstm_recurrent_sequence__triton_step_fused_fw,
)

import jax
import jax.numpy as jnp
import pytest

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "recurrent_sequence-jax__native"


@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_native_recurrent_sequence_native_step_vs_native_parallel_stablef_fp32(
    test_session_folder: Path, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_autograd,
        target_fn=mlstm_recurrent_sequence__native_fw,
        baseline_name="native_parallel_stablef_autograd",
        target_name="native_recurrent_sequence_native_step",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=jnp.float32,
        atol_fw=1.1e-2,  # Max absolute difference: 0.01150007
        rtol_fw=5e-2,
        atol_fwbw=2e-2,
        rtol_fwbw=5e-2,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        use_jit=False,
        run_backward=False,
    )


@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_native_recurrent_sequence_triton_step_fused_vs_native_parallel_stablef_fp32(
    test_session_folder: Path, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_autograd,
        target_fn=mlstm_recurrent_sequence__triton_step_fused_fw,
        baseline_name="native_parallel_stablef_autograd",
        target_name="native_recurrent_sequence_triton_step_fused",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=jnp.float32,
        atol_fw=1.1e-2,  # Max absolute difference: 0.0114983
        rtol_fw=5e-2,
        atol_fwbw=2e-2,
        rtol_fwbw=5e-2,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        use_jit=False,
        run_backward=False,
    )
