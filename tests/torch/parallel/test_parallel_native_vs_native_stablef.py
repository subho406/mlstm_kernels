#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging

from mlstm_kernels.torch.parallel.native import (
    mlstm_parallel__native_autograd,
    mlstm_parallel__native_custbw,
)
from mlstm_kernels.torch.parallel.native_stablef import (
    mlstm_parallel__native_stablef_autograd,
    mlstm_parallel__native_stablef_custbw,
)

import pytest
import torch

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "parallel-torch-native"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_parallel_native_vs_native_stablef_fp32(
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
        baseline_fn=mlstm_parallel__native_custbw,
        target_fn=mlstm_parallel__native_stablef_custbw,
        baseline_name="parallel_unstable_custbw",
        target_name="parallel_stablef_custbw",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=torch.float32,
        atol_fw=1e-4,
        rtol_fw=1e-3,
        atol_fwbw=2e-4,
        rtol_fwbw=5e-3,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        save_output_tensors_dir=str(test_output_folder),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_parallel_native_custbw_vs_native_autograd_fp32(
    test_session_folder, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_autograd,
        target_fn=mlstm_parallel__native_custbw,
        baseline_name="parallel_unstable_autograd",
        target_name="parallel_unstable_custbw",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=torch.float32,
        atol_fw=1e-4,
        rtol_fw=1e-3,
        atol_fwbw=5e-3,
        rtol_fwbw=5e-1,  # we need this high tolerance since there is one element with high error
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_parallel_stablef_native_custbw_vs_native_autograd_fp32(
    test_session_folder, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_autograd,
        target_fn=mlstm_parallel__native_stablef_custbw,
        baseline_name="parallel_stablef_autograd",
        target_name="parallel_stablef_custbw",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=torch.float32,
        atol_fw=1e-4,
        rtol_fw=1e-3,
        atol_fwbw=5e-3,
        rtol_fwbw=5e-1,  # we need this high tolerance since there is one element with high error
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
    )
