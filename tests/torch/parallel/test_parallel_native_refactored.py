import logging

from mlstm_kernels.torch.parallel.native import mlstm_parallel__native_custbw
from mlstm_kernels.torch.parallel.native_stablef import mlstm_parallel__native_stablef_custbw
from mlstm_kernels.mlstm.parallel.stable_torch_fwbw import (
    mlstm_parallel_torch_ownbw as mlstm_parallel_stable_torch_ownbw,
)
from mlstm_kernels.mlstm.parallel.torch_fwbw import mlstm_parallel_torch_ownbw

import pytest
import pytest
import torch

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "parallel-torch-native"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_parallel_stable_refactored_vs_previous_fp32(
    test_session_folder, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel_stable_torch_ownbw,
        target_fn=mlstm_parallel__native_stablef_custbw,
        baseline_name="old_parallel_stable_obw",
        target_name="new_parallel_stable_obw",
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
        vmax=1e-4,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_parallel_unstable_refactored_vs_previous_fp32(
    test_session_folder, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel_torch_ownbw,
        target_fn=mlstm_parallel__native_custbw,
        baseline_name="old_parallel_unstable_obw",
        target_name="new_parallel_unstable_obw",
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
        vmax=1e-4,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
    )
