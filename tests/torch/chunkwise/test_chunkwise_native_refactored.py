import logging

from mlstm_kernels.mlstm.chunkwise.torch_fwbw.torch_fwbw import mlstm_chunkwise_torch_ownbw
from mlstm_kernels.torch.chunkwise.native import mlstm_chunkwise__native_custbw

import pytest
import torch

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "parallel-torch-native"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_chunkwise_refactored_vs_previous_fp32(
    test_session_folder, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_chunkwise_torch_ownbw,
        target_fn=mlstm_chunkwise__native_custbw,
        baseline_name="old_chunkwise_obw",
        target_name="new_chunkwise_obw",
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
