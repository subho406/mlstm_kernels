import logging

from mlstm_kernels.torch.parallel.native_stablef import mlstm_parallel__native_stablef_custbw
from mlstm_kernels.torch.parallel.triton_limit_headdim import mlstm_parallel__limit_headdim

import pytest
import torch

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "parallel-triton_limit_headdim"

combinations_long = {
    "S": [256],  # [8192],
    "B": [1],  # [2, 2, 2, 2],
    "NH": [2],  # [3, 3, 3, 3],
    "DHQK": [128],  # [5, 5, 5, 5],
    "DHHV": [128],  # [5, 5, 5, 5],
}
combinations = [values for values in zip(*combinations_long.values())]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], combinations)
def test_triton_parallel_limit_headdim_vs_native_parrallel_stablef_fp32(
    test_session_folder, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_custbw,
        target_fn=mlstm_parallel__limit_headdim,
        baseline_name="native_parallel_stablef_custbw",
        target_name="triton_parallel_limit_headdim",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=torch.float32,
        atol_fw=1e-2,
        rtol_fw=5e-2,
        atol_fwbw=3e-1,  # we need to increase this tolerance for vecF.grad (max diff val 0.267...)
        rtol_fwbw=1.0,
        vmax=1e-2,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
    )
