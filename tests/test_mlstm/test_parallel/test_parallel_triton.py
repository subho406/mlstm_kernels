import logging

from mlstm_kernels.mlstm.parallel import (
    mlstm_parallel_stable_torch_autograd,
    mlstm_parallel_torch_autograd,
    mlstm_parallel_triton,
)
from mlstm_kernels.test_utils.test_fixtures import test_session_folder  # noqa
from mlstm_kernels.test_utils.test_templates.template_parallel_interface import template_test_parallel_interface

import pytest
import torch

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "parallel-triton"

# The parallel kernel currently does not support the different head dimensions for qk and v.
parallel_combinations = {
    "S": [256],  # [8192],
    "B": [1],  # [2, 2, 2, 2],
    "NH": [2],  # [3, 3, 3, 3],
    "DHQK": [128],  # [5, 5, 5, 5],
    "DHHV": [128],  # [5, 5, 5, 5],
}
parallel_combinations = [values for values in zip(*parallel_combinations.values())]


class TestParallelStableTorchVsParallelTorchAutograd:
    """Test torch implementations forwards with backward computed by autograd."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], parallel_combinations)
    def test_torch_parallel_unstable_vs_parallel_triton_fp32(self, test_session_folder, S, B, NH, DHQK, DHHV):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_test_parallel_interface(
            baseline_fn=mlstm_parallel_torch_autograd,
            target_fn=mlstm_parallel_triton,
            baseline_name="unstable_ag",
            target_name="triton",
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            dtype=torch.float32,
            atol_fw=1e-1,
            rtol_fw=1e-1,
            atol_fwbw=0.5,
            rtol_fwbw=0.5,
            test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
            save_dir=str(test_session_folder),
            add_fp64_baseline=False,
        )
