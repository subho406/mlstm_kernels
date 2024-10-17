import logging

import pytest
import torch

from mlstm_kernels.mlstm.chunkwise import mlstm_chunkwise_max_triton_v3, mlstm_chunkwise_max_triton_v3noslice
from mlstm_kernels.test_utils.test_fixtures import test_session_folder  # noqa
from mlstm_kernels.test_utils.test_templates.template_parallel_interface import template_test_parallel_interface

from ..test_params import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "chunkwise-triton"


class TestChunkwiseV3vsV3ReducedSlicing:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
    def test_chunkwise_torch_vs_stable_torch_fp32(self, test_session_folder, S, B, NH, DHQK, DHHV):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_test_parallel_interface(
            baseline_fn=mlstm_chunkwise_max_triton_v3,
            target_fn=mlstm_chunkwise_max_triton_v3noslice,
            baseline_name="max_triton_v3",
            target_name="max_triton_v3noslice",
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            dtype=torch.float32,
            atol_fw=1e-5,
            rtol_fw=1e-5,
            atol_fwbw=1e-5,
            rtol_fwbw=1e-5,
            test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
            save_dir=str(test_session_folder),
            add_fp64_baseline=False,
        )
