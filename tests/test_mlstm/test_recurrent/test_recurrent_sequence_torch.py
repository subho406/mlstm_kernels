import logging

from mlstm_kernels.mlstm.parallel import mlstm_parallel_stable_torch_autograd, mlstm_parallel_torch_autograd
from mlstm_kernels.mlstm.recurrent import mlstm_recurrent_sequence_torch_autograd
from mlstm_kernels.test_utils.test_fixtures import test_session_folder  # noqa
from mlstm_kernels.test_utils.test_templates.template_parallel_interface import template_test_parallel_interface

import pytest
import torch

from ..test_params import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "recurrent_seq-torch"


class TestRecurrentSequenceTorchVsParallelTorchAutograd:
    """Test torch implementations forwards with backward computed by autograd."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
    def test_torch_parallel_stable_vs_unstable_fp32(self, test_session_folder, S, B, NH, DHQK, DHHV):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_test_parallel_interface(
            baseline_fn=mlstm_parallel_stable_torch_autograd,
            target_fn=mlstm_recurrent_sequence_torch_autograd,
            baseline_name="torch_parallel_stable_ag",
            target_name="torch_recurrent_sequence",
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            dtype=torch.float32,
            atol_fw=1e-4,
            rtol_fw=1e-4,
            atol_fwbw=1e-4,
            rtol_fwbw=1e-2,
            vmax=1e-4,
            test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
            save_dir=str(test_session_folder),
            add_fp64_baseline=False,
        )
