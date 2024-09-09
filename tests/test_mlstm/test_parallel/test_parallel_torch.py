import pytest
import logging
import torch

from mlstm_kernels.test_utils.test_templates.template_parallel_interface import (
    template_test_parallel_interface,
)
from mlstm_kernels.test_utils.test_fixtures import test_session_folder  # noqa

from mlstm_kernels.mlstm.parallel import (
    mlstm_parallel_torch_autograd,
    mlstm_parallel_stable_torch_autograd,
)

from ..test_params import final_combinations

LOGGER = logging.getLogger(__name__)


class TestParallelStableTorchVsParallelTorchLong:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.xfail(reason="Fails due to numerical instability")
    @pytest.mark.parametrize(
        ["S", "B", "NH", "DHQK", "DHHV", "target_dtype"], final_combinations
    )
    def test_torch_parallel_stable_vs_unstable(
        self, test_session_folder, S, B, NH, DHQK, DHHV, target_dtype
    ):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_test_parallel_interface(
            baseline_fn=mlstm_parallel_stable_torch_autograd,
            target_fn=mlstm_parallel_torch_autograd,
            baseline_name="stable_ag",
            target_name="unstable_ag",
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            dtype=getattr(torch, target_dtype),
            atol_fw=1.0,  # 3.0
            rtol_fw=1.0,
            atol_fwbw=1.5,  # 3.5
            rtol_fwbw=1.0,
            vmax=1.0,
            test_folder_name_prefix="parallel-torch",
            save_dir=str(test_session_folder),
            add_fp64_baseline=True,
        )
