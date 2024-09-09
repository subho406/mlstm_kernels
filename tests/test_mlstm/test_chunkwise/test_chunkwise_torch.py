import pytest
import logging
import torch

from mlstm_kernels.test_utils.test_templates.template_parallel_interface import (
    template_test_parallel_interface,
)
from mlstm_kernels.test_utils.test_fixtures import test_session_folder  # noqa

from mlstm_kernels.mlstm.parallel import (
    mlstm_parallel_stable_torch_autograd,
    mlstm_parallel_torch_autograd,
)

from mlstm_kernels.mlstm.chunkwise import mlstm_chunkwise_torch_autograd

from ..test_params import final_combinations

LOGGER = logging.getLogger(__name__)


class TestChunkwiseTorchAgVsStableTorchLong:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.xfail(reason="Fails due to numerical instability")
    @pytest.mark.parametrize(
        ["S", "B", "NH", "DHQK", "DHHV", "target_dtype"], final_combinations
    )
    def test_chunkwise_torch_vs_stable_torch(
        self, test_session_folder, S, B, NH, DHQK, DHHV, target_dtype
    ):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_test_parallel_interface(
            baseline_fn=mlstm_parallel_stable_torch_autograd,
            target_fn=mlstm_chunkwise_torch_autograd,
            baseline_name="parallel_stable_ag",
            target_name="chunkwise_ag",
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
            test_folder_name_prefix="chunkwise-torch",
            save_dir=str(test_session_folder),
            add_fp64_baseline=True,
        )


class TestChunkwiseTorchAgVsUnstableTorchLong:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.xfail(reason="Fails due to numerical instability")
    @pytest.mark.parametrize(
        ["S", "B", "NH", "DHQK", "DHHV", "target_dtype"], final_combinations
    )
    def test_chunkwise_torch_vs_unstable_torch(
        self, test_session_folder, S, B, NH, DHQK, DHHV, target_dtype
    ):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_test_parallel_interface(
            baseline_fn=mlstm_parallel_torch_autograd,
            target_fn=mlstm_chunkwise_torch_autograd,
            baseline_name="parallel_unstable_ag",
            target_name="chunkwise_ag",
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
            test_folder_name_prefix="chunkwise-torch",
            save_dir=str(test_session_folder),
            add_fp64_baseline=True,
        )
