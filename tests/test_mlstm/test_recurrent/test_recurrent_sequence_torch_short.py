import pytest
import logging
import torch

from ...common import test_session_folder

from .common_template import template_torch_parallel_vs_torch_recurrent_sequence


LOGGER = logging.getLogger(__name__)


combinations_short = {
    "S": [32, 32, 32],
    "B": [1, 1, 2],
    "NH": [1, 1, 3],
    "DHQK": [16, 16, 16],
    "DHHV": [16, 32, 16],
}
combinations_short_list = [values for values in zip(*combinations_short.values())]


class TestRecurrentVsParallelTorchShort:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], combinations_short_list)
    def test_recurrent_vs_parallel_short_fp32(
        self, test_session_folder, S, B, NH, DHQK, DHHV
    ):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_torch_parallel_vs_torch_recurrent_sequence(
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            DTYPE=torch.float32,
            atol_fw=1e-3,
            rtol_fw=1e-2,
            atol_fwbw=1e-2,
            rtol_fwbw=1e-2,
            test_folder_name=f"torch_parallel_vs_torch_recurrent_sequence_S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}",
            save_dir=str(test_session_folder),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], combinations_short_list)
    def test_recurrent_vs_parallel_short_fp16(
        self, test_session_folder, S, B, NH, DHQK, DHHV
    ):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_torch_parallel_vs_torch_recurrent_sequence(
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            DTYPE=torch.float16,
            atol_fw=1e-1,
            rtol_fw=1e-1,
            atol_fwbw=5e-1,
            rtol_fwbw=5e-1,
            test_folder_name=f"torch_parallel_vs_torch_recurrent_sequence_S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}",
            save_dir=str(test_session_folder),
        )
