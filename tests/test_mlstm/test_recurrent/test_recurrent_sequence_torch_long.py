import pytest
import logging
import torch

from ...common import test_session_folder

from .common_templates import template_torch_parallel_vs_torch_recurrent_sequence


LOGGER = logging.getLogger(__name__)


# combinations_long = {
#     "S": [512, 2048, 4096, 8192],
#     "B": [1, 1, 1, 1],
#     "NH": [1, 1, 1, 1],
#     "DHQK": [128, 128, 128, 128],
#     "DHHV": [128, 128, 128, 128],
# }
combinations_long = {
    "S": [128, 1024, 4096, 8192],
    "B":  [1, 1, 1, 1],   # [2, 2, 2, 2],
    "NH": [1, 1, 1, 1],  # [3, 3, 3, 3],
    "DHQK": [16,16,16,16], #[5, 5, 5, 5],
    "DHHV": [16,16,16,16], #[5, 5, 5, 5],
}
combinations_long_list = [values for values in zip(*combinations_long.values())]


class TestRecurrentVsParallelTorchLong:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.xfail(reason="Fails due to numerical instability")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], combinations_long_list)
    def test_recurrent_vs_parallel_long_fp16(
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
            atol_fw=1.0,  # 3.0
            rtol_fw=1.0,
            atol_fwbw=1.5,  # 3.5
            rtol_fwbw=1.0,
            vmax=1.0,
            test_folder_name=f"torch_parallel_vs_torch_recurrent_sequence_S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}",
            save_dir=str(test_session_folder),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.xfail(reason="Fails due to numerical instability")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], combinations_long_list)
    def test_recurrent_vs_parallel_long_bf16(
        self, test_session_folder, S, B, NH, DHQK, DHHV
    ):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_torch_parallel_vs_torch_recurrent_sequence(
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            DTYPE=torch.bfloat16,
            atol_fw=1.0,  # 3.0
            rtol_fw=1.0,
            atol_fwbw=1.5,  # 3.5
            rtol_fwbw=1.0,
            vmax=1.0,
            test_folder_name=f"torch_parallel_vs_torch_recurrent_sequence_S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}",
            save_dir=str(test_session_folder),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.xfail(reason="Fails due to numerical instability")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], combinations_long_list)
    def test_recurrent_vs_parallel_long_fp32(
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
            atol_fw=1.0,  # 3.0
            rtol_fw=1.0,
            atol_fwbw=1.5,  # 3.5
            rtol_fwbw=1.0,
            vmax=1e-3,
            test_folder_name=f"torch_parallel_vs_torch_recurrent_sequence_S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}",
            save_dir=str(test_session_folder),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.xfail(reason="Fails due to numerical instability or OOM.")
    @pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], combinations_long_list)
    def test_recurrent_vs_parallel_long_fp64(
        self, test_session_folder, S, B, NH, DHQK, DHHV
    ):
        print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
        template_torch_parallel_vs_torch_recurrent_sequence(
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            DTYPE=torch.float64,
            atol_fw=1.0,  # 3.0
            rtol_fw=1.0,
            atol_fwbw=1.5,  # 3.5
            rtol_fwbw=1.0,
            vmax=1e-3,
            test_folder_name=f"torch_parallel_vs_torch_recurrent_sequence_S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}",
            save_dir=str(test_session_folder),
        )
