import pytest
import logging
import torch
from itertools import product

from ..common_templates import template_test_parallel_interface
from ...common import test_session_folder  # noqa

from mlstm_kernels.mlstm.parallel import (
    mlstm_parallel_torch_autograd,
    mlstm_parallel_stable_torch_autograd,
)


LOGGER = logging.getLogger(__name__)


# combinations_long = {
#     "S": [512, 2048, 4096, 8192],
#     "B": [1, 1, 1, 1],
#     "NH": [1, 1, 1, 1],
#     "DHQK": [128, 128, 128, 128],
#     "DHHV": [128, 128, 128, 128],
# }
# combinations_long = {
#     "S": [128, 1024, 4096, 8192],
#     "B":  [1, 1, 1, 1],   # [2, 2, 2, 2],
#     "NH": [1, 1, 1, 1],  # [3, 3, 3, 3],
#     "DHQK": [16,16,16,16], #[5, 5, 5, 5],
#     "DHHV": [16,16,16,16], #[5, 5, 5, 5],
# }
combinations_long = {
    "S": [8192],
    "B": [1],  # [2, 2, 2, 2],
    "NH": [1],  # [3, 3, 3, 3],
    "DHQK": [16],  # [5, 5, 5, 5],
    "DHHV": [16],  # [5, 5, 5, 5],
}
combinations_long_list = [values for values in zip(*combinations_long.values())]
target_dtypes = ["float16", "bfloat16", "float32", "float64"]

final_combinations = [
    (*combinations, dtype)
    for combinations, dtype in product(combinations_long_list, target_dtypes)
]


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
