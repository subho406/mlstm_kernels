import logging

from mlstm_kernels.torch.chunkwise.triton_xl_chunk import mlstm_chunkwise__xl_chunk
from mlstm_kernels.torch.parallel.native_stablef import mlstm_parallel__native_stablef_custbw


import pytest
import torch

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "chunkwise-triton_xl_chunk"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_triton_chunkwise_xl_chunk_vs_native_parrallel_stablef_fp32(
    test_session_folder, test_output_folder, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_custbw,
        target_fn=mlstm_chunkwise__xl_chunk,
        baseline_name="native_parallel_stablef_custbw",
        target_name="triton_chunkwise_xl_chunk",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=torch.float32,
        atol_fw=2e-2,
        rtol_fw=5e-2,
        atol_fwbw=3e-1,  # we need to increase this tolerance for vecF.grad (max diff val 0.267...)
        rtol_fwbw=0.5,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        save_output_tensors_dir=str(test_output_folder),
    )
