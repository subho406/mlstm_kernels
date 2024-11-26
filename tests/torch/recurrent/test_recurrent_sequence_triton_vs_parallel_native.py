import logging
from functools import partial

import pytest
import torch

from mlstm_kernels.torch.parallel.native_stablef import (
    mlstm_parallel__native_stablef_autograd,
)
from mlstm_kernels.torch.recurrent.native_sequence import (
    mlstm_recurrent_sequence__triton_step_fused_fw,
    mlstm_recurrent_sequence__triton_step_fw,
)

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "recurrent_seq-triton"


@pytest.mark.skip(reason="Triton step kernel (non-fused) has a bug.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_recurrent_sequence_triton_step_vs_native_parrallel_stablef_fp32(
    test_session_folder, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_autograd,
        target_fn=mlstm_recurrent_sequence__triton_step_fw,
        baseline_name="native_parallel_stablef_autograd",
        target_name="native_recurrent_sequence__triton_step_fw",
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
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        run_backward=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_recurrent_sequence_triton_step_fused_vs_native_parrallel_stablef_fp32(
    test_session_folder, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_autograd,
        target_fn=mlstm_recurrent_sequence__triton_step_fused_fw,
        baseline_name="native_parallel_stablef_autograd",
        target_name="native_recurrent_sequence__triton_step_fused_fw",
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
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        run_backward=False,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_recurrent_sequence_triton_step_fused_vs_native_parrallel_stablef_state_bf16(
    test_session_folder, mlstm_parallel_interface_test, S, B, NH, DHQK, DHHV
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=mlstm_parallel__native_stablef_autograd,
        target_fn=partial(
            mlstm_recurrent_sequence__triton_step_fused_fw, dtype_state=torch.bfloat16
        ),
        baseline_name="native_parallel_stablef_autograd",
        target_name="native_recurrent_sequence__triton_step_fused_fw",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=torch.float32,
        atol_fw=1.2e-1,
        rtol_fw=1e-2,
        atol_fwbw=1e-4,
        rtol_fwbw=1e-2,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        run_backward=False,
    )
