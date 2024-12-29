#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
from functools import partial

import pytest
import torch

from mlstm_kernels.torch.parallel.native_siging import (
    mlstm_siging_parallel__native_autograd,
    mlstm_siging_parallel__native_custbw,
)

from ...conftest import final_combinations

LOGGER = logging.getLogger(__name__)

TEST_FOLDER_NAME_PREFIX = "parallel-torch-native_siging"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_parallel_native_siging_stablef_normalized(
    test_session_folder,
    test_output_folder,
    mlstm_parallel_interface_test,
    S,
    B,
    NH,
    DHQK,
    DHHV,
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=partial(
            mlstm_siging_parallel__native_autograd, stable_fgate=True, normalize=True
        ),
        target_fn=partial(mlstm_siging_parallel__native_custbw, stable_fgate=True, normalize=True),
        baseline_name="parallel_siging_stable_normalized_autograd",
        target_name="parallel_siging_stable_normalized_custbw",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=torch.float32,
        atol_fw=1e-4,
        rtol_fw=1e-3,
        atol_fwbw=2e-3,
        rtol_fwbw=1e-2,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        save_output_tensors_dir=str(test_output_folder),
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
@pytest.mark.parametrize(["S", "B", "NH", "DHQK", "DHHV"], final_combinations)
def test_parallel_native_siging_stablef_unnormalized(
    test_session_folder,
    test_output_folder,
    mlstm_parallel_interface_test,
    S,
    B,
    NH,
    DHQK,
    DHHV,
):
    print(f"S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}")
    mlstm_parallel_interface_test(
        baseline_fn=partial(
            mlstm_siging_parallel__native_autograd, stable_fgate=True, normalize=False
        ),
        target_fn=partial(mlstm_siging_parallel__native_custbw, stable_fgate=True, normalize=False),
        baseline_name="parallel_siging_stable_unnormalized_autograd",
        target_name="parallel_siging_stable_unnormalized_custbw",
        S=S,
        B=B,
        NH=NH,
        DHQK=DHQK,
        DHHV=DHHV,
        dtype=torch.float32,
        atol_fw=1e-4,
        rtol_fw=1e-3,
        atol_fwbw=2e-4,
        rtol_fwbw=5e-3,
        vmax=1e-3,
        test_folder_name_prefix=TEST_FOLDER_NAME_PREFIX,
        save_dir=str(test_session_folder),
        add_fp64_baseline=False,
        save_output_tensors_dir=str(test_output_folder),
    )

