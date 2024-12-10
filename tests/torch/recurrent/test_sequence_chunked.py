#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import pytest
import torch

from mlstm_kernels.torch.recurrent.native_sequence import (
    mlstm_recurrent_sequence__native_fw,
    mlstm_recurrent_sequence__triton_step_fused_fw,
)


def test_state_passing__native_step(mlstm_state_passing_test, state_passing_qkvif):
    num_chunks = state_passing_qkvif[0].shape[2] // 64  # <- chunk size = 64

    mlstm_state_passing_test(
        kernel_fn=mlstm_recurrent_sequence__native_fw,
        q=state_passing_qkvif[0],
        k=state_passing_qkvif[1],
        v=state_passing_qkvif[2],
        igate_preact=state_passing_qkvif[3],
        fgate_preact=state_passing_qkvif[4],
        num_chunks=num_chunks,
        rtol=1e-6,
        atol=1e-6,
        device="cuda",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
def test_state_passing__triton_step_fused(
    mlstm_state_passing_test, state_passing_qkvif
):
    num_chunks = state_passing_qkvif[0].shape[2] // 64  # <- chunk size = 64

    mlstm_state_passing_test(
        kernel_fn=mlstm_recurrent_sequence__triton_step_fused_fw,
        q=state_passing_qkvif[0],
        k=state_passing_qkvif[1],
        v=state_passing_qkvif[2],
        igate_preact=state_passing_qkvif[3],
        fgate_preact=state_passing_qkvif[4],
        num_chunks=num_chunks,
        rtol=1e-6,
        atol=1e-6,
        device="cuda",
    )


# There is probably a bug in triton_step
# @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
# def test_state_passing__triton_step(mlstm_state_passing_test, state_passing_qkvif):
#     num_chunks = state_passing_qkvif[0].shape[2] // 64  # <- chunk size = 64

#     mlstm_state_passing_test(
#         kernel_fn=mlstm_recurrent_sequence__triton_step_fw,
#         q=state_passing_qkvif[0],
#         k=state_passing_qkvif[1],
#         v=state_passing_qkvif[2],
#         igate_preact=state_passing_qkvif[3],
#         fgate_preact=state_passing_qkvif[4],
#         num_chunks=num_chunks,
#         rtol=2e-3,
#         atol=2e-3,
#         device="cuda",
#     )
