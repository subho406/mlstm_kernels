#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

from functools import partial
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .losses_tests import loss_layernorm_offset_quadratic


def load_torch_parallel_test_data(
    file: Path, targets_from: Literal["baseline", "target"] = "target"
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Load test data from a torch parallel test.
    Loads the outputs from

    Args:
        file: Path to the file containing the test data.
        targets_from: The target data to load. Either "baseline" or "target".

    Returns:
        A tuple containing the input data and the target data.
    """
    data = np.load(file)

    input_data = {
        "matQ": data["matQ"],
        "matK": data["matK"],
        "matV": data["matV"],
        "vecI": data["vecI"],
        "vecF": data["vecF"],
    }

    target_data = {
        "matH": data[f"matH_{targets_from}"],
        "matQ_grad": data[f"matQ_{targets_from}_grad"],
        "matK_grad": data[f"matK_{targets_from}_grad"],
        "matV_grad": data[f"matV_{targets_from}_grad"],
        "vecI_grad": data[f"vecI_{targets_from}_grad"],
        "vecF_grad": data[f"vecF_{targets_from}_grad"],
    }

    return input_data, target_data


def check_jax_against_pytorch_reference(
    torch_test_data_file: Path,
    jax_mlstm_parallel_fn: callable,
    atol_fw: float,
    rtol_fw: float,
    atol_fwbw: float,
    rtol_fwbw: float,
) -> None:
    torch_input_data, torch_target_data = load_torch_parallel_test_data(
        torch_test_data_file, targets_from="target"
    )

    matQ = jnp.array(torch_input_data["matQ"], dtype=jnp.float32)
    matK = jnp.array(torch_input_data["matK"], dtype=jnp.float32)
    matV = jnp.array(torch_input_data["matV"], dtype=jnp.float32)
    vecI = jnp.array(torch_input_data["vecI"], dtype=jnp.float32)
    vecF = jnp.array(torch_input_data["vecF"], dtype=jnp.float32)

    jax_mlstm_parallel_fn = jax.jit(jax_mlstm_parallel_fn)

    matH_jax = jax_mlstm_parallel_fn(matQ, matK, matV, vecI, vecF)

    def func_and_loss(q, k, v, i, f, func):
        h = func(q=q, k=k, v=v, i=i, f=f)
        return loss_layernorm_offset_quadratic(h)

    jax_mlstm_parallel_grad_fn = jax.grad(
        partial(func_and_loss, func=jax_mlstm_parallel_fn), (0, 1, 2, 3, 4)
    )

    jax_mlstm_parallel_fn = jax.jit(jax_mlstm_parallel_fn)
    # Note: we only run the backward pass, but do not check for correctness, see below.
    matQ_grad, matK_grad, matV_grad, vecI_grad, vecF_grad = jax_mlstm_parallel_grad_fn(
        matQ, matK, matV, vecI, vecF
    )

    jax_target_data = {
        "matH": jax.device_get(matH_jax),
        "matQ_grad": jax.device_get(matQ_grad),
        "matK_grad": jax.device_get(matK_grad),
        "matV_grad": jax.device_get(matV_grad),
        "vecI_grad": jax.device_get(vecI_grad),
        "vecF_grad": jax.device_get(vecF_grad),
    }

    fw_keys = ["matH"]
    bw_keys = ["matQ_grad", "matK_grad", "matV_grad", "vecI_grad", "vecF_grad"]
    for key in fw_keys:
        np.testing.assert_allclose(
            torch_target_data[key],
            jax_target_data[key],
            atol=atol_fw,
            rtol=rtol_fw,
            err_msg=f"Failed for {key}",
        )


    # We cannot test for correctness, since the losses have a random component
    # which is not reproducible between the two frameworks.
    # we would need to save and load the random component.
    # for key in bw_keys:
    #     np.testing.assert_allclose(torch_target_data[key], jax_target_data[key], atol=atol_fwbw, rtol=rtol_fwbw, err_msg=f"Failed for {key}")
