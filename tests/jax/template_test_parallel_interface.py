#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mlstm_kernels.jax.utils import dtype2str, to_numpy
from mlstm_kernels.utils.test.checks import check_correctness
from mlstm_kernels.utils.time import Stopwatch

from .losses_tests import loss_layernorm_offset_quadratic

LOGGER = logging.getLogger(__name__)


def template_test_parallel_interface(
    baseline_fn: Callable,
    target_fn: Callable,
    baseline_name: str,
    target_name: str,
    S: int = 2048,
    B: int = 2,
    NH: int = 3,
    DHQK: int = 128,  # dim per head
    DHHV: int = 256,
    vecI_offset: float = 0.0,
    vecF_offset: float = 0.0,
    ln_eps: float = 1e-5,
    dtype=jnp.float32,
    eps: float = 1e-6,
    vmax: float = None,
    atol_fw: float = 1e-3,
    rtol_fw: float = 1e-2,
    atol_fwbw: float = 1e-2,
    rtol_fwbw: float = 1e-2,
    seed: int = 0,
    max_num_batchhead_plots: int = 1,  # -1 means all
    test_folder_name_prefix: str = "torch_parallel_vs_torch_recurrent_sequence",
    save_dir: str = ".",
    run_backward: bool = True,
    add_fp64_baseline: bool = True,  # whether to check against a fp64 baseline too
    return_output_tensors: bool = False,
    save_output_tensors_dir: str | Path | None = None,
    use_jit: bool = True,
) -> tuple[jax.Array, ...] | None:
    """This is a generic test function that tests the parallel interface of the mLSTM.
    It tests the outputs and gradients.
    """

    LOGGER.info(
        f"JAX Test {test_folder_name_prefix} target={target_name} vs. baseline={baseline_name} with S={S}, B={B}, NH={NH}, DHQK={DHQK}, DHHV={DHHV}, DTYPE={dtype}"
    )

    rng = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rng, 5)
    matQ = jax.random.normal(
        rngs[0],
        shape=(B, NH, S, DHQK),
        dtype=jnp.float32,
    )
    matK = jax.random.normal(
        rngs[1],
        shape=(B, NH, S, DHQK),
        dtype=jnp.float32,
    )
    matV = jax.random.normal(
        rngs[2],
        shape=(B, NH, S, DHHV),
        dtype=jnp.float32,
    )
    vecI = vecI_offset + jax.random.normal(rngs[3], shape=(B, NH, S), dtype=jnp.float32)
    vecF = vecF_offset + jax.random.normal(rngs[4], shape=(B, NH, S), dtype=jnp.float32)

    baseline_dtype = dtype
    matQ_baseline = matQ.copy().astype(dtype=baseline_dtype)
    matK_baseline = matK.copy().astype(dtype=baseline_dtype)
    matV_baseline = matV.copy().astype(dtype=baseline_dtype)
    vecI_baseline = vecI.copy().astype(dtype=baseline_dtype)
    vecF_baseline = vecF.copy().astype(dtype=baseline_dtype)

    target_dtype = dtype
    matQ_target = matQ.copy().astype(dtype=target_dtype)
    matK_target = matK.copy().astype(dtype=target_dtype)
    matV_target = matV.copy().astype(dtype=target_dtype)
    vecI_target = vecI.copy().astype(dtype=target_dtype)
    vecF_target = vecF.copy().astype(dtype=target_dtype)

    test_folder_name = f"{test_folder_name_prefix}_{target_name}-vs-{baseline_name}_S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}"

    # same dtype baseline

    def func_and_loss(q, k, v, i, f, func):
        h = func(q=q, k=k, v=v, i=i, f=f, eps=eps)
        return loss_layernorm_offset_quadratic(h)

    if run_backward:
        baseline_grad_fn = jax.grad(
            partial(func_and_loss, func=baseline_fn), argnums=(0, 1, 2, 3, 4)
        )
        target_grad_fn = jax.grad(
            partial(func_and_loss, func=target_fn), argnums=(0, 1, 2, 3, 4)
        )

    if use_jit:
        baseline_fn = jax.jit(baseline_fn, static_argnames=("eps"))
        target_fn = jax.jit(target_fn, static_argnames=("eps"))
        baseline_grad_fn = jax.jit(baseline_grad_fn)
        target_grad_fn = jax.jit(target_grad_fn)

    sw = Stopwatch()
    sw.start()
    matH_baseline = baseline_fn(
        q=matQ_baseline,
        k=matK_baseline,
        v=matV_baseline,
        i=vecI_baseline,
        f=vecF_baseline,
        eps=eps,
    )
    fw_seconds = sw.lap()
    if run_backward:
        (
            matQ_baseline_grad,
            matK_baseline_grad,
            matV_baseline_grad,
            vecI_baseline_grad,
            vecF_baseline_grad,
        ) = baseline_grad_fn(
            matQ_baseline, matK_baseline, matV_baseline, vecI_baseline, vecF_baseline
        )
    else:
        matQ_baseline_grad = None
        matK_baseline_grad = None
        matV_baseline_grad = None
        vecI_baseline_grad = None
        vecF_baseline_grad = None

    fwbw_seconds = sw.stop()

    print(
        f"{baseline_name} | fw (ms): {fw_seconds * 1000}, fwbw (ms): {fwbw_seconds * 1000}"
    )
    LOGGER.info(
        f"{baseline_name} | fw (ms): {fw_seconds * 1000}, fwbw (ms): {fwbw_seconds * 1000}"
    )

    # target
    sw = Stopwatch()
    sw.start()
    matH_target = target_fn(
        q=matQ_target,
        k=matK_target,
        v=matV_target,
        i=vecI_target,
        f=vecF_target,
        eps=eps,
    )
    fw_seconds = sw.lap()
    if run_backward:
        (
            matQ_target_grad,
            matK_target_grad,
            matV_target_grad,
            vecI_target_grad,
            vecF_target_grad,
        ) = target_grad_fn(
            matQ_target, matK_target, matV_target, vecI_target, vecF_target
        )
    else:
        matQ_target_grad = None
        matK_target_grad = None
        matV_target_grad = None
        vecI_target_grad = None
        vecF_target_grad = None

    fwbw_seconds = sw.stop()
    print(
        f"{target_name} | fw (ms): {fw_seconds * 1000}, fwbw (ms): {fwbw_seconds * 1000}"
    )
    LOGGER.info(
        f"{target_name} | fw (ms): {fw_seconds * 1000}, fwbw (ms): {fwbw_seconds * 1000}"
    )

    test_specifier_template_str = "{specifier}_bl-{dtype}"

    ## checks baseline same dtype
    def do_checks(
        matH_baseline: jax.Array,
        matH_target: jax.Array,
        matQ_baseline_grad: jax.Array | None,
        matQ_target_grad: jax.Array | None,
        matK_baseline_grad: jax.Array | None,
        matK_target_grad: jax.Array | None,
        matV_baseline_grad: jax.Array | None,
        matV_target_grad: jax.Array | None,
        vecI_baseline_grad: jax.Array | None,
        vecI_target_grad: jax.Array | None,
        vecF_baseline_grad: jax.Array | None,
        vecF_target_grad: jax.Array | None,
    ):
        matH_match = check_correctness(
            test_specifier=test_specifier_template_str.format(
                specifier="matH", dtype=dtype2str(matH_baseline.dtype)
            ),
            baseline=to_numpy(matH_baseline),
            target=to_numpy(matH_target),
            atol=atol_fw,
            rtol=rtol_fw,
            vmax=vmax,
            max_num_batchhead_plots=max_num_batchhead_plots,
            dtype_str=dtype2str(matH_baseline.dtype),
            savepath=f"{save_dir}/{test_folder_name}",
        )
        if run_backward:
            matQgrad_match = check_correctness(
                test_specifier=test_specifier_template_str.format(
                    specifier="matQgrad", dtype=dtype2str(matQ_baseline_grad.dtype)
                ),
                baseline=to_numpy(matQ_baseline_grad),
                target=to_numpy(matQ_target_grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
                vmax=vmax,
                max_num_batchhead_plots=max_num_batchhead_plots,
                dtype_str=dtype2str(matQ_baseline_grad.dtype),
                savepath=f"{save_dir}/{test_folder_name}",
            )
            matKgrad_match = check_correctness(
                test_specifier=test_specifier_template_str.format(
                    specifier="matKgrad", dtype=dtype2str(matK_baseline_grad.dtype)
                ),
                baseline=to_numpy(matK_baseline_grad),
                target=to_numpy(matK_target_grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
                vmax=vmax,
                max_num_batchhead_plots=max_num_batchhead_plots,
                dtype_str=dtype2str(matK_baseline_grad.dtype),
                savepath=f"{save_dir}/{test_folder_name}",
            )
            matVgrad_match = check_correctness(
                test_specifier=test_specifier_template_str.format(
                    specifier="matVgrad", dtype=dtype2str(matV_baseline_grad.dtype)
                ),
                baseline=to_numpy(matV_baseline_grad),
                target=to_numpy(matV_target_grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
                vmax=vmax,
                max_num_batchhead_plots=max_num_batchhead_plots,
                dtype_str=dtype2str(matV_baseline_grad.dtype),
                savepath=f"{save_dir}/{test_folder_name}",
            )
            vecIgrad_match = check_correctness(
                test_specifier=test_specifier_template_str.format(
                    specifier="vecIgrad", dtype=dtype2str(vecI_baseline_grad.dtype)
                ),
                baseline=to_numpy(vecI_baseline_grad),
                target=to_numpy(vecI_target_grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
                vmax=vmax,
                max_num_batchhead_plots=max_num_batchhead_plots,
                dtype_str=dtype2str(vecI_baseline_grad.dtype),
                savepath=f"{save_dir}/{test_folder_name}",
            )
            vecFgrad_match = check_correctness(
                test_specifier=test_specifier_template_str.format(
                    specifier="vecFgrad", dtype=dtype2str(vecF_baseline_grad.dtype)
                ),
                baseline=to_numpy(vecF_baseline_grad),
                target=to_numpy(vecF_target_grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
                vmax=vmax,
                max_num_batchhead_plots=max_num_batchhead_plots,
                dtype_str=dtype2str(vecF_baseline_grad.dtype),
                savepath=f"{save_dir}/{test_folder_name}",
            )
        else:
            matQgrad_match = True
            matKgrad_match = True
            matVgrad_match = True
            vecIgrad_match = True
            vecFgrad_match = True

        return (
            matH_match,
            matQgrad_match,
            matKgrad_match,
            matVgrad_match,
            vecIgrad_match,
            vecFgrad_match,
        )

    (
        matH_match,
        matQgrad_match,
        matKgrad_match,
        matVgrad_match,
        vecIgrad_match,
        vecFgrad_match,
    ) = do_checks(
        matH_baseline=matH_baseline,
        matH_target=matH_target,
        matQ_baseline_grad=matQ_baseline_grad,
        matQ_target_grad=matQ_target_grad,
        matK_baseline_grad=matK_baseline_grad,
        matK_target_grad=matK_target_grad,
        matV_baseline_grad=matV_baseline_grad,
        matV_target_grad=matV_target_grad,
        vecI_baseline_grad=vecI_baseline_grad,
        vecI_target_grad=vecI_target_grad,
        vecF_baseline_grad=vecF_baseline_grad,
        vecF_target_grad=vecF_target_grad,
    )

    # float64 baseline
    if add_fp64_baseline:
        baseline_fp64_dtype = jnp.float64
        matQ_baseline_fp64 = matQ.copy().astype(dtype=baseline_fp64_dtype)
        matK_baseline_fp64 = matK.copy().astype(dtype=baseline_fp64_dtype)
        matV_baseline_fp64 = matV.copy().astype(dtype=baseline_fp64_dtype)
        vecI_baseline_fp64 = vecI.copy().astype(dtype=baseline_fp64_dtype)
        vecF_baseline_fp64 = vecF.copy().astype(dtype=baseline_fp64_dtype)

        matH_baseline_fp64 = baseline_fn(
            q=matQ_baseline_fp64,
            k=matK_baseline_fp64,
            v=matV_baseline_fp64,
            i=vecI_baseline_fp64,
            f=vecF_baseline_fp64,
            eps=eps,
        )
        if run_backward:
            (
                matQ_baseline_fp64_grad,
                matK_baseline_fp64_grad,
                matV_baseline_fp64_grad,
                vecI_baseline_fp64_grad,
                vecF_baseline_fp64_grad,
            ) = baseline_grad_fn(
                matQ_baseline_fp64,
                matK_baseline_fp64,
                matV_baseline_fp64,
                vecI_baseline_fp64,
                vecF_baseline_fp64,
            )
        (
            matH_match_bl_fp64,
            matQgrad_match_bl_fp64,
            matKgrad_match_bl_fp64,
            matVgrad_match_bl_fp64,
            vecIgrad_match_bl_fp64,
            vecFgrad_match_bl_fp64,
        ) = do_checks(
            matH_baseline=matH_baseline_fp64,
            matH_target=matH_target,
            matQ_baseline_grad=matQ_baseline_fp64_grad,
            matQ_target_grad=matQ_target_grad,
            matK_baseline_grad=matK_baseline_fp64_grad,
            matK_target_grad=matK_target_grad,
            matV_baseline_grad=matV_baseline_fp64_grad,
            matV_target_grad=matV_target_grad,
            vecI_baseline_grad=vecI_baseline_fp64_grad,
            vecI_target_grad=vecI_target_grad,
            vecF_baseline_grad=vecF_baseline_fp64_grad,
            vecF_target_grad=vecF_target_grad,
        )

    np.testing.assert_allclose(
        to_numpy(matH_baseline),
        to_numpy(matH_target),
        atol=atol_fw,
        rtol=rtol_fw,
        err_msg="matH",
    )
    if run_backward:
        np.testing.assert_allclose(
            to_numpy(matQ_baseline_grad),
            to_numpy(matQ_target_grad),
            atol=atol_fwbw,
            rtol=rtol_fwbw,
            err_msg="matQgrad",
        )
        np.testing.assert_allclose(
            to_numpy(matK_baseline_grad),
            to_numpy(matK_target_grad),
            atol=atol_fwbw,
            rtol=rtol_fwbw,
            err_msg="matKgrad",
        )
        np.testing.assert_allclose(
            to_numpy(matV_baseline_grad),
            to_numpy(matV_target_grad),
            atol=atol_fwbw,
            rtol=rtol_fwbw,
            err_msg="matVgrad",
        )
        np.testing.assert_allclose(
            to_numpy(vecI_baseline_grad),
            to_numpy(vecI_target_grad),
            atol=atol_fwbw,
            rtol=rtol_fwbw,
            err_msg="vecIgrad",
        )
        np.testing.assert_allclose(
            to_numpy(vecF_baseline_grad),
            to_numpy(vecF_target_grad),
            atol=atol_fwbw,
            rtol=rtol_fwbw,
            err_msg="vecFgrad",
        )

    if add_fp64_baseline:
        np.testing.assert_allclose(
            to_numpy(matH_baseline_fp64),
            to_numpy(matH_target),
            atol=atol_fw,
            rtol=rtol_fw,
        )
        if run_backward:
            np.testing.assert_allclose(
                to_numpy(matQ_baseline_fp64_grad),
                to_numpy(matQ_target_grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
            )
            np.testing.assert_allclose(
                to_numpy(matK_baseline_fp64_grad),
                to_numpy(matK_target_grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
            )
            np.testing.assert_allclose(
                to_numpy(matV_baseline_fp64_grad),
                to_numpy(matV_target_grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
            )
            np.testing.assert_allclose(
                to_numpy(vecI_baseline_fp64_grad),
                to_numpy(vecI_target_grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
            )
            np.testing.assert_allclose(
                to_numpy(vecF_baseline_fp64_grad),
                to_numpy(vecF_target_grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
            )

    if save_output_tensors_dir is not None:
        save_output_tensors_dir = Path(save_output_tensors_dir)
        save_output_tensors_dir.mkdir(parents=True, exist_ok=True)
        save_output_tensors_file = save_output_tensors_dir / f"{test_folder_name}.npz"
        output_tensors = {
            "matH_baseline": to_numpy(matH_baseline),
            "matQ_baseline_grad": to_numpy(matQ_baseline_grad),
            "matK_baseline_grad": to_numpy(matK_baseline_grad),
            "matV_baseline_grad": to_numpy(matV_baseline_grad),
            "vecI_baseline_grad": to_numpy(vecI_baseline_grad),
            "vecF_baseline_grad": to_numpy(vecF_baseline_grad),
            "matH_target": to_numpy(matH_target),
            "matQ_target_grad": to_numpy(matQ_target_grad),
            "matK_target_grad": to_numpy(matK_target_grad),
            "matV_target_grad": to_numpy(matV_target_grad),
            "vecI_target_grad": to_numpy(vecI_target_grad),
            "vecF_target_grad": to_numpy(vecF_target_grad),
            "matQ": to_numpy(matQ),
            "matK": to_numpy(matK),
            "matV": to_numpy(matV),
            "vecI": to_numpy(vecI),
            "vecF": to_numpy(vecF),
        }
        np.savez(save_output_tensors_file, **output_tensors)

    if return_output_tensors:
        return (
            matH_baseline,
            matQ_baseline_grad,
            matK_baseline_grad,
            matV_baseline_grad,
            vecI_baseline_grad,
            vecF_baseline_grad,
            matH_target,
            matQ_target_grad,
            matK_target_grad,
            matV_target_grad,
            vecI_target_grad,
            vecF_target_grad,
        )
    else:
        return None
