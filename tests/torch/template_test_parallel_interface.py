#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
from collections.abc import Callable
from pathlib import Path

from mlstm_kernels.torch.utils import dtype2str, to_numpy
from mlstm_kernels.utils.test.checks import check_correctness
from mlstm_kernels.utils.time import Stopwatch

import numpy as np
import torch

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
    dtype=torch.float32,
    device=torch.device("cuda:0"),
    EPS: float = 1e-6,
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
) -> tuple[torch.Tensor, ...] | None:
    """This is a generic test function that tests the parallel interface of the mLSTM.
    It tests the outputs and gradients.
    """

    LOGGER.info(
        f"Test {test_folder_name_prefix} target={target_name} vs. baseline={baseline_name} with S={S}, B={B}, NH={NH}, DHQK={DHQK}, DHHV={DHHV}, DTYPE={dtype}"
    )

    torch.manual_seed(seed)
    matQ = torch.randn((B, NH, S, DHQK), dtype=torch.float32, device=device)
    matK = torch.randn((B, NH, S, DHQK), dtype=torch.float32, device=device)
    matV = torch.randn((B, NH, S, DHHV), dtype=torch.float32, device=device)
    vecI = vecI_offset + torch.randn((B, NH, S), dtype=torch.float32, device=device)
    vecF = vecF_offset + torch.randn((B, NH, S), dtype=torch.float32, device=device)

    baseline_dtype = dtype
    matQ_baseline = matQ.clone().to(dtype=baseline_dtype).detach().requires_grad_(True)
    matK_baseline = matK.clone().to(dtype=baseline_dtype).detach().requires_grad_(True)
    matV_baseline = matV.clone().to(dtype=baseline_dtype).detach().requires_grad_(True)
    vecI_baseline = vecI.clone().to(dtype=baseline_dtype).detach().requires_grad_(True)
    vecF_baseline = vecF.clone().to(dtype=baseline_dtype).detach().requires_grad_(True)

    target_dtype = dtype
    matQ_target = matQ.clone().to(dtype=target_dtype).detach().requires_grad_(True)
    matK_target = matK.clone().to(dtype=target_dtype).detach().requires_grad_(True)
    matV_target = matV.clone().to(dtype=target_dtype).detach().requires_grad_(True)
    vecI_target = vecI.clone().to(dtype=target_dtype).detach().requires_grad_(True)
    vecF_target = vecF.clone().to(dtype=target_dtype).detach().requires_grad_(True)

    test_folder_name = f"{test_folder_name_prefix}_{target_name}-vs-{baseline_name}_S{S}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}"

    # same dtype baseline
    sw = Stopwatch()
    sw.start()
    matH_baseline = baseline_fn(
        q=matQ_baseline,
        k=matK_baseline,
        v=matV_baseline,
        i=vecI_baseline,
        f=vecF_baseline,
        eps=EPS,
    )
    fw_seconds = sw.lap()
    if run_backward:
        loss_layernorm_offset_quadratic(matH_baseline).backward()
    fwbw_seconds = sw.stop()

    print(f"{baseline_name} | fw (ms): {fw_seconds * 1000}, fwbw (ms): {fwbw_seconds * 1000}")
    LOGGER.info(f"{baseline_name} | fw (ms): {fw_seconds * 1000}, fwbw (ms): {fwbw_seconds * 1000}")

    # target
    sw = Stopwatch()
    sw.start()
    matH_target = target_fn(
        q=matQ_target,
        k=matK_target,
        v=matV_target,
        i=vecI_target,
        f=vecF_target,
        eps=EPS,
    )
    fw_seconds = sw.lap()
    if run_backward:
        loss_layernorm_offset_quadratic(matH_target, eps=ln_eps).backward()
    fwbw_seconds = sw.stop()
    print(f"{target_name} | fw (ms): {fw_seconds * 1000}, fwbw (ms): {fwbw_seconds * 1000}")
    LOGGER.info(f"{target_name} | fw (ms): {fw_seconds * 1000}, fwbw (ms): {fwbw_seconds * 1000}")

    test_specifier_template_str = "{specifier}_bl-{dtype}"

    ## checks baseline same dtype
    def do_checks(
        matH_baseline,
        matH_target,
        matQ_baseline,
        matQ_target,
        matK_baseline,
        matK_target,
        matV_baseline,
        matV_target,
        vecI_baseline,
        vecI_target,
        vecF_baseline,
        vecF_target,
    ):
        matH_match = check_correctness(
            test_specifier=test_specifier_template_str.format(specifier="matH", dtype=dtype2str(matH_baseline.dtype)),
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
                    specifier="matQgrad", dtype=dtype2str(matQ_baseline.grad.dtype)
                ),
                baseline=to_numpy(matQ_baseline.grad),
                target=to_numpy(matQ_target.grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
                vmax=vmax,
                max_num_batchhead_plots=max_num_batchhead_plots,
                dtype_str=dtype2str(matQ_baseline.grad.dtype),
                savepath=f"{save_dir}/{test_folder_name}",
            )
            matKgrad_match = check_correctness(
                test_specifier=test_specifier_template_str.format(
                    specifier="matKgrad", dtype=dtype2str(matK_baseline.grad.dtype)
                ),
                baseline=to_numpy(matK_baseline.grad),
                target=to_numpy(matK_target.grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
                vmax=vmax,
                max_num_batchhead_plots=max_num_batchhead_plots,
                dtype_str=dtype2str(matK_baseline.grad.dtype),
                savepath=f"{save_dir}/{test_folder_name}",
            )
            matVgrad_match = check_correctness(
                test_specifier=test_specifier_template_str.format(
                    specifier="matVgrad", dtype=dtype2str(matV_baseline.grad.dtype)
                ),
                baseline=to_numpy(matV_baseline.grad),
                target=to_numpy(matV_target.grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
                vmax=vmax,
                max_num_batchhead_plots=max_num_batchhead_plots,
                dtype_str=dtype2str(matV_baseline.grad.dtype),
                savepath=f"{save_dir}/{test_folder_name}",
            )
            vecIgrad_match = check_correctness(
                test_specifier=test_specifier_template_str.format(
                    specifier="vecIgrad", dtype=dtype2str(vecI_baseline.grad.dtype)
                ),
                baseline=to_numpy(vecI_baseline.grad),
                target=to_numpy(vecI_target.grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
                vmax=vmax,
                max_num_batchhead_plots=max_num_batchhead_plots,
                dtype_str=dtype2str(vecI_baseline.grad.dtype),
                savepath=f"{save_dir}/{test_folder_name}",
            )
            vecFgrad_match = check_correctness(
                test_specifier=test_specifier_template_str.format(
                    specifier="vecFgrad", dtype=dtype2str(vecF_baseline.grad.dtype)
                ),
                baseline=to_numpy(vecF_baseline.grad),
                target=to_numpy(vecF_target.grad),
                atol=atol_fwbw,
                rtol=rtol_fwbw,
                vmax=vmax,
                max_num_batchhead_plots=max_num_batchhead_plots,
                dtype_str=dtype2str(vecF_baseline.grad.dtype),
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
        matQ_baseline=matQ_baseline,
        matQ_target=matQ_target,
        matK_baseline=matK_baseline,
        matK_target=matK_target,
        matV_baseline=matV_baseline,
        matV_target=matV_target,
        vecI_baseline=vecI_baseline,
        vecI_target=vecI_target,
        vecF_baseline=vecF_baseline,
        vecF_target=vecF_target,
    )

    # float64 baseline
    if add_fp64_baseline:
        baseline_fp64_dtype = torch.float64
        matQ_baseline_fp64 = matQ.clone().to(dtype=baseline_fp64_dtype).detach().requires_grad_(True)
        matK_baseline_fp64 = matK.clone().to(dtype=baseline_fp64_dtype).detach().requires_grad_(True)
        matV_baseline_fp64 = matV.clone().to(dtype=baseline_fp64_dtype).detach().requires_grad_(True)
        vecI_baseline_fp64 = vecI.clone().to(dtype=baseline_fp64_dtype).detach().requires_grad_(True)
        vecF_baseline_fp64 = vecF.clone().to(dtype=baseline_fp64_dtype).detach().requires_grad_(True)

        matH_baseline_fp64 = baseline_fn(
            q=matQ_baseline_fp64,
            k=matK_baseline_fp64,
            v=matV_baseline_fp64,
            i=vecI_baseline_fp64,
            f=vecF_baseline_fp64,
            eps=EPS,
        )
        loss_layernorm_offset_quadratic(matH_baseline_fp64).backward()

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
            matQ_baseline=matQ_baseline_fp64,
            matQ_target=matQ_target,
            matK_baseline=matK_baseline_fp64,
            matK_target=matK_target,
            matV_baseline=matV_baseline_fp64,
            matV_target=matV_target,
            vecI_baseline=vecI_baseline_fp64,
            vecI_target=vecI_target,
            vecF_baseline=vecF_baseline_fp64,
            vecF_target=vecF_target,
        )

    np.testing.assert_allclose(to_numpy(matH_baseline), to_numpy(matH_target), atol=atol_fw, rtol=rtol_fw, err_msg="matH")
    if run_backward:
        np.testing.assert_allclose(
            to_numpy(matQ_baseline.grad), to_numpy(matQ_target.grad), atol=atol_fwbw, rtol=rtol_fwbw, err_msg="matQgrad"
        )
        np.testing.assert_allclose(
            to_numpy(matK_baseline.grad), to_numpy(matK_target.grad), atol=atol_fwbw, rtol=rtol_fwbw, err_msg="matKgrad"
        )
        np.testing.assert_allclose(
            to_numpy(matV_baseline.grad), to_numpy(matV_target.grad), atol=atol_fwbw, rtol=rtol_fwbw, err_msg="matVgrad"
        )
        np.testing.assert_allclose(
            to_numpy(vecI_baseline.grad), to_numpy(vecI_target.grad), atol=atol_fwbw, rtol=rtol_fwbw, err_msg="vecIgrad"
        )
        np.testing.assert_allclose(
            to_numpy(vecF_baseline.grad), to_numpy(vecF_target.grad), atol=atol_fwbw, rtol=rtol_fwbw, err_msg="vecFgrad"
        )

    if add_fp64_baseline:
        np.testing.assert_allclose(to_numpy(matH_baseline_fp64), to_numpy(matH_target), atol=atol_fw, rtol=rtol_fw, err_msg="matH_fp64")
        if run_backward:
            np.testing.assert_allclose(
                to_numpy(matQ_baseline_fp64.grad), to_numpy(matQ_target.grad), atol=atol_fwbw, rtol=rtol_fwbw, err_msg="matQgrad_fp64"
            )
            np.testing.assert_allclose(
                to_numpy(matK_baseline_fp64.grad), to_numpy(matK_target.grad), atol=atol_fwbw, rtol=rtol_fwbw, err_msg="matKgrad_fp64"
            )
            np.testing.assert_allclose(
                to_numpy(matV_baseline_fp64.grad), to_numpy(matV_target.grad), atol=atol_fwbw, rtol=rtol_fwbw, err_msg="matVgrad_fp64"
            )
            np.testing.assert_allclose(
                to_numpy(vecI_baseline_fp64.grad), to_numpy(vecI_target.grad), atol=atol_fwbw, rtol=rtol_fwbw, err_msg="vecIgrad_fp64"
            )
            np.testing.assert_allclose(
                to_numpy(vecF_baseline_fp64.grad), to_numpy(vecF_target.grad), atol=atol_fwbw, rtol=rtol_fwbw, err_msg="vecFgrad_fp64"
            )

    if save_output_tensors_dir is not None:
        save_output_tensors_dir = Path(save_output_tensors_dir)
        save_output_tensors_dir.mkdir(parents=True, exist_ok=True)
        save_output_tensors_file = save_output_tensors_dir / f"{test_folder_name}.npz"
        output_tensors = {
            "matH_baseline": to_numpy(matH_baseline),
            "matQ_baseline_grad": to_numpy(matQ_baseline.grad),
            "matK_baseline_grad": to_numpy(matK_baseline.grad),
            "matV_baseline_grad": to_numpy(matV_baseline.grad),
            "vecI_baseline_grad": to_numpy(vecI_baseline.grad),
            "vecF_baseline_grad": to_numpy(vecF_baseline.grad),
            "matH_target": to_numpy(matH_target),
            "matQ_target_grad": to_numpy(matQ_target.grad),
            "matK_target_grad": to_numpy(matK_target.grad),
            "matV_target_grad": to_numpy(matV_target.grad),
            "vecI_target_grad": to_numpy(vecI_target.grad),
            "vecF_target_grad": to_numpy(vecF_target.grad),
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
            matQ_baseline.grad,
            matK_baseline.grad,
            matV_baseline.grad,
            vecI_baseline.grad,
            vecF_baseline.grad,
            matH_target,
            matQ_target.grad,
            matK_target.grad,
            matV_target.grad,
            vecI_target.grad,
            vecF_target.grad,
        )
    else:
        return None
