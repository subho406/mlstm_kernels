import logging
import torch
from collections.abc import Callable
from mlstm_kernels.test_utils import check_correctness, loss_layernorm_offset_quadratic
from mlstm_kernels.time_utils import Stopwatch


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
    dtype=torch.float32,
    baseline_dtype=None,  # if None, then use DTYPE
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
) -> bool:
    """This is a generic test function that tests the parallel interface of the mlstm.
    It tests the outputs and gradients.
    """

    LOGGER.info(
        f"Test {test_folder_name_prefix} target={target_name} vs. baseline={baseline_name} with S={S}, B={B}, NH={NH}, DHQK={DHQK}, DHHV={DHHV}, DTYPE={dtype}"
    )

    if baseline_dtype is None:
        baseline_dtype = dtype

    torch.manual_seed(seed)
    matQ = torch.randn((B, NH, S, DHQK), dtype=torch.float32, device=device)
    matK = torch.randn((B, NH, S, DHQK), dtype=torch.float32, device=device)
    matV = torch.randn((B, NH, S, DHHV), dtype=torch.float32, device=device)
    vecI = torch.randn((B, NH, S), dtype=torch.float32, device=device)
    vecF = torch.randn((B, NH, S), dtype=torch.float32, device=device)

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

    sw = Stopwatch()
    sw.start()
    matH_p_torch_ag = baseline_fn(
        q=matQ_baseline,
        k=matK_baseline,
        v=matV_baseline,
        i=vecI_baseline,
        f=vecF_baseline,
        eps=EPS,
    )
    fw_seconds = sw.lap()
    loss_layernorm_offset_quadratic(matH_p_torch_ag).backward()
    fwbw_seconds = sw.stop()

    print(
        f"{baseline_name} | fw (ms): {fw_seconds * 1000}, fwbw (ms): {fwbw_seconds * 1000}"
    )
    LOGGER.info(
        f"{baseline_name} | fw (ms): {fw_seconds * 1000}, fwbw (ms): {fwbw_seconds * 1000}"
    )

    sw = Stopwatch()
    sw.start()

    matH_rseq_torch_ag = target_fn(
        q=matQ_target,
        k=matK_target,
        v=matV_target,
        i=vecI_target,
        f=vecF_target,
        eps=EPS,
    )
    fw_seconds = sw.lap()
    loss_layernorm_offset_quadratic(matH_rseq_torch_ag).backward()
    fwbw_seconds = sw.stop()
    print(
        f"{target_name} | fw (ms): {fw_seconds * 1000}, fwbw (ms): {fwbw_seconds * 1000}"
    )
    LOGGER.info(
        f"{target_name} | fw (ms): {fw_seconds * 1000}, fwbw (ms): {fwbw_seconds * 1000}"
    )
    # forward checks
    matH_match = check_correctness(
        test_specifier="matH",
        baseline=matH_p_torch_ag,
        target=matH_rseq_torch_ag,
        atol=atol_fw,
        rtol=rtol_fw,
        vmax=vmax,
        max_num_batchhead_plots=max_num_batchhead_plots,
        savepath=f"{save_dir}/{test_folder_name}",
    )

    matQgrad_match = check_correctness(
        test_specifier="matQgrad",
        baseline=matQ_baseline.grad,
        target=matQ_target.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        vmax=vmax,
        max_num_batchhead_plots=max_num_batchhead_plots,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    matKgrad_match = check_correctness(
        test_specifier="matKgrad",
        baseline=matK_baseline.grad,
        target=matK_target.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        vmax=vmax,
        max_num_batchhead_plots=max_num_batchhead_plots,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    matVgrad_match = check_correctness(
        test_specifier="matVgrad",
        baseline=matV_baseline.grad,
        target=matV_target.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        vmax=vmax,
        max_num_batchhead_plots=max_num_batchhead_plots,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    vecIgrad_match = check_correctness(
        test_specifier="vecIgrad",
        baseline=vecI_baseline.grad,
        target=vecI_target.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        vmax=vmax,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    vecFgrad_match = check_correctness(
        test_specifier="vecFgrad",
        baseline=vecF_baseline.grad,
        target=vecF_target.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        vmax=vmax,
        max_num_batchhead_plots=max_num_batchhead_plots,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    assert matH_match
    assert matQgrad_match
    assert matKgrad_match
    assert matVgrad_match
    assert vecIgrad_match
    assert vecFgrad_match
