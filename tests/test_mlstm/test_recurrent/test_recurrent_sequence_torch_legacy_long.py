import pytest
import logging
import torch

from ...common import test_session_folder

LOGGER = logging.getLogger(__name__)

import logging
import torch

from mlstm_kernels.test_utils import check_correctness, loss_layernorm_offset_quadratic


LOGGER = logging.getLogger(__name__)


def template_torch_parallel_vs_torch_recurrent_sequence(
    S: int = 2048,
    B: int = 2,
    NH: int = 3,
    DHQK: int = 128,  # dim per head
    DHHV: int = 256,
    DTYPE=torch.float32,
    DEVICE=torch.device("cuda:0"),
    EPS: float = 1e-6,
    atol_fw: float = 1e-3,
    rtol_fw: float = 1e-2,
    atol_fwbw: float = 1e-2,
    rtol_fwbw: float = 1e-2,
    seed: int = 0,
    vmax: float = None,
    max_num_batchhead_plots: int = 1, # -1 means all
    test_folder_name: str = "torch_parallel_vs_torch_recurrent_sequence",
    save_dir: str = ".",
) -> bool:
    from mlstm_kernels.mlstm.parallel import mlstm_parallel_torch_autograd

    #! We test the recurrent sequence with the legacy implementation
    from mlstm_kernels.mlstm.recurrent._torch_fw_legacy import (
        mlstm_recurrent_sequence_stabilized,
    )

    LOGGER.info(
        f"Running parallel vs. recurrent sequence test with S={S}, B={B}, NH={NH}, DHQK={DHQK}, DHHV={DHHV}, DTYPE={DTYPE}"
    )

    torch.manual_seed(seed)
    matQ = torch.randn((B, NH, S, DHQK), dtype=torch.float32, device=DEVICE)
    matK = torch.randn((B, NH, S, DHQK), dtype=torch.float32, device=DEVICE)
    matV = torch.randn((B, NH, S, DHHV), dtype=torch.float32, device=DEVICE)
    vecI = torch.randn((B, NH, S), dtype=torch.float32, device=DEVICE)
    vecF = torch.randn((B, NH, S), dtype=torch.float32, device=DEVICE)

    test_dtype = DTYPE
    matQ_p_torch_ag = matQ.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    matK_p_torch_ag = matK.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    matV_p_torch_ag = matV.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    vecI_p_torch_ag = vecI.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    vecF_p_torch_ag = vecF.clone().to(dtype=test_dtype).detach().requires_grad_(True)

    matQ_rseq_torch_ag = matQ.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    matK_rseq_torch_ag = matK.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    matV_rseq_torch_ag = matV.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    vecI_rseq_torch_ag = vecI.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    vecF_rseq_torch_ag = vecF.clone().to(dtype=test_dtype).detach().requires_grad_(True)

    matH_p_torch_ag = mlstm_parallel_torch_autograd(
        q=matQ_p_torch_ag,
        k=matK_p_torch_ag,
        v=matV_p_torch_ag,
        i=vecI_p_torch_ag,
        f=vecF_p_torch_ag,
        eps=EPS,
    )
    matH_rseq_torch_ag = mlstm_recurrent_sequence_stabilized(
        matQ_rseq_torch_ag,
        matK_rseq_torch_ag,
        matV_rseq_torch_ag,
        vecI_rseq_torch_ag.unsqueeze(-1),
        vecF_rseq_torch_ag.unsqueeze(-1),
        eps=EPS,
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

    loss_layernorm_offset_quadratic(matH_p_torch_ag).backward()
    loss_layernorm_offset_quadratic(matH_rseq_torch_ag).backward()

    matQgrad_match = check_correctness(
        test_specifier="matQgrad",
        baseline=matQ_p_torch_ag.grad,
        target=matQ_rseq_torch_ag.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        vmax=vmax,
        max_num_batchhead_plots=max_num_batchhead_plots,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    matKgrad_match = check_correctness(
        test_specifier="matKgrad",
        baseline=matK_p_torch_ag.grad,
        target=matK_rseq_torch_ag.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        vmax=vmax,
        max_num_batchhead_plots=max_num_batchhead_plots,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    matVgrad_match = check_correctness(
        test_specifier="matVgrad",
        baseline=matV_p_torch_ag.grad,
        target=matV_rseq_torch_ag.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        max_num_batchhead_plots=max_num_batchhead_plots,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    vecIgrad_match = check_correctness(
        test_specifier="vecIgrad",
        baseline=vecI_p_torch_ag.grad,
        target=vecI_rseq_torch_ag.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        vmax=vmax,
        max_num_batchhead_plots=max_num_batchhead_plots,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    vecFgrad_match = check_correctness(
        test_specifier="vecFgrad",
        baseline=vecF_p_torch_ag.grad,
        target=vecF_rseq_torch_ag.grad,
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


# combinations_long = {
#     "S": [512, 2048, 4096, 8192],
#     "B": [1, 1, 1, 1],
#     "NH": [1, 1, 1, 1],
#     "DHQK": [128, 128, 128, 128],
#     "DHHV": [128, 128, 128, 128],
# }
combinations_long = {
    "S": [128, 1024, 4096, 8192],
    "B": [1, 1, 1, 1],  # [2, 2, 2, 2],
    "NH": [1, 1, 1, 1],  # [3, 3, 3, 3],
    "DHQK": [16, 16, 16, 16],  # [5, 5, 5, 5],
    "DHHV": [16, 16, 16, 16],  # [5, 5, 5, 5],
}
combinations_long_list = [values for values in zip(*combinations_long.values())]


class TestRecurrentVsParallelTorchLong:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    # @pytest.mark.xfail(reason="Fails due to numerical instability")
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
