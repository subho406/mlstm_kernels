
# test against parallel torch fwbw for H outputs

# test against recurrent sequence for last states

import pytest

import torch

from mlstm_kernels.test_utils import check_correctness, loss_layernorm_offset_quadratic

from ...common import test_session_folder


def template_torch_chunkwise_vs_torch_parallel_recurrent_sequence(
    S: int = 2048,
    L: int = 64, # chunk size
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
    test_folder_name: str = "torch_parallel_vs_torch_recurrent_sequence",
    save_dir: str = ".",
) -> bool:
    from mlstm_kernels.mlstm.parallel import mlstm_parallel_torch_autograd
    from mlstm_kernels.mlstm.chunkwise import mlstm_chunkwise_torch_autograd
    from mlstm_kernels.mlstm.chunkwise import mlstm_chunkwise_torch_ownbw

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

    matQ_cw_torch_ag = matQ.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    matK_cw_torch_ag = matK.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    matV_cw_torch_ag = matV.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    vecI_cw_torch_ag = vecI.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    vecF_cw_torch_ag = vecF.clone().to(dtype=test_dtype).detach().requires_grad_(True)

    matQ_cw_torch_obw = matQ.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    matK_cw_torch_obw = matK.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    matV_cw_torch_obw = matV.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    vecI_cw_torch_obw = vecI.clone().to(dtype=test_dtype).detach().requires_grad_(True)
    vecF_cw_torch_obw = vecF.clone().to(dtype=test_dtype).detach().requires_grad_(True)


    matH_p_torch_ag = mlstm_parallel_torch_autograd(
        q=matQ_p_torch_ag,
        k=matK_p_torch_ag,
        v=matV_p_torch_ag,
        i=vecI_p_torch_ag,
        f=vecF_p_torch_ag,
    )

    matH_cw_torch_ag = mlstm_chunkwise_torch_autograd(
        q=matQ_cw_torch_ag,
        k=matK_cw_torch_ag,
        v=matV_cw_torch_ag,
        i=vecI_cw_torch_ag,
        f=vecF_cw_torch_ag,
        chunk_size=L,
    )
    matH_cw_torch_obw = mlstm_chunkwise_torch_ownbw(
        q=matQ_cw_torch_obw,
        k=matK_cw_torch_obw,
        v=matV_cw_torch_obw,
        i=vecI_cw_torch_obw,
        f=vecF_cw_torch_obw,
        chunk_size=L,
    )

    # forward checks
    # Note we do not check the autograd backward pass as it must match if forward matches
    # (if pytorch is correct, which we assume)
    matH_ag_match = check_correctness(
        test_specifier="matH_ag",
        baseline=matH_p_torch_ag,
        target=matH_cw_torch_ag,
        atol=atol_fw,
        rtol=rtol_fw,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    matH_obw_match = check_correctness(
        test_specifier="matH_obw",
        baseline=matH_p_torch_ag,
        target=matH_cw_torch_obw,
        atol=atol_fw,
        rtol=rtol_fw,
        savepath=f"{save_dir}/{test_folder_name}",
    )

    loss_layernorm_offset_quadratic(matH_p_torch_ag).backward()
    loss_layernorm_offset_quadratic(matH_cw_torch_obw).backward()

    matQgrad_match = check_correctness(
        test_specifier="matQgrad_obw",
        baseline=matQ_p_torch_ag.grad,
        target=matQ_cw_torch_obw.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    matKgrad_match = check_correctness(
        test_specifier="matKgrad_obw",
        baseline=matK_p_torch_ag.grad,
        target=matK_cw_torch_obw.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    matVgrad_match = check_correctness(
        test_specifier="matVgrad_obw",
        baseline=matV_p_torch_ag.grad,
        target=matV_cw_torch_obw.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    vecIgrad_match = check_correctness(
        test_specifier="vecIgrad",
        baseline=vecI_p_torch_ag.grad,
        target=vecI_cw_torch_obw.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    vecFgrad_match = check_correctness(
        test_specifier="vecFgrad_obw",
        baseline=vecF_p_torch_ag.grad,
        target=vecF_cw_torch_obw.grad,
        atol=atol_fwbw,
        rtol=rtol_fwbw,
        savepath=f"{save_dir}/{test_folder_name}",
    )
    assert matH_ag_match
    assert matH_obw_match
    assert matQgrad_match
    assert matKgrad_match
    assert matVgrad_match
    assert vecIgrad_match
    assert vecFgrad_match

# def template_torch_chunkwise_vs_torch_recurrent_sequence_initial_last_states(
#     S: int = 2048,
#     B: int = 2,
#     NH: int = 3,
#     DHQK: int = 128,  # dim per head
#     DHHV: int = 256,
#     DTYPE=torch.float32,
#     DEVICE=torch.device("cuda:0"),
#     EPS: float = 1e-6,
#     atol_fw: float = 1e-3,
#     rtol_fw: float = 1e-2,
#     atol_fwbw: float = 1e-2,
#     rtol_fwbw: float = 1e-2,
#     seed: int = 0,
#     test_folder_name: str = "torch_parallel_vs_torch_recurrent_sequence",
#     save_dir: str = ".",
# ) -> bool:
#     from mlstm_kernels.mlstm.recurrent import mlstm_recurrent_sequence_torch_autograd
#     from mlstm_kernels.mlstm.chunkwise import mlstm_chunkwise_torch_autograd
#     from mlstm_kernels.mlstm.chunkwise import mlstm_chunkwise_torch_ownbw

#     torch.manual_seed(seed)
#     matQ = torch.randn((B, NH, S, DHQK), dtype=torch.float32, device=DEVICE)
#     matK = torch.randn((B, NH, S, DHQK), dtype=torch.float32, device=DEVICE)
#     matV = torch.randn((B, NH, S, DHHV), dtype=torch.float32, device=DEVICE)
#     vecI = torch.randn((B, NH, S), dtype=torch.float32, device=DEVICE)
#     vecF = torch.randn((B, NH, S), dtype=torch.float32, device=DEVICE)

#     test_dtype = DTYPE
#     matQ_p_torch_ag = matQ.clone().to(dtype=test_dtype).detach().requires_grad_(True)
#     matK_p_torch_ag = matK.clone().to(dtype=test_dtype).detach().requires_grad_(True)
#     matV_p_torch_ag = matV.clone().to(dtype=test_dtype).detach().requires_grad_(True)
#     vecI_p_torch_ag = vecI.clone().to(dtype=test_dtype).detach().requires_grad_(True)
#     vecF_p_torch_ag = vecF.clone().to(dtype=test_dtype).detach().requires_grad_(True)

#     matQ_rseq_torch_ag = matQ.clone().to(dtype=test_dtype).detach().requires_grad_(True)
#     matK_rseq_torch_ag = matK.clone().to(dtype=test_dtype).detach().requires_grad_(True)
#     matV_rseq_torch_ag = matV.clone().to(dtype=test_dtype).detach().requires_grad_(True)
#     vecI_rseq_torch_ag = vecI.clone().to(dtype=test_dtype).detach().requires_grad_(True)
#     vecF_rseq_torch_ag = vecF.clone().to(dtype=test_dtype).detach().requires_grad_(True)

#     matH_p_torch_ag = mlstm_parallel_torch_autograd(
#         q=matQ_p_torch_ag,
#         k=matK_p_torch_ag,
#         v=matV_p_torch_ag,
#         i=vecI_p_torch_ag,
#         f=vecF_p_torch_ag,
#     )
#     (
#         matH_rseq_torch_ag,
#         (matC_last_rseq_torch_ag, vecN_last_rseq_torch_ag, scaM_last_rseq_torch_ag),
#     ) = mlstm_recurrent_sequence_torch_autograd(
#         q=matQ_rseq_torch_ag,
#         k=matK_rseq_torch_ag,
#         v=matV_rseq_torch_ag,
#         i=vecI_rseq_torch_ag,
#         f=vecF_rseq_torch_ag,
#         return_last_states=True,
#     )

#     # forward checks
#     matH_match = check_correctness(
#         test_specifier="matH",
#         baseline=matH_p_torch_ag,
#         target=matH_rseq_torch_ag,
#         atol=atol_fw,
#         rtol=rtol_fw,
#         savepath=f"{save_dir}/{test_folder_name}",
#     )

#     loss_layernorm_offset_quadratic(matH_p_torch_ag).backward()
#     loss_layernorm_offset_quadratic(matH_rseq_torch_ag).backward()

#     matQgrad_match = check_correctness(
#         test_specifier="matQgrad",
#         baseline=matQ_p_torch_ag.grad,
#         target=matQ_rseq_torch_ag.grad,
#         atol=atol_fwbw,
#         rtol=rtol_fwbw,
#         savepath=f"{save_dir}/{test_folder_name}",
#     )
#     matKgrad_match = check_correctness(
#         test_specifier="matKgrad",
#         baseline=matK_p_torch_ag.grad,
#         target=matK_rseq_torch_ag.grad,
#         atol=atol_fwbw,
#         rtol=rtol_fwbw,
#         savepath=f"{save_dir}/{test_folder_name}",
#     )
#     matVgrad_match = check_correctness(
#         test_specifier="matVgrad",
#         baseline=matV_p_torch_ag.grad,
#         target=matV_rseq_torch_ag.grad,
#         atol=atol_fwbw,
#         rtol=rtol_fwbw,
#         savepath=f"{save_dir}/{test_folder_name}",
#     )
#     vecIgrad_match = check_correctness(
#         test_specifier="vecIgrad",
#         baseline=vecI_p_torch_ag.grad,
#         target=vecI_rseq_torch_ag.grad,
#         atol=atol_fwbw,
#         rtol=rtol_fwbw,
#         savepath=f"{save_dir}/{test_folder_name}",
#     )
#     vecFgrad_match = check_correctness(
#         test_specifier="vecFgrad",
#         baseline=vecF_p_torch_ag.grad,
#         target=vecF_rseq_torch_ag.grad,
#         atol=atol_fwbw,
#         rtol=rtol_fwbw,
#         savepath=f"{save_dir}/{test_folder_name}",
#     )
#     assert matH_match
#     assert matQgrad_match
#     assert matKgrad_match
#     assert matVgrad_match
#     assert vecIgrad_match
#     assert vecFgrad_match



combinations_short = {
    "S": [32, 32, 32, 64],
    "L": [16, 16, 16, 32],
    "B": [1, 1, 2, 1],
    "NH": [1, 1, 3, 1],
    "DHQK": [16, 16, 16, 16],
    "DHHV": [16, 32, 16, 16],
}
combinations_short_list = [values for values in zip(*combinations_short.values())]

combinations_long = {
    "S": [512],
    "L": [64],
    "B": [1],
    "NH": [1],
    "DHQK": [128],
    "DHHV": [128],
}
combinations_long_list = [values for values in zip(*combinations_long.values())]


class TestRecurrentVsParallelTorch:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "L", "B", "NH", "DHQK", "DHHV"], combinations_short_list)
    def test_recurrent_vs_parallel_short_fp32(
        self, test_session_folder, S, L, B, NH, DHQK, DHHV
    ):
        params = f"S{S}L{L}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}"
        print(params)
        template_torch_chunkwise_vs_torch_parallel_recurrent_sequence(
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            DTYPE=torch.float32,
            atol_fw=1e-3,
            rtol_fw=1e-2,
            atol_fwbw=1e-2,
            rtol_fwbw=1e-2,
            test_folder_name=f"torch_parallel_vs_torch_recurrent_sequence_{params}",
            save_dir=str(test_session_folder),
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    @pytest.mark.parametrize(["S", "L", "B", "NH", "DHQK", "DHHV"], combinations_long_list)
    def test_recurrent_vs_parallel_long_fp32(
        self, test_session_folder, S, L, B, NH, DHQK, DHHV
    ):
        params = f"S{S}L{L}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}"
        print(params)
        template_torch_chunkwise_vs_torch_parallel_recurrent_sequence(
            S=S,
            B=B,
            NH=NH,
            DHQK=DHQK,
            DHHV=DHHV,
            DTYPE=torch.float32,
            atol_fw=1e-3,
            rtol_fw=1e-2,
            atol_fwbw=1e-2,
            rtol_fwbw=1e-2,
            test_folder_name=f"torch_parallel_vs_torch_recurrent_sequence_{params}",
            save_dir=str(test_session_folder),
        )

    # @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    # @pytest.mark.parametrize(["S", "L", "B", "NH", "DHQK", "DHHV"], combinations_short_list)
    # def test_recurrent_vs_parallel_short_fp16(
    #     self, test_session_folder, S, L, B, NH, DHQK, DHHV
    # ):
    #     params = f"S{S}L{L}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}"
    #     print(params)
    #     template_torch_chunkwise_vs_torch_parallel_recurrent_sequence(
    #         S=S,
    #         B=B,
    #         NH=NH,
    #         DHQK=DHQK,
    #         DHHV=DHHV,
    #         DTYPE=torch.float16,
    #         atol_fw=1e-1,
    #         rtol_fw=1e-1,
    #         atol_fwbw=5e-1,
    #         rtol_fwbw=5e-1,
    #         test_folder_name=f"torch_parallel_vs_torch_recurrent_sequence_{params}",
    #         save_dir=str(test_session_folder),
    #     )

    # @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available.")
    # @pytest.mark.parametrize(["S", "L", "B", "NH", "DHQK", "DHHV"], combinations_long_list)
    # def test_recurrent_vs_parallel_long_fp16(
    #     self, test_session_folder, S, L, B, NH, DHQK, DHHV
    # ):
    #     params = f"S{S}L{L}B{B}NH{NH}DHQK{DHQK}DHHV{DHHV}"
    #     print(params)
    #     template_torch_chunkwise_vs_torch_parallel_recurrent_sequence(
    #         S=S,
    #         B=B,
    #         NH=NH,
    #         DHQK=DHQK,
    #         DHHV=DHHV,
    #         DTYPE=torch.float16,
    #         atol_fw=3.,
    #         rtol_fw=1.,
    #         atol_fwbw=3.5,
    #         rtol_fwbw=1.,
    #         test_folder_name=f"torch_parallel_vs_torch_recurrent_sequence_{params}",
    #         save_dir=str(test_session_folder),
    #     )
