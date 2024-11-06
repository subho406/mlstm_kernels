import sys

sys.path.append("../..")
import argparse
import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
from mlstm_kernels.mlstm.chunkwise.max_triton_fwbw_v5xlchunksize.triton_fwbw import (
    mlstm_chunkwise_max_triton as mlstm_chunkwise_max_triton_v5,
)
from mlstm_kernels.test_utils.test_losses import (
    loss_layernorm_offset_quadratic as loss_fn,
)

import torch
import triton
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel",
        type=str,
        required=False,
        default="max_triton_v5xlchunksize",
        help="Kernel to benchmark.",
    )
    parser.add_argument(
        "--torch_profiler",
        action="store_true",
        required=False,
        help="whether to run torch profiler",
    )

    args = parser.parse_args()
    print(args)

    kernel = args.kernel
    print(f"Kernel: {kernel}")
    if kernel == "max_triton_v5xlchunksize":
        kernel_fn = mlstm_chunkwise_max_triton_v5
    else:
        raise ValueError(f"Kernel {kernel} not recognized!")

    # params
    S = 1024  # 32  # seq len
    B = 2  # batch size
    NH = 8  # num heads
    DHQK = 512  # dim per head
    DHV = 512  # dim per head
    SIZ_B_DH_PARALLEL = 128
    SIZ_B_DH_LOOP = 64

    CHUNK_SIZE = 256  # S  # 16  # S
    NC = S // CHUNK_SIZE
    SIZ_B_L_PARALLEL = 64
    SIZ_B_L_LOOP = 64  # SIZ_B_LQ

    DTYPE = torch.float32
    DEVICE = torch.device("cuda")
    EPS = 0.0

    LOSS_SEED = 0
    LN_EPS = 1e-5

    NUM_WARPS_INTRA = 4

    DTYPE = torch.bfloat16
    OUTPUT_DTYPE = torch.float32
    DEVICE = torch.device("cuda")
    EPS = 0.0
    torch.manual_seed(0)
    torch.manual_seed(1)
    matQ = torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE)
    matK = torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE)
    matV = torch.randn((B, NH, S, DHV), dtype=DTYPE, device=DEVICE)
    vecI = torch.randn((B, NH, S), dtype=DTYPE, device=DEVICE)
    vecF = 5.0 + torch.randn((B, NH, S), dtype=DTYPE, device=DEVICE)
    # vecI_shaped = rearrange(vecI, "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE)
    # vecB = rearrange(logsigmoid(vecF), "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE).cumsum(-1)

    matC_initial = torch.zeros((B, NH, DHQK, DHV), dtype=DTYPE, device=DEVICE)
    vecN_initial = torch.zeros((B, NH, DHQK), dtype=DTYPE, device=DEVICE)
    scaM_initial = torch.zeros((B, NH), dtype=DTYPE, device=DEVICE)

    matQ_tr_v5 = matQ.clone().detach().requires_grad_(True)
    matK_tr_v5 = matK.clone().detach().requires_grad_(True)
    matV_tr_v5 = matV.clone().detach().requires_grad_(True)
    vecI_tr_v5 = vecI.clone().detach().requires_grad_(True)
    vecF_tr_v5 = vecF.clone().detach().requires_grad_(True)

    matC_initial_tr_v5 = matC_initial.clone().detach().requires_grad_(True)
    vecN_initial_tr_v5 = vecN_initial.clone().detach().requires_grad_(True)
    scaM_initial_tr_v5 = scaM_initial.clone().detach().requires_grad_(True)

    warmup_iters = 25
    main_iters = 100

    dt_state = torch.float32
    inputs = dict(
        q=matQ_tr_v5,
        k=matK_tr_v5,
        v=matV_tr_v5,
        i=vecI_tr_v5,
        f=vecF_tr_v5,
        c_initial=matC_initial_tr_v5,
        n_initial=vecN_initial_tr_v5,
        m_initial=scaM_initial_tr_v5,
        chunk_size_inter=CHUNK_SIZE,
        chunk_size_intra=CHUNK_SIZE,
        siz_b_L_parallel=SIZ_B_L_PARALLEL,
        siz_b_L_loop=SIZ_B_L_LOOP,
        siz_b_DH_parallel=SIZ_B_DH_PARALLEL,
        siz_b_DH_loop=SIZ_B_DH_LOOP,
        num_warps_intra=NUM_WARPS_INTRA,
        eps=EPS,
        return_last_states=True,
    )
    # for i in tqdm(range(warmup_iters), desc="warmup"):
    #     torch.cuda.nvtx.range_push(f"iter-{i}")
    #     torch.cuda.nvtx.range_push(f"fw-{i}")
    #     (matH_tr_v5, _) = kernel_fn(**inputs)
    #     torch.cuda.nvtx.range_pop()
    #     loss_tr_v5 = loss_fn(matH_tr_v5, seed=LOSS_SEED, eps=LN_EPS)
    #     torch.cuda.nvtx.range_push(f"bw-{i}")
    #     loss_tr_v5.backward()
    #     torch.cuda.nvtx.range_pop()
    #     torch.cuda.nvtx.range_pop()

    # for i in tqdm(range(main_iters)):
    #     torch.cuda.nvtx.range_push(f"iter-{i}")
    #     h_out_pt, (matC_new_pt, vecN_new_pt, scaM_new_pt) = kernel_fn(**inputs)
    #     torch.cuda.nvtx.range_pop()

    if args.torch_profiler:
        for i in tqdm(range(warmup_iters), desc="warmup"):
            (matH_tr_v5, _) = kernel_fn(**inputs)
            loss_tr_v5 = loss_fn(matH_tr_v5, seed=LOSS_SEED, eps=LN_EPS)
            loss_tr_v5.backward()

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
        sort_by_keyword = str(DEVICE) + "_time_total"

        with profile(activities=activities, record_shapes=True) as prof:
            for i in tqdm(range(main_iters), desc="warmup"):
                with record_function("fw"):
                    (matH_tr_v5, _) = kernel_fn(**inputs)
                with record_function("loss"):
                    # loss_tr_v5 = loss_fn(matH_tr_v5, seed=LOSS_SEED, eps=LN_EPS)
                    loss_tr_v5 = matH_tr_v5.sum()
                with record_function("bw"):
                    loss_tr_v5.backward()

        print(
            prof.key_averages().table(
                sort_by=sort_by_keyword, row_limit=50, max_name_column_width=100
            )
        )
    else:
        for i in tqdm(range(warmup_iters), desc="warmup"):
            (matH_tr_v5, _) = kernel_fn(**inputs)
            loss_tr_v5 = loss_fn(matH_tr_v5, seed=LOSS_SEED, eps=LN_EPS)
            loss_tr_v5.backward()
