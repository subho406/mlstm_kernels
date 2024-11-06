import sys

sys.path.append("../..")
import argparse
import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
from mlstm_kernels.mlstm.chunkwise.max_triton_fwbw_v5xlchunksize._triton_combine_recurrent_parallel import (
    mlstm_chunkwise_fw,
)

import torch
import triton
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

    args = parser.parse_args()

    kernel = args.kernel
    print(f"Kernel: {kernel}")
    if kernel == "max_triton_v5xlchunksize":
        kernel_fn = mlstm_chunkwise_fw
    else:
        raise ValueError(f"Kernel {kernel} not recognized!")

    # params
    S = 1024  # 32  # seq len
    B = 1  # batch size
    NH = 4  # num heads
    DHQK = 512  # dim per head
    DHV = 512  # dim per head
    SIZ_B_DHQK = 64
    SIZ_B_DHV = 128

    CHUNK_SIZE = 256  # S  # 16  # S
    NC = S // CHUNK_SIZE
    SIZ_B_LQ = 64
    SIZ_B_LKV = SIZ_B_LQ

    NUM_WARPS_INTRA = 4

    DTYPE = torch.bfloat16
    OUTPUT_DTYPE = torch.float32
    DEVICE = torch.device("cuda")
    EPS = 0.0
    torch.manual_seed(0)
    matQ_p_tr = torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE)
    matK_p_tr = torch.randn((B, NH, S, DHQK), dtype=DTYPE, device=DEVICE)
    matV_p_tr = torch.randn((B, NH, S, DHV), dtype=DTYPE, device=DEVICE)
    vecI_p_tr = torch.randn((B, NH, S), dtype=DTYPE, device=DEVICE)
    vecF_p_tr = 5.0 + torch.randn((B, NH, S), dtype=DTYPE, device=DEVICE)

    warmup_iters = 5
    main_iters = 5

    dt_state = torch.float32
    inputs = dict(
        matQ=matQ_p_tr,
        matK=matK_p_tr,
        matV=matV_p_tr,
        vecI=vecI_p_tr,
        vecF=vecF_p_tr,
        eps=EPS,
        chunk_size_intra=CHUNK_SIZE,
        chunk_size_inter=64,
        siz_b_LQ=SIZ_B_LQ,
        siz_b_LKV=SIZ_B_LKV,
        siz_b_DHQK=SIZ_B_DHQK,
        siz_b_DHHV=SIZ_B_DHV,
        num_warps_intra=NUM_WARPS_INTRA,
        output_dtype=DTYPE,
    )

    for i in tqdm(range(warmup_iters), desc="warmup"):
        torch.cuda.nvtx.range_push(f"iter-{i}")
        _ = kernel_fn(**inputs)
        torch.cuda.nvtx.range_pop()

    # for i in tqdm(range(main_iters)):
    #     torch.cuda.nvtx.range_push(f"iter-{i}")
    #     h_out_pt, (matC_new_pt, vecN_new_pt, scaM_new_pt) = kernel_fn(**inputs)
    #     torch.cuda.nvtx.range_pop()
