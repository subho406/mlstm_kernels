import sys

sys.path.append("../..")
import os

import argparse

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
import torch

from mlstm_kernels.components.ln import MultiHeadLayerNorm
from mlstm_kernels.mlstm.parallel import mlstm_torch_autograd
from mlstm_kernels.mlstm.recurrent._torch_fw_legacy import (
    mlstm_recurrent_sequence_stabilized,
)
from mlstm_kernels.mlstm.recurrent.torch_fw import (
    recurrent_step_fw as recurrent_step_fw_torch,
)
from mlstm_kernels.mlstm.recurrent.triton_fw import (
    recurrent_step_fw as recurrent_step_fw_triton,
)
from mlstm_kernels.mlstm.recurrent.triton_fused_fw import (
    recurrent_step_fw as recurrent_step_fw_triton_fused,
)
from tqdm import tqdm
import triton


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel", type=str, required=True, help="Kernel to benchmark."
    )

    args = parser.parse_args()

    kernel = args.kernel
    print(f"Kernel: {kernel}")
    if kernel == "torch":
        kernel_fn = recurrent_step_fw_torch
    elif kernel == "torch_compile":
        kernel_fn = torch.compile(recurrent_step_fw_torch)
    elif kernel == "triton":
        kernel_fn = recurrent_step_fw_triton
    elif kernel == "triton_fused":
        kernel_fn = recurrent_step_fw_triton_fused
    else:
        raise ValueError(f"Kernel {kernel} not recognized!")

    # params
    S = 12  # seq len
    B = 6  # batch size
    NH = 4  # num heads
    DHQK = 1024  # dim per head
    DHV = 1024  # DHQK

    DTYPE = torch.float16
    DEVICE = torch.device("cuda:0")
    EPS = 0.0
    torch.manual_seed(0)
    matC_old = torch.zeros((B, NH, DHQK, DHV), dtype=DTYPE, device=DEVICE)
    vecN_old = torch.zeros((B, NH, DHQK), dtype=DTYPE, device=DEVICE)
    scaM_old = torch.zeros((B, NH, 1), dtype=DTYPE, device=DEVICE)

    vecQ = torch.randn((B, NH, DHQK), dtype=DTYPE, device=DEVICE)
    vecK = torch.randn((B, NH, DHQK), dtype=DTYPE, device=DEVICE)
    vecV = torch.randn((B, NH, DHV), dtype=DTYPE, device=DEVICE)
    scaI = torch.randn((B, NH, 1), dtype=DTYPE, device=DEVICE)
    scaF = torch.randn((B, NH, 1), dtype=DTYPE, device=DEVICE)

    warmup_iters = 5
    main_iters = 1000

    dt_state = torch.float16

    if kernel == "torch" or kernel == "torch_compile":
        inputs = (matC_old, vecN_old, scaM_old, vecQ, vecK, vecV, scaI, scaF)

    else:
        inputs = (matC_old, vecN_old, scaM_old, vecQ, vecK, vecV, scaI, scaF, dt_state)

    print(f"warmup")
    for i in tqdm(range(warmup_iters)):
        h_out_pt, (matC_new_pt, vecN_new_pt, scaM_new_pt) = kernel_fn(*inputs)

    # print(f"main")
    # for i in tqdm(range(main_iters)):
    #     h_out_pt, (matC_new_pt, vecN_new_pt, scaM_new_pt) = recurrent_step_fw_triton_fused(matC_old, vecN_old, scaM_old, vecQ, vecK, vecV, scaI, scaF, DTYPE=dt_state, BLOCK_DQK=BLOCK_DQK, BLOCK_DV=BLOCK_DV)
