
import sys
sys.path.append("../..")
import os
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
import torch

from mlstm_kernels.components.ln import MultiHeadLayerNorm
from mlstm_kernels.mlstm.parallel import mlstm_torch_autograd
from mlstm_kernels.mlstm.recurrent._torch_fw_legacy import mlstm_recurrent_sequence_stabilized
from mlstm_kernels.mlstm.recurrent.torch_fw import recurrent_step_fw as recurrent_step_fw_torch
from mlstm_kernels.mlstm.recurrent.triton_fw import recurrent_step_fw as recurrent_step_fw_triton
from mlstm_kernels.mlstm.recurrent.triton_fused_fw import recurrent_step_fw as recurrent_step_fw_triton_fused
from tqdm import tqdm
import triton


if __name__ == "__main__":
    # params
    S = 12 # seq len
    B = 6 # batch size
    NH = 4 # num heads
    DHQK = 1024 # dim per head
    DHV = 1024 #DHQK

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

    BLOCK_DQK: int = 64
    BLOCK_DV: int = 64

    warmup_iters = 5
    main_iters = 1000

    dt_state = torch.float16

    print(f"warmup")
    for i in tqdm(range(warmup_iters)):
        h_out_pt, (matC_new_pt, vecN_new_pt, scaM_new_pt) = recurrent_step_fw_triton_fused(matC_old, vecN_old, scaM_old, vecQ, vecK, vecV, scaI, scaF, DTYPE=dt_state, BLOCK_DQK=BLOCK_DQK, BLOCK_DV=BLOCK_DV)

    # print(f"main")
    # for i in tqdm(range(main_iters)):
    #     h_out_pt, (matC_new_pt, vecN_new_pt, scaM_new_pt) = recurrent_step_fw_triton_fused(matC_old, vecN_old, scaM_old, vecQ, vecK, vecV, scaI, scaF, DTYPE=dt_state, BLOCK_DQK=BLOCK_DQK, BLOCK_DV=BLOCK_DV)
