import torch
from torch.amp import custom_fwd, custom_bwd
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional
from einops import rearrange
from ....kernel_utils import contiguous_noctx

"""Triton.

Forward and backward pass of the mLSTM chunkwise formulation.

Notation:
Dimensions:
    B: batch size
    NH: number of heads
    S: sequence length (K, V)
    T: sequence length (Q)
    DHQK: hidden dimension (Q, K)
    DHHV: hidden dimension (H, V)
    NC: number of chunks
    L: chunk size

Variables:
    vecA, a: forward gate contribution, contribution of forget gates from last chunk state C_{k-1} to current timestep t
    vecB, b: backward gate contribution, contribution of forget and input gates up to next chunk state C_k (form current timestep t)
    scaG, g: "go through" gate contribution, contribution of forget gates from C_{k-1} to C_k.
    matD, D: gating matrix for the parallel form.
"""


# Note: we only pass stride for the head dimension (we do not access individual batch elements directly)
@triton.jit
def _mlstm_chunkwise__recurrent_fw_C_kernel(
    matK,  # (B, NH, S, DHQK)
    matV,  # (B, NH, S, DHHV)
    vecB,  # (B, NH, NC, L)
    vecI,  # (B, NH, NC, L)
    matC_states,  # (B, NH, (NC + 1) * DHQK, DHHV)
    vecN_states,  # (B, NH, (NC + 1) * DHQK)
    scaMinter_states,  # (B, NH, (NC + 1))
    matC_initial,  # (B, NH, DHQK, DHHV)
    vecN_initial,  # (B, NH, DHQK)
    scaMinter_initial,  # (B, NH)
    qk_scale,
    str_matK_B_NH,
    str_matK_S,
    str_matK_DHQK,
    str_matV_B_NH,
    str_matV_S,
    str_matV_DHHV,
    str_vecBI_B_NH,
    str_vecBI_NC,
    str_vecBI_L,
    str_matCstates_B_NH,
    str_matCstates_NCDHQK,
    str_matCstates_DHHV,
    str_vecNstates_B_NH,
    str_vecNstates_NCDHQK,
    str_scaMinterstates_B_NH,
    str_scaMinterstates_NC,
    str_matCinitial_B_NH,
    str_matCinitial_DHQK,
    str_matCinitial_DHHV,
    str_vecNinitial_B_NH,
    str_vecNinitial_DHQK,
    str_scaMinterinitial_B_NH,
    B: tl.constexpr,
    NH: tl.constexpr,
    S: tl.constexpr,
    DHQK: tl.constexpr,
    DHHV: tl.constexpr,
    NC: tl.constexpr,
    L: tl.constexpr,
    siz_b_DHQK: tl.constexpr,
    siz_b_DHHV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    DTYPE: tl.constexpr = tl.float16,
):
    idx_b_DHQK, idx_b_DHHV, idx_b_BNH = (
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )

    # create running states in shared memory
    matC_val = tl.zeros((siz_b_DHQK, siz_b_DHHV), dtype=tl.float32)
    vecN_val = tl.zeros((siz_b_DHQK,), dtype=tl.float32)
    scaMinter_val = tl.zeros((1,), dtype=tl.float32)
    # scaMinter_next_val = tl.zeros((1,), dtype=tl.float32) # TODO we create this in the loop

    if USE_INITIAL_STATE:
        # each thread block loads a (siz_b_DHQK, siz_b_DHHV) block from matC_initial
        matCinitial_ptr = tl.make_block_ptr(
            base=matC_initial + idx_b_BNH * str_matCinitial_B_NH,
            shape=(DHQK, DHHV),
            strides=(str_matCinitial_DHQK, str_matCinitial_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(0, 1),
        )
        # each thread block loads a (siz_b_DHQK,) chunk from vecN_initial
        vecNinitial_ptr = (
            vecN_initial
            + idx_b_BNH * str_vecNinitial_B_NH
            + idx_b_DHQK * siz_b_DHQK
            + tl.arange(0, siz_b_DHQK)
        )
        # each thread block loads the scaMinter_initial
        scaMinterinitial_ptr = scaMinter_initial + idx_b_BNH * str_scaMinterinitial_B_NH
        
        # load initial states
        matC_val = tl.load(matCinitial_ptr, boundary_check=(0,1)).to(tl.float32)
        vecN_val = tl.load(vecNinitial_ptr).to(tl.float32)
        scaMinter_val = tl.load(scaMinterinitial_ptr).to(tl.float32)

    tl.static_print("matC_val", matC_val)
    tl.static_print("vecN_val", vecN_val)
    tl.static_print("scaMinter_val", scaMinter_val)
    
    # iterate over chunks
    for k in range(NC):
        # create pointer for matCstates_k, vecNstates_k, scaMinterstates_k
        # each thread block stores a (siz_b_DHQK, siz_b_DHHV) block to matC_states_k
        matCstates_k_ptr = tl.make_block_ptr(
            base=matC_states + idx_b_BNH * str_matCstates_B_NH + k * DHQK * DHHV,
            shape=(DHQK, DHHV),
            strides=(str_matCstates_NCDHQK, str_matCstates_DHHV),
            offsets=(idx_b_DHQK * siz_b_DHQK, idx_b_DHHV * siz_b_DHHV),
            block_shape=(siz_b_DHQK, siz_b_DHHV),
            order=(0, 1),
        )
        # TODO from here

        # store the states from the previous iteration
        # TODO

        # load / compute vecA_k, scaG_k
        # TODO

        # scaM_inter_k update

        # load matK_k, matV_k

        # matC_k update

        # vecN_k update

        # move to next iteration


    # store the states from the last iteration
    # TODO



def _mlstm_chunkwise__recurrent_fw_C(
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    matC_states: torch.Tensor = None,  # (B, NH, (NC + 1) * DHQK, DHHV)
    vecN_states: torch.Tensor = None,  # (B, NH, (NC + 1) * DHQK)
    scaMinter_states: torch.Tensor = None,  # (B, NH, (NC + 1)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaMinter_initial: torch.Tensor = None,  # (B, NH)
    qk_scale: float = None,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:  # matC_states (B, NH, (NC+1) * DHQK, DHHV), vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1))
    B, NH, S, DHQK = matK.shape
    DHHV = matV.shape[-1]

    NC = NUM_CHUNKS
    L = CHUNK_SIZE

    siz_b_DHQK = min(64, triton.next_power_of_2(DHQK))
    siz_b_DHHV = min(64, triton.next_power_of_2(DHHV))

    num_b_DHQK = triton.cdiv(DHQK, siz_b_DHQK)
    num_b_DHHV = triton.cdiv(DHHV, siz_b_DHHV)

    num_stages = 1
    num_warps = 4 if siz_b_DHQK == 64 else 2

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    USE_INITIAL_STATE = matC_initial is not None
    if USE_INITIAL_STATE:
        assert vecN_initial is not None and scaMinter_initial is not None
        str_matCinitial_B_NH = matC_initial.stride(1)
        str_matCinitial_DHQK = matC_initial.stride(2)
        str_matCinitial_DHHV = matC_initial.stride(3)
        str_vecNinitial_B_NH = vecN_initial.stride(1)
        str_vecNinitial_DHQK = vecN_initial.stride(2)
        str_scaMinterinitial_B_NH = scaMinter_initial.stride(1)
    else:
        str_matCinitial_B_NH = 0
        str_matCinitial_DHQK = 0
        str_matCinitial_DHHV = 0
        str_vecNinitial_B_NH = 0
        str_vecNinitial_DHQK = 0
        str_scaMinterinitial_B_NH = 0

    grid = (num_b_DHQK, num_b_DHHV, NUM_CHUNKS)

    # TODO make these tensors empty not zeros
    matC_states = (
        torch.zeros(B, NH, (NC + 1) * DHQK, DHHV, device=matK.device, dtype=matK.dtype)
        if matC_states is None
        else matC_states
    )
    vecN_states = (
        torch.zeros(B, NH, (NC + 1) * DHQK, device=matK.device, dtype=matK.dtype)
        if vecN_states is None
        else vecN_states
    )
    scaMinter_states = (
        torch.zeros(B, NH, (NC + 1), device=matK.device, dtype=matK.dtype)
        if scaMinter_states is None
        else scaMinter_states
    )

    _mlstm_chunkwise__recurrent_fw_C_kernel[grid](
        matK=matK,
        matV=matV,
        vecB=vecB,
        vecI=vecI,
        matC_states=matC_states,
        vecN_states=vecN_states,
        scaMinter_states=scaMinter_states,
        matC_initial=matC_initial,
        vecN_initial=vecN_initial,
        scaMinter_initial=scaMinter_initial,
        qk_scale=qk_scale,
        str_matK_B_NH=matK.stride(1),
        str_matK_S=matK.stride(2),
        str_matK_DHQK=matK.stride(3),
        str_matV_B_NH=matV.stride(1),
        str_matV_S=matV.stride(2),
        str_matV_DHHV=matV.stride(3),
        str_vecBI_B_NH=vecB.stride(1),
        str_vecBI_NC=vecB.stride(2),
        str_vecBI_L=vecB.stride(3),
        str_matCstates_B_NH=matC_states.stride(1),
        str_matCstates_NCDHQK=matC_states.stride(2),
        str_matCstates_DHHV=matC_states.stride(3),
        str_vecNstates_B_NH=vecN_states.stride(1),
        str_vecNstates_NCDHQK=vecN_states.stride(2),
        str_scaMinterstates_B_NH=scaMinter_states.stride(1),
        str_scaMinterstates_NC=scaMinter_states.stride(2),
        str_matCinitial_B_NH=str_matCinitial_B_NH,
        str_matCinitial_DHQK=str_matCinitial_DHQK,
        str_matCinitial_DHHV=str_matCinitial_DHHV,
        str_vecNinitial_B_NH=str_vecNinitial_B_NH,
        str_vecNinitial_DHQK=str_vecNinitial_DHQK,
        str_scaMinterinitial_B_NH=str_scaMinterinitial_B_NH,
        B=B,
        NH=NH,
        S=S,
        DHQK=DHQK,
        DHHV=DHHV,
        NC=NC,
        L=L,
        siz_b_DHQK=siz_b_DHQK,
        siz_b_DHHV=siz_b_DHHV,
        USE_INITIAL_STATE=USE_INITIAL_STATE,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    return matC_states, vecN_states, scaMinter_states


def _mlstm_chunkwise__parallel_fw(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    # these states must be all states up to the last chunk, i.e. :-1
    matC_states: torch.Tensor,  # (B, NH, NC * DHQK, DHHV)
    vecN_states: torch.Tensor,  # (B, NH, NC * DHQK)
    scaMinter_states: torch.Tensor,  # (B, NH, NC)
    vecI: torch.Tensor,  # (B, NH, NC, L)
    vecB: torch.Tensor,  # (B, NH, NC, L)
    qk_scale: float,
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
    EPS: float = 1e-6,
) -> tuple[
    torch.Tensor, torch.Tensor
]:  # matH_out (B, NH, S, DHHV), vecN_out (B, NH, S)
    """This function defines the grid and block sizes for the kernel launch and calls the kernel."""
    pass


@contiguous_noctx
def _mlstm_chunkwise_fw(
    matQ: torch.Tensor,  # (B, NH, S, DHQK)
    matK: torch.Tensor,  # (B, NH, S, DHQK)
    matV: torch.Tensor,  # (B, NH, S, DHHV)
    vecI: torch.Tensor,  # (B, NH, S)
    vecF: torch.Tensor,  # (B, NH, S)
    matC_initial: torch.Tensor = None,  # (B, NH, DHQK, DHHV)
    vecN_initial: torch.Tensor = None,  # (B, NH, DHQK)
    scaM_initial: torch.Tensor = None,  # (B, NH)
    qk_scale: float = None,
    return_last_states: bool = False,
    return_all_states: bool = False,
    CHUNK_SIZE: int = 64,
    EPS: float = 1e-6,
) -> tuple[
    torch.Tensor,  # matH_out (B, NH, S, DHHV)
    torch.Tensor,  # vecN_out (B, NH, S)
    Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],  # last_states (matC_states (B, NH, DHQK, DHHV), vecN_states (B, NH, DHQK), scaMinter_states (B, NH))
    Optional[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],  # all_states (matC_states (B, NH, (NC+1) * DHQK, DHHV), vecN_states (B, NH, (NC+1) * DHQK), scaMinter_states (B, NH, (NC+1)))
]:
    B, NH, S, DHQK = matQ.shape
    DHHV = matV.shape[-1]
    assert (
        S % CHUNK_SIZE == 0
    ), f"Sequence length {S} is not divisible by chunk size {CHUNK_SIZE}."
    NC = S // CHUNK_SIZE

    vecI = rearrange(vecI, "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE)
    vecF = rearrange(vecF, "b nh (nc l) -> b nh nc l", l=CHUNK_SIZE)

    # compute the gates, the g and the a and b vectors
    vecF_logsig = F.logsigmoid(vecF)

    vecB_f_cs = vecF_logsig.cumsum(-1)
    vecA_f_rcs = vecF_logsig.sum(-1, keepdim=True) - vecB_f_cs

    vecB = vecB_f_cs
    vecA = vecA_f_rcs + vecI
    scaG = vecF_logsig.sum(-1)

    if qk_scale is None:
        qk_scale = DHQK**-0.5

    #! materialize the  C_k, n_k, m_k states for each chunk
    matC_k_states, vecN_k_states, scaMinter_k_states = _mlstm_chunkwise__recurrent_fw_C(
        matK=matK,
        matV=matV,
        vecA=vecA,
        scaG=scaG,
        matC_initial=matC_initial,
        vecN_initial=vecN_initial,
        scaMinter_initial=scaM_initial,
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
    )

    #! compute the outputs within each chunk
    matH_out, vecN_out, vecM_out = _mlstm_chunkwise__parallel_fw(
        matQ=matQ,
        matK=matK,
        matV=matV,
        matC_states=matC_k_states[:, :, :-DHQK, :],
        vecN_states=vecN_k_states[:, :, :-DHQK],
        scaMinter_states=scaMinter_k_states[:, :, :-1],
        vecI=vecI,
        vecB=vecB,
        qk_scale=qk_scale,
        CHUNK_SIZE=CHUNK_SIZE,
        NUM_CHUNKS=NC,
        EPS=EPS,
    )

    ret_tuple = (
        matH_out,
        vecN_out,
        vecM_out,
    )
    if return_last_states:
        ret_tuple += (
            (
                matC_k_states[:, :, -DHQK:, :],
                vecN_k_states[:, :, -DHQK:],
                scaMinter_k_states[:, :, -1],
            ),
        )
    else:
        ret_tuple += (None,)

    if return_all_states:
        ret_tuple += ((matC_k_states, vecN_k_states, scaMinter_k_states),)
    else:
        ret_tuple += (None,)

    return ret_tuple  # (matH_out, vecN_out, vecM_out, optional(last_states), optional(all_states))
