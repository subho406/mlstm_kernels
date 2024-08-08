import torch
from einops import rearrange
import torch.nn.functional as F
# Do we need to carry over the initial scaMinter value? Should work without it.

def _mlstm_chunkwise__recurrent_fw(
    matK: torch.Tensor, # (B, NH, S, DHQK)
    matV: torch.Tensor, # (B, NH, S, DHV)
    vecA: torch.Tensor, # (B, NH, NC, L)
    vecG: torch.Tensor, # (B, NH, NC)
    matC_states: torch.Tensor, # (B, NH, NC * DHQK, DHV)
    vecN_states: torch.Tensor, # (B, NH, NC * DHQK)
    scaMinter_states: torch.Tensor, # (B, NH, NC)
    initial_matC: torch.Tensor = None, # (B, NH, DHQK, DHV)
    initial_vecN: torch.Tensor = None, # (B, NH, DHQK)
    initial_scaMinter: torch.Tensor = None, # (B, NH)
    CHUNK_SIZE: int = 64,
    NUM_CHUNKS: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, NH, S, DHQK, DHV = *matK.shape, matV.shape[-1]
    _dtype, _device = matK.dtype, matK.device
    
    # initial states
    C_prev_k = torch.zeros((B, NH, DHQK, DHV), dtype=_dtype, device=_device) if initial_matC is None else initial_matC
    n_prev_k = torch.zeros((B, NH, DHQK), dtype=_dtype, device=_device) if initial_vecN is None else initial_vecN
    m_prev_k = torch.zeros((B, NH, 1), dtype=_dtype, device=_device) if initial_scaMinter is None else initial_scaMinter

    # store initial states in the states tensors
    # TODO from here

    vecA_max = vecA.max(-1).values

    for k in range(1, NUM_CHUNKS):

        # m_k
        m_a_k = vecA_max[:, :, k - 1]
        g_k = vecG[:, :, k - 1]
        m_k_inter = torch.max(g_k + m_prev_k, m_a_k)
        scaMinter_states[:, :, k] = m_k_inter

        # C_k
        matK_chunk = matK[:, :, (k - 1)*CHUNK_SIZE:k*CHUNK_SIZE, :].clone()
        matV_chunk = matV[:, :, (k - 1)*CHUNK_SIZE:k*CHUNK_SIZE, :].clone()
        a_k = vecA[:, :, k - 1, :].clone()

        matK_chunk_gated = matK_chunk * torch.exp(a_k - m_k_inter).unsqueeze(-1)

        C_k = (
            torch.exp(g_k + m_prev_k - m_k_inter) * C_prev_k
            + matK_chunk_gated.transpose(-2, -1) @ matV_chunk
        )
        
        matC_states[:, :, k*DHQK:(k+1)*DHQK, :] = C_k

        # n_k
        n_k = torch.exp(
            g_k + m_prev_k - m_k_inter
        ) * n_prev_k + matK_chunk_gated.transpose(-2, -1).sum(-1)
        
        vecN_states[:, :, k*DHQK:(k+1)*DHQK] = n_k

        # move to the next iteration
        m_prev_k = m_k_inter
        C_prev_k = C_k
        n_prev_k = n_k

    return matC_states, vecN_states, scaMinter_states

def _mlstm_chunkwise__parallel_fw(matQ: torch.Tensor, # (B, NH, S, DHQK)
                                  matK: torch.Tensor, # (B, NH, S, DHQK)
                                  matV: torch.Tensor, # (B, NH, S, DHV)
                                  matC_states: torch.Tensor, # (B, NH, NC * DHQK, DHV)
                                  vecN_states: torch.Tensor, # (B, NH, NC * DHQK)
                                  scaMinter_states: torch.Tensor, # (B, NH, NC)
                                  
                                  ):
    pass



def mlstm_chunkwise_fw(
    matQ: torch.Tensor,
    matK: torch.Tensor,
    matV: torch.Tensor,
    vecI: torch.Tensor,
    vecF: torch.Tensor,
    seq_chunk_size: int = 64,
    detach_denominator: bool = False,
    return_last_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    B, NH, S, DHQK = matQ.shape
    DHV = matV.shape[-1]
    NC = S // seq_chunk_size
    L = seq_chunk_size

    _dtype, _device = matQ.dtype, matQ.device
    # _, _, NC, L, _ = matQ.shape

    matQ = matQ * DHQK**-0.5

    vecI = rearrange(vecI, "b nh (nc l) -> b nh nc l", l=seq_chunk_size)
    vecF = rearrange(vecF, "b nh (nc l) -> b nh nc l", l=seq_chunk_size)

    # compute the gates, the g and the a and b vectors
    vecF_logsig = F.logsigmoid(vecF)

    vecB_f_cs = vecF_logsig.cumsum(-1)
    vecA_f_rcs = vecF_logsig.sum(-1, keepdim=True) - vecB_f_cs

    vecB = vecB_f_cs
    vecA = vecA_f_rcs + vecI
    vecG = vecF_logsig.sum(-1)

    #! loop 1: materialize the  C_k, n_k, m_k states
    matC_k_states, vecN_k_states, scaMinter_k_states = _mlstm_chunkwise__recurrent_fw(
        matQ=matQ,
        matK=matK,
        matV=matV,
        vecA=vecA,
        vecG=vecG,
        matC_states=torch.zeros((B, NH, NC * DHQK, DHV), dtype=_dtype, device=_device),
        vecN_states=torch.zeros((B, NH, NC * DHQK), dtype=_dtype, device=_device),
        scaMinter_states=torch.zeros((B, NH, NC), dtype=_dtype, device=_device),
        qk_scale=DHQK**-0.5,
        CHUNK_SIZE=seq_chunk_size,
        NUM_CHUNKS=NC,
    )


    matC_k_states = rearrange(matC_k_states, "b nh (nc dhqk) dhv -> b nh nc dhqk dhv", nc=NC)
    vecN_k_states = rearrange(vecN_k_states, "b nh (nc dhqk) -> b nh nc dhqk", nc=NC)
    # scaMinter_k_states = rearrange(scaMinter_k_states, "b nh nc -> b nh nc")

    matQ = rearrange(matQ, "b nh (nc l) dh -> b nh nc l dh", l=seq_chunk_size)
    matK = rearrange(matK, "b nh (nc l) dh -> b nh nc l dh", l=seq_chunk_size)
    matV = rearrange(matV, "b nh (nc l) dh -> b nh nc l dh", l=seq_chunk_size)
    
    ltr = torch.tril(
        torch.ones(
            (L, L),
            dtype=torch.bool,
            device=_device,
        )
    )

    #! compute the H_states in parallel

    # ? Compute intra chunk contribution: H_intra
    vecF_logsig_cs_chunk = vecF_logsig.cumsum(-1)

    matF_logsig_chunk = (
        vecF_logsig_cs_chunk[:, :, :, :, None] - vecF_logsig_cs_chunk[:, :, :, None, :]
    )

    matF_logsig_mask_chunk = torch.where(ltr, matF_logsig_chunk, -float("inf"))

    matLogD_chunk = matF_logsig_mask_chunk + vecI[:, :, :, None, :]

    # max_state intra
    vecMintra_k = torch.max(
        matLogD_chunk, dim=-1, keepdim=False
    ).values  # (B, NH, NC, L)

    # max_state combined
    vecM_b_inter = vecB + scaMinter_k_states[:, :, :, None]  # (B, NH, NC, L)
    vecM_k_inter_intra = torch.maximum(vecM_b_inter, vecMintra_k)  # (B, NH, NC, L)

    vecM_k_inter_intra = vecM_k_inter_intra[:, :, :, :, None]  # (B, NH, NC, L, 1)
    vecM_b_inter = vecM_b_inter[:, :, :, :, None]  # (B, NH, NC, L, 1)

    matLogD_stabilized_chunk = matLogD_chunk - vecM_k_inter_intra
    matD_chunk = torch.exp(matLogD_stabilized_chunk)

    matS_chunk = (matQ @ matK.transpose(-2, -1)) / (DHQK**-0.5)

    matM_chunk = matS_chunk * matD_chunk

    # ? Combine H_intra with H_inter
    matQ_chunk_gated = matQ * torch.exp(vecM_b_inter - vecM_k_inter_intra)

    numerator_common = matQ_chunk_gated @ matC_k_states + matM_chunk @ matV

    denom_common = matQ_chunk_gated @ vecN_k_states.unsqueeze(-1) + matM_chunk.sum(
        dim=-1, keepdim=True
    )

    max_denom_common = torch.maximum(
        torch.abs(denom_common), torch.exp(-vecM_k_inter_intra)
    )

    if detach_denominator:
        max_denom_common = max_denom_common.detach()

    matH_k_chunk = numerator_common / max_denom_common

    H_out = rearrange(matH_k_chunk, "b nh nc l dh -> b nh (nc l) dh")

    if return_last_states:
        return H_out, (matC_k_states, vecN_k_states)

    return H_out
