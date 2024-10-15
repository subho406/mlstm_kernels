"""Contains the functions to compute the FLOPs for the mlstm kernels and blocks."""

from typing import Any
from collections.abc import Callable

from dataclasses import dataclass
from typing import Literal


def count_flops_fw_C(L, Nc, dqk, dv, Nh, factor_exp, factor_max, factor_mask) -> int:
    return (
        Nc
        * Nh
        * (
            3
            + 3 * L
            + factor_max
            + factor_exp * (1 + 2 * L)
            + 2 * dqk * dv
            + dqk
            + L * (3 * dqk * dv + 7 * dqk)
        )
    )


def count_flops_fw_H(L, Nc, dqk, dv, Nh, factor_exp, factor_max, factor_mask) -> int:
    return (
        Nc
        * Nh
        * (
            (L * (L + 1)) // 2
            + L * L * (7 + factor_mask + factor_max + factor_exp + 2 * dqk + 2 * dv)
            + (1 + L) * factor_max
        )
    )


def count_flops_mlstm_chunkwise_fw(
    L, Nc, dqk, dv, Nh, factor_exp, factor_max, factor_mask
) -> tuple[int, int, int]:
    flops_fw_C = count_flops_fw_C(
        L, Nc, dqk, dv, Nh, factor_exp, factor_max, factor_mask
    )
    flops_fw_H = count_flops_fw_H(
        L, Nc, dqk, dv, Nh, factor_exp, factor_max, factor_mask
    )
    flops_fw_total = flops_fw_C + flops_fw_H
    return flops_fw_total, flops_fw_C, flops_fw_H


def _count_ln_flops(d):
    # not sure about this
    # d for mean, d to subtract mean, d for variance, d for division
    return 4 * d


def count_flops_mlstm_v1_layer_fw(
    S,
    d,
    dqk,
    dv,
    Nh,
    chunk_size,
    factor_sig=1,
    factor_exp=1,
    factor_max=1,
    factor_mask=1,
    count_ln_flops: Callable[[int], int] = _count_ln_flops,
) -> int:
    L = chunk_size
    Nc = S // L
    qkv_outproj_flops = 2 * S * d * (dqk + 2 * dv + 2 * Nh)
    ogate_flops = 2 * S * dv * (factor_sig + d)
    mlstm_cell_flops = count_flops_mlstm_chunkwise_fw(
        L=L,
        Nc=Nc,
        dqk=dqk,
        dv=dv,
        Nh=Nh,
        factor_exp=factor_exp,
        factor_max=factor_max,
        factor_mask=factor_mask,
    )
    ln_flops = count_ln_flops(d) + count_ln_flops(dv)
    skip_flops = S * d
    return qkv_outproj_flops + ogate_flops + mlstm_cell_flops[0] + ln_flops + skip_flops


def count_flops_ffn_layer_fw(
    S, d, pf, factor_gelu=1, count_ln_flops: Callable[[int], int] = _count_ln_flops
) -> int:
    upproj_flops = 2 * S * d * d * pf + S * d * pf * factor_gelu
    downproj_flops = 2 * S * d * d * pf
    ln_flops = count_ln_flops(d)
    skip_flops = S * d
    return upproj_flops + downproj_flops + ln_flops + skip_flops


def count_flops_mlstm_v1_block_fw(
    S,
    d,
    dqk,
    dv,
    Nh,
    chunk_size=64,
    pf_ffn=4,
    factor_sig=1,
    factor_exp=1,
    factor_gelu=1,
    factor_max=1,
    factor_mask=1,
    count_ln_flops: Callable[[int], int] = _count_ln_flops,
    return_detailed_flops=False,
    **kwargs,
):
    mlstm_v1_layer_flops = count_flops_mlstm_v1_layer_fw(
        S=S,
        d=d,
        dqk=dqk,
        dv=dv,
        Nh=Nh,
        chunk_size=chunk_size,
        factor_sig=factor_sig,
        factor_exp=factor_exp,
        factor_max=factor_max,
        factor_mask=factor_mask,
        count_ln_flops=count_ln_flops,
    )
    ffn_layer_flops = count_flops_ffn_layer_fw(
        S=S, d=d, pf=pf_ffn, factor_gelu=factor_gelu, count_ln_flops=count_ln_flops
    )
    total_flops = mlstm_v1_layer_flops + ffn_layer_flops
    if return_detailed_flops:
        return total_flops, dict(
            mlstm_v1_layer_flops=mlstm_v1_layer_flops, ffn_layer_flops=ffn_layer_flops
        )
    total_linear_layer_flops = ffn_layer_flops
    total_mlstm_flops = mlstm_v1_layer_flops
    return total_flops, total_linear_layer_flops, total_mlstm_flops


def count_flops_mlstm_v2_block_fw(
    S,
    d,
    dqk,
    dv,
    Nh,
    qk_block_size=4,
    qk_pf=1,
    v_block_size=4,
    v_pf=1,
    conv1d_kernel_size=4,
    pf=2,
    chunk_size=64,
    factor_exp=1,
    factor_swish=1,
    factor_max=1,
    factor_mask=1,
    count_ln_flops: Callable[[int], int] = _count_ln_flops,
    return_detailed_flops=False,
    **kwargs,
):
    linear_layer_flops = (
        2 * S * d * d * pf * (1 + 2 * v_pf) + S * d * pf * v_pf * factor_swish
    )
    qkv_proj_flops = (
        2 * pf * d * (2 * S * qk_block_size * qk_pf + S * v_block_size * v_pf)
    )
    conv1d_flops = (
        2 * conv1d_kernel_size * (S + conv1d_kernel_size - 1) * pf * d
        + S * pf * d
        + S * pf * d * factor_swish
    )
    skip_ln_mlstm_flops = 3 * S * d * pf * v_pf + count_ln_flops(d * pf * v_pf)
    skip_ln_linear_layer_flops = S * d + count_ln_flops(d)

    mlstm_cell_flops = count_flops_mlstm_chunkwise_fw(
        L=chunk_size,
        Nc=S // chunk_size,
        dqk=dqk,
        dv=dv,
        Nh=Nh,
        factor_exp=factor_exp,
        factor_max=factor_max,
        factor_mask=factor_mask,
    )
    total_flops = (
        linear_layer_flops
        + qkv_proj_flops
        + conv1d_flops
        + skip_ln_mlstm_flops
        + skip_ln_linear_layer_flops
        + mlstm_cell_flops[0]
    )
    total_linear_layer_flops = linear_layer_flops + skip_ln_linear_layer_flops
    mlstm_cell_conv_flops = mlstm_cell_flops[0] + conv1d_flops + skip_ln_mlstm_flops
    if return_detailed_flops:
        return total_flops, dict(
            linear_layer_flops=linear_layer_flops,
            qkv_proj_flops=qkv_proj_flops,
            conv1d_flops=conv1d_flops,
            skip_ln_mlstm_flops=skip_ln_mlstm_flops,
            skip_ln_linear_layer_flops=skip_ln_linear_layer_flops,
            mlstm_cell_total_flops=mlstm_cell_flops[0],
            mlstm_cell_fw_C_flops=mlstm_cell_flops[1],
            mlstm_cell_fw_H_flops=mlstm_cell_flops[2],
        )
    return total_flops, total_linear_layer_flops, mlstm_cell_conv_flops


@dataclass
class FLOPsComputation:
    # inputs
    batch_size: int
    model_type: Literal["mlstm_v1", "mlstm_v2"]
    model_config: dict[str, Any]
    model_size_name: str | None = None
    model_tag: str | None = None

    # results
    total_flops: int | None = None
    linear_layer_flops: int | None = None
    mlstm_other_flops: int | None = None


def count_fw_flops(
    flop_computations: FLOPsComputation | list[FLOPsComputation],
) -> FLOPsComputation | list[FLOPsComputation]:
    out = []
    if isinstance(flop_computations, FLOPsComputation):
        flop_computations = [flop_computations]
    for flop_comp in flop_computations:
        if flop_comp.model_type == "mlstm_v1":
            total_flops, linear_layer_flops, mlstm_other_flops = (
                count_flops_mlstm_v1_block_fw(**flop_comp.model_config)
            )
        elif flop_comp.model_type == "mlstm_v2":
            # we have to multiply by 2 to account for the two layers in v1
            total_flops, linear_layer_flops, mlstm_other_flops = (
                count_flops_mlstm_v2_block_fw(**flop_comp.model_config)
            )
            total_flops *= 2
            linear_layer_flops *= 2
            mlstm_other_flops *= 2
        else:
            raise ValueError(f"Model type {flop_comp.model_type} not supported")

        flop_comp.total_flops = total_flops * flop_comp.batch_size
        flop_comp.linear_layer_flops = linear_layer_flops * flop_comp.batch_size
        flop_comp.mlstm_other_flops = mlstm_other_flops * flop_comp.batch_size
        out.append(flop_comp)

    if len(out) == 1:
        return out[0]
    return out
