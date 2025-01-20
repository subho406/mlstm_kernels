#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FONTSIZE_DELTA = 2.5
FONTSIZE = 12
SMALL_OFFSET = 1
FONTSIZE_SMALL = FONTSIZE - SMALL_OFFSET
FONTSIZE_TICKS = FONTSIZE_SMALL

MARKERSIZE = 6.0
LINEWIDTH = 2.0  # default 1.5

FIGSIZE = (2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54)
FIGSIZE_2COL = (4 * 0.7 * 12 * 1 / 2.54, 2 * 0.7 * 8 * 1 / 2.54)

GRIDSPEC_KWARGS = {"wspace": 0.115, "hspace": 0}


def get_tb_plot_mpl_context(fontsize_delta: int = FONTSIZE_DELTA):
    return mpl.rc_context(
        rc={
            "text.usetex": False,
            "font.size": FONTSIZE + fontsize_delta,
            "axes.labelsize": FONTSIZE + fontsize_delta,
            "legend.fontsize": FONTSIZE_SMALL + fontsize_delta,
            "xtick.labelsize": FONTSIZE_TICKS + fontsize_delta,
            "ytick.labelsize": FONTSIZE_TICKS + fontsize_delta,
            "axes.titlesize": FONTSIZE + fontsize_delta,
            "lines.markersize": MARKERSIZE,
            "lines.linewidth": LINEWIDTH,
        }
    )


def savefig(fig, filename: str):
    dir = Path("./plots/")
    dir.mkdir(parents=True, exist_ok=True)

    if filename is not None:
        for file_ending in ["png", "pdf", "svg"]:
            file = Path(f"./plots/plot_{filename}.{file_ending}")
            fig.savefig(file, dpi=300, bbox_inches="tight", pad_inches=-0.0020)


kernel_colors = {
    "torch_flashattn": "#165b89",
    "torch_cudnn": "#439ebd",
    "flashattn3": "#80a8b3",
    "mamba2_noconv": "#d08814",
    "mamba2": "#d08814",
    "mamba": "#ffd449",
    "chunk_simple_gla": "#4fb72e",
    "fused_chunk_gla": "#548c2f",
    "chunk_gla": "#548c2f",
    "mlstmexp_torch_native": "#e52e66",
    "mlstmexp_triton_xl_chunk": "#e52e66",
    "mlstmexp_triton_limit_chunk": "#f0acb9",
    "mlstmsig_triton_xl_chunk": "#9a3c73",
}


kernel_labels = {
    "torch_flashattn": "Torch FlashAttn",
    "torch_cudnn": "cuDNN FlashAttn",
    "flashattn3": "FlashAttn 3",
    "mamba2_noconv": "Mamba 2 SSD",
    "mamba2": "Mamba 2",
    "mamba": "Mamba",
    "chunk_simple_gla": "Simple GLA (FLA)",
    "fused_chunk_gla": "GLA (fused)",
    "chunk_gla": "GLA (FLA)",
    "mlstmexp_torch_native": "mLSTM (torch)",
    "mlstmexp_triton_xl_chunk": "mLSTMexp (TFLA XL chunk)",
    "mlstmexp_triton_limit_chunk": "mLSTMexp (FLA limit chunk)",
    "mlstmsig_triton_xl_chunk": "mLSTMsig (TFLA XL chunk)",
}

style_dict = {
    key: {"color": kernel_colors[key], "label": value}
    for key, value in kernel_labels.items()
}

col_order_consttoken = [
    "torch_flashattn",
    "torch_cudnn",
    "flashattn3",
    "chunk_gla",
    "mamba",
    "mamba2",
    # "chunk_simple_gla",
    "mlstmexp_triton_limit_chunk",
    "mlstmexp_triton_xl_chunk",
    "mlstmsig_triton_xl_chunk",
]


def map_consttoken_fwbw_data_col_to_plot_col_mapping(fwbw: bool) -> dict[str, str]:
    fwbw_str = "fwbw" if fwbw else "fw"
    return {
        f"torch_flash__bfloat16__{fwbw_str}__nh-32_hdq-128_hdv-128": "torch_flashattn",
        f"torch_cudnn__bfloat16__{fwbw_str}__nh-32_hdq-128_hdv-128": "torch_cudnn",
        f"flashattn3____bfloat16__{fwbw_str}__nh-32_hdq-128_hdv-128": "flashattn3",
        f"mamba____bfloat16__{fwbw_str}__nh-1_hdv-8192_hdq-16": "mamba",
        f"mamba2____bfloat16__{fwbw_str}__nh-128_hdv-64_hdq-64": "mamba2",
        f"chunk_gla____bfloat16__{fwbw_str}__nh-8_hdv-512_hdq-256": "chunk_gla",
        f"chunkwise--triton_limit_chunk__bfloat16__{fwbw_str}__cs-64_nh-8_hdv-512_hdq-256": "mlstmexp_triton_limit_chunk",
        f"chunkwise--triton_xl_chunk__bfloat16__{fwbw_str}__cs-128_nh-8_hdv-512_hdq-256": "mlstmexp_triton_xl_chunk",
        f"chunkwise--triton_xl_chunk_siging__bfloat16__{fwbw_str}__cs-128_nh-8_hdv-512_hdq-256_n-False": "mlstmsig_triton_xl_chunk",
        f"fused_chunk_gla__bfloat16__{fwbw_str}__nh-8_hdv-512_hdq-256": "fused_chunk_gla",
        f"chunk_simple_gla__bfloat16__{fwbw_str}__nh-8_hdv-512_hdq-256": "chunk_simple_gla",
    }


col_order_fwbw_ = []
col_order_fw = [
    "torch_flash__bfloat16__fw__nh-32_hdq-128_hdv-128",
    "torch_cudnn__bfloat16__fw__nh-32_hdq-128_hdv-128",
    "flashattn3____bfloat16__fw__nh-32_hdq-128_hdv-128",
    "mamba____bfloat16__fw__nh-1_hdv-8192_hdq-16",
    "mamba2____bfloat16__fw__nh-128_hdv-64_hdq-64",
    "chunk_gla____bfloat16__fw__nh-8_hdv-512_hdq-256",
    # "fused_chunk_gla__bfloat16__fw__nh-8_hdv-512_hdq-256",
    # "chunk_simple_gla__bfloat16__fw__nh-8_hdv-512_hdq-256",
    # "mamba2_noconv____bfloat16__fw__nh-64_hdv-64_hdq-64",
    "chunkwise--mlstmexp_triton_limit_chunk__bfloat16__fw__cs-64_nh-8_hdv-512_hdq-256",
    "chunkwise--mlstmexp_triton_xl_chunk__bfloat16__fw__cs-128_nh-8_hdv-512_hdq-256",
    # "chunkwise--native_custbw____bfloat16__fw__cs-128_nh-8_hdv-512_hdq-256",
    # "chunkwise--native_custbw____bfloat16__fw__cs-256_nh-8_hdv-512_hdq-256",
    # "chunkwise--native_custbw____bfloat16__fw__cs-64_nh-8_hdv-512_hdq-256",
    # "chunkwise--native_autograd____bfloat16__fw__cs-256_nh-8_hdv-512_hdq-256",
]

legend_order = [
    "Torch FlashAttn",
    "cuDNN FlashAttn",
    "FlashAttn 3",
    "GLA (FLA)",
    "Mamba",
    "Mamba 2",
    "mLSTMexp (FLA limit_chunk)",
    "mLSTMexp (TFLA xl_chunk)",
    "mLSTMsig (TFLA xl_chunk)",
]
