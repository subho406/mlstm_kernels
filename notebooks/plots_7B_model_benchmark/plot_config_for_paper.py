#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import matplotlib as mpl

FONTSIZE = 12
SMALL_OFFSET = 1
FONTSIZE_SMALL = FONTSIZE - SMALL_OFFSET
FONTSIZE_TICKS = FONTSIZE_SMALL
fontsize_delta = 0

MARKERSIZE = 6.0
LINEWIDTH = 2.0  # default 1.5

nice_look_term = 0
half_of_icml_width = (
    6.75 / 2 - 0.25 / 2 - 1
)  # ICML width of page: 6.75 inches + 0.25 inches between columns
factor = 5.5
desired_aspect_ratio = 2
FIGSIZE_2COL = (
    factor * half_of_icml_width,
    factor * half_of_icml_width / desired_aspect_ratio,
)
one_col_fig_size_factor = 1
FIGSIZE = (
    one_col_fig_size_factor * (FIGSIZE_2COL[0]),
    one_col_fig_size_factor * (FIGSIZE_2COL[1] * 2),
)

GRIDSPEC_KWARGS = {"wspace": 0.115, "hspace": 0}

model_colors = {
    "mlstm_simple": mpl.colormaps["tab10"].colors[0],
    "xlstm": mpl.colormaps["tab10"].colors[1],
    "llama2": mpl.colormaps["tab10"].colors[2],
    "llama3": mpl.colormaps["tab10"].colors[3],
    "ministral8b": mpl.colormaps["tab10"].colors[5],
    "codestral_mamba": mpl.colormaps["tab10"].colors[6],
    "falcon_mamba": mpl.colormaps["tab10"].colors[4],
    "zamba2": mpl.colormaps["tab10"].colors[7],
}

xlstm_colors = {
    "llama3": "#165b89ff",
    "llama2": "#80a8b3ff",
    # "xLSTM": "#cc4391ff",
    "xlstm": "#861657ff",
    "codestral_mamba": "#d08814ff",
    "falcon_mamba": "#ffd449ff",
    "RWKV4": "#145815ff",
}

model_labels = {
    "mlstm_simple": "mLSTM simple",
    "xlstm": "xLSTM 7B",
    "llama2": "Llama 2 7B",
    "llama3": "Llama 3 8B",
    "ministral8b": "Ministral8B",
    "codestral_mamba": "CodestralMamba 7B",
    "falcon_mamba": "FalconMamba 7B",
    "zamba2": "Zamba2",
}

linestyle_mapping = {
    "__tcm__": {"linestyle": "--", "label": ""},
}

style_dict = {
    "mlstm_simple": {
        "color": model_colors["mlstm_simple"],
        "label": model_labels["mlstm_simple"],
    },
    "xlstm": {"color": xlstm_colors["xlstm"], "label": model_labels["xlstm"]},
    "llama2": {"color": xlstm_colors["llama2"], "label": model_labels["llama2"]},
    "llama3": {"color": xlstm_colors["llama3"], "label": model_labels["llama3"]},
    "ministral8b": {
        "color": model_colors["ministral8b"],
        "label": model_labels["ministral8b"],
    },
    "codestral_mamba": {
        "color": xlstm_colors["codestral_mamba"],
        "label": model_labels["codestral_mamba"],
    },
    "falcon_mamba": {
        "color": xlstm_colors["falcon_mamba"],
        "label": model_labels["falcon_mamba"],
    },
    "zamba2": {"color": model_colors["zamba2"], "label": model_labels["zamba2"]},
}
