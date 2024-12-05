import matplotlib as mpl

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
    "xlstm": "xLSTM",
    "llama2": "Llama2",
    "llama3": "Llama3",
    "ministral8b": "Ministral8B",
    "codestral_mamba": "Mamba2",
    "falcon_mamba": "Mamba1",
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
