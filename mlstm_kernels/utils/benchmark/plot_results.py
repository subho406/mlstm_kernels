import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def plot_benchmark_result_table(
    result_df: pd.DataFrame,
    x_axis_param: str,
    title=None,
    legend_args: dict[str, Any] = dict(loc="lower left", bbox_to_anchor=(1.0, 0.0)),
    legend_order: list[str] = None,
    figsize=(2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54),
    grid_alpha: float = 0.2,
    plot_kwargs: dict[str, Any] = {"marker": "o", "linestyle": "-"},
    style_dict: dict[str, Any] = None,
    style_dict_colname_mapping_exact: bool = True,
    linestyle_mapping: dict[str, Any] = None,
    additional_exclude_col_regex: str = None,
    filename: str = None,
    y_label: str = "Time [ms]",
    x_label: str = None,
    add_legend: bool = True,
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
    else:
        f = ax.get_figure()
        ax.set_title(title)

    x_axis_vals = result_df[f"P--{x_axis_param}"]

    exclude_regex = "P--.*|Unnamed.*"
    if additional_exclude_col_regex is not None:
        exclude_regex += f"|{additional_exclude_col_regex}"
    y_axis_val_df = result_df.drop(
        result_df.filter(regex=exclude_regex, axis=1).columns, axis=1
    )

    for col in y_axis_val_df.columns:
        plot_kwargs_col = copy.deepcopy(plot_kwargs)
        if style_dict is not None:
            if style_dict_colname_mapping_exact:
                plot_kwargs_col.update(style_dict.get(col, {}))
            else:
                for col_key in style_dict.keys():
                    if col_key in col:
                        plot_kwargs_col.update(style_dict.get(col_key, {}))
        if linestyle_mapping is not None:
            for col_key in linestyle_mapping.keys():
                if col_key in col:
                    plot_kwargs_col.update(linestyle_mapping.get(col_key, {}))
        if "label" in plot_kwargs_col:
            ax.plot(x_axis_vals, y_axis_val_df[col].values, **plot_kwargs_col)
        else:
            ax.plot(
                x_axis_vals, y_axis_val_df[col].values, label=col, **plot_kwargs_col
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    xlabel = x_axis_param if x_label is None else x_label
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_label)
    if add_legend:
        ax.legend(**legend_args)
    ax.grid(alpha=grid_alpha)

    def savefig(file_ending):
        dir = Path(f"./plots/")
        dir.mkdir(parents=True, exist_ok=True)
        file = Path(f"./plots/plot_{filename}.{file_ending}")
        f.savefig(
            file,
            dpi=300,
            bbox_inches="tight",
        )

    if filename is not None:
        for file_ending in ["png", "pdf", "svg"]:
            savefig(file_ending)

    return f


def select_columns(df, selected_columns, keep_col_regex: str):
    keep_col = df.filter(regex=keep_col_regex)
    selected_df = df[selected_columns.values()]
    selected_df.columns = selected_columns.keys()
    selected_df = pd.concat([keep_col, selected_df], axis=1)
    return selected_df


FONTSIZE = 12
SMALL_OFFSET = 1
FONTSIZE_SMALL = FONTSIZE - SMALL_OFFSET
FONTSIZE_TICKS = 11

fontsize_delta = 0


def rc_context_wrapper(func: Callable, **kwargs):
    with mpl.rc_context(
        rc={
            "text.usetex": False,
            "font.size": FONTSIZE + fontsize_delta,
            "axes.labelsize": FONTSIZE + fontsize_delta,
            "legend.fontsize": FONTSIZE_SMALL + fontsize_delta,
            "xtick.labelsize": FONTSIZE_TICKS + fontsize_delta,
            "ytick.labelsize": FONTSIZE_TICKS + fontsize_delta,
            "axes.titlesize": FONTSIZE + fontsize_delta,
            "lines.markersize": 6.0,  # * default: 6.0
            "lines.linewidth": 2.0,  # * default: 1.5
        }
    ):
        return func(**kwargs)
