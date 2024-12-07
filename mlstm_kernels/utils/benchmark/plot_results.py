import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def savefig(fig, filename: str):
    dir = Path("./plots/")
    dir.mkdir(parents=True, exist_ok=True)

    if filename is not None:
        for file_ending in ["png", "pdf", "svg"]:
            file = Path(f"./plots/plot_{filename}.{file_ending}")
            fig.savefig(
                file,
                dpi=300,
                bbox_inches="tight",
            )


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
        dir = Path("./plots/")
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


def create_group_names_from_cols(data_df: pd.DataFrame, colnames: str) -> list[str]:
    group_names = []
    group_cols = data_df[colnames].astype(int)
    for i, row in group_cols.iterrows():
        group_str = ""
        for i, colname in enumerate(colnames):
            group_str += f"{colname}={row[colname]}"
            if i < len(colnames) - 1:
                group_str += "\n"
        # print(group_str)
        group_names.append(group_str)
    return group_names


def create_bar_length_df(
    data_df: pd.DataFrame, colnames: list[str], offset: float = 0.0
) -> pd.DataFrame:
    # subtract the min from selected columns respectively and add an arbitrary offset
    bar_length_df = data_df[colnames].sub(data_df[colnames].min()).add(offset)
    bar_length_data_df = data_df.copy()
    bar_length_data_df[colnames] = bar_length_df
    return bar_length_data_df


def create_runtime_bar_plot(
    data_df: pd.DataFrame,
    group_col_names: list[str],
    title: str = None,
    bar_label_font_size: int = 10,
    bar_length_df: pd.DataFrame = None,
    plot_column_order: list[str] = None,
    style_dict: dict[str, Any] = None,
    fillna_val: float = -0.2,
    fillna_exclude_cols: list[str] = None,
    fillna_str: str = "OOSM",
    legend_args: dict[str, Any] = dict(loc="lower left", bbox_to_anchor=(1.0, 0.0)),
    legend_order: list[str] = None,
    figsize=(2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54),
    grid_alpha: float = 0.2,
    yticks: list[float] = None,
    y_label: str = None,
    ax=None,
):
    group_names = create_group_names_from_cols(
        data_df=data_df, colnames=group_col_names
    )
    raw_data_df = data_df.drop(columns=group_col_names)
    # data df contains only the columns to plot
    if fillna_exclude_cols is not None:
        raw_data_nan_cols_df = raw_data_df[fillna_exclude_cols].round(2)
    else:
        fillna_exclude_cols = []
        raw_data_nan_cols_df = pd.DataFrame()
    raw_data_nonan_cols_df = raw_data_df.drop(columns=fillna_exclude_cols)
    if bar_length_df is None:
        bar_length_df = raw_data_nonan_cols_df.fillna(fillna_val).round(2)
    else:
        bar_length_df = bar_length_df.drop(columns=fillna_exclude_cols)
        bar_length_df = (
            bar_length_df.drop(columns=group_col_names).fillna(fillna_val).round(2)
        )
    raw_data_nonan_cols_df = raw_data_nonan_cols_df.round(2).fillna(fillna_str)

    if fillna_exclude_cols is not None:
        raw_data_df = pd.concat([raw_data_nonan_cols_df, raw_data_nan_cols_df], axis=1)
        bar_length_df = pd.concat([bar_length_df, raw_data_nan_cols_df], axis=1)
    else:
        raw_data_df = raw_data_nonan_cols_df

    # x-axis locations
    x = np.arange(len(raw_data_df))
    width = 1 / (len(raw_data_df.columns) + 1)
    multiplier = 0

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
    else:
        f = ax.get_figure()
        ax.set_title(title)

    if plot_column_order is not None:
        columns = plot_column_order
    else:
        columns = bar_length_df.columns

    for col in columns:
        offset = width * multiplier
        if style_dict is None:
            rects = ax.bar(x + offset, bar_length_df[col], width, label=col)
        else:
            rects = ax.bar(x + offset, bar_length_df[col], width, **style_dict[col])
        ax.bar_label(
            rects, labels=raw_data_df[col], padding=2, fontsize=bar_label_font_size
        )
        multiplier += 1

    if y_label is None:
        y_label = "Time [ms]"
    ax.set_ylabel(y_label)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xticks(x + width, group_names)
    if legend_args and legend_order is None:
        ax.legend(**legend_args)
    elif legend_args and legend_order is not None:
        handles, labels = ax.get_legend_handles_labels()
        label_handle_dict = dict(zip(labels, handles))
        handles = [label_handle_dict[label] for label in legend_order]
        ax.legend(handles=handles, **legend_args)
    ax.grid(alpha=grid_alpha, which="both")

    if yticks is not None:
        ax.set_yticks(yticks)
        y_formatter = plt.ScalarFormatter()
        # y_formatter.set_scientific(False)
        # y_formatter.set_useOffset(10.0)
        ax.get_yaxis().set_major_formatter(y_formatter)

    return f
