import copy
from typing import Any

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
    y_label: str = "Time [ms]",
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

    ax.set_xlabel(x_axis_param)
    ax.set_ylabel(y_label)
    ax.legend(**legend_args)
    ax.grid(alpha=grid_alpha)
    return f
