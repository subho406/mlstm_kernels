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
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
    else:
        f = ax.get_figure()
        ax.set_title(title)

    x_axis_vals = result_df[f"P--{x_axis_param}"]

    y_axis_val_df = result_df.drop(result_df.filter(regex="P--.*|Unnamed.*", axis=1).columns, axis=1)

    for col in y_axis_val_df.columns:
        if style_dict is not None:
            plot_kwargs.update(style_dict.get(col, {}))
        ax.plot(x_axis_vals, y_axis_val_df[col].values, label=col, **plot_kwargs)

    ax.set_xlabel(x_axis_param)
    ax.set_ylabel("Time [ms]")
    ax.legend(**legend_args)
    ax.grid(alpha=grid_alpha)
    return f
