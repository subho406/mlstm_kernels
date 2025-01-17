from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .generate_transfer_behavior_data import generate_qkv, make_gate_offset_sweep
from .mlstm_cell_func import mlstm_cell_func


def make_single_transfer_behavior_meshplot(
    ax: Axes,
    transfer_data: np.ndarray,
    igate_preact_offsets: list[float],
    fgate_preact_offsets: list[float],
    levels: list[float],
    x_label: str = "Forget Gate Preactivation",
    y_label: str = "Input Gate Preactivation",
    add_colorbar: bool = True,
    colorbar_ax: Axes = None,
    colobar_tick_format: str = "%4.1f",
    title: str = None,
    fig: Figure = None,
) -> Figure:
    if fig is None:
        fig = ax.get_figure()

    grid_x, grid_y = np.meshgrid(
        fgate_preact_offsets, igate_preact_offsets, indexing="xy"
    )
    data_z = transfer_data  # .transpose(0,1)

    plt.tight_layout()
    cmap = plt.get_cmap("PiYG")
    norm = mpl.colors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True)

    im = ax.pcolormesh(grid_x, grid_y, data_z, cmap=cmap, norm=norm)
    if add_colorbar:
        if colorbar_ax is None:
            fig.colorbar(mappable=im, ax=ax, format=colobar_tick_format)
        else:
            # NOTE: somehow this is not working
            fig.colorbar(mappable=im, cax=colorbar_ax, format=colobar_tick_format)

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    return fig


def generate_before_after_norm_transfer_behavior_plot(
    mlstm_func_specifier: str,
    norm_specifier: str,
    metric_specifier: str,
    seq_len: int,
    dhqk: int,
    dhhv: int,
    norm_eps: float,
    backend_eps: float,
    qkv_std: tuple[float, float, float],
    z_levels: list[float],
    igate_preact_offsets: list[float],
    fgate_preact_offsets: list[float],
    igate_preact_init_fn=torch.zeros,
    fgate_preact_init_fn=torch.zeros,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    fig_height: float = 7.5,
    fig_title: str = None,
) -> Figure:
    mlstm_fn = partial(
        mlstm_cell_func,
        mlstm_func_specifier=mlstm_func_specifier,
        norm_specifier=norm_specifier,
        norm_eps=norm_eps,
        backend_eps=backend_eps,
    )
    q_std, k_std, v_std = qkv_std
    q, k, v = generate_qkv(
        batch_size=1,
        num_heads=1,
        seq_len=seq_len,
        head_dim_qk=dhqk,
        head_dim_hv=dhhv,
        q_std=q_std,
        k_std=k_std,
        v_std=v_std,
    )

    zdata_after_norm, zdata_before_norm, _ = make_gate_offset_sweep(
        mlstm_func=mlstm_fn,
        q=q,
        k=k,
        v=v,
        fgate_preact_offsets=fgate_preact_offsets,
        igate_preact_offsets=igate_preact_offsets,
        igate_preact_init_fn=igate_preact_init_fn,
        fgate_preact_init_fn=fgate_preact_init_fn,
        metric_specifier=metric_specifier,
        dtype=dtype,
        device=device,
    )

    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(2 * fig_height, fig_height),
        sharey=True,
        sharex=True,
        gridspec_kw={"width_ratios": [0.46, 0.54]},
    )

    fig = make_single_transfer_behavior_meshplot(
        ax=ax1,
        transfer_data=zdata_before_norm,
        igate_preact_offsets=igate_preact_offsets,
        fgate_preact_offsets=fgate_preact_offsets,
        levels=z_levels,
        title="Gain before Norm",
        add_colorbar=False,
        fig=fig,
    )
    fig = make_single_transfer_behavior_meshplot(
        ax=ax2,
        transfer_data=zdata_after_norm,
        igate_preact_offsets=igate_preact_offsets,
        fgate_preact_offsets=fgate_preact_offsets,
        levels=z_levels,
        y_label=None,
        add_colorbar=True,
        title="Gain after Norm",
        fig=fig,
    )
    fig.suptitle(fig_title)

    return fig
