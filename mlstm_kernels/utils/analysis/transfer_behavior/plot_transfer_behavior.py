import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def make_single_transfer_behavior_meshplot(
    ax: Axes,
    transfer_data: np.ndarray,
    igate_preact_offsets: list[float],
    fgate_preact_offsets: list[float],
    levels: list[float],
) -> mpl.figure.Figure:
    grid_x, grid_y = np.meshgrid(
        fgate_preact_offsets, igate_preact_offsets, indexing="xy"
    )
    data_z = transfer_data  # .transpose(0,1)

    cmap = plt.get_cmap("PiYG")
    norm = mpl.colors.BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=True)

    im = ax.pcolormesh(grid_x, grid_y, data_z, cmap=cmap, norm=norm)
    fig = ax.get_figure()
    fig.colorbar(im, ax=ax)
    ax.set_ylabel("Input Gate Preactivation")
    ax.set_xlabel("Forget Gate Preactivation")
    return fig
