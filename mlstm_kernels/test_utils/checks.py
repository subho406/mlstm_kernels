from pathlib import Path
import torch
import logging
import numpy as np

from ..plot_utils import (
    plot_numerical_diffs_per_batchhead,
    plot_error_statistics_over_time_per_batchhead,
    compute_errors_per_batchhead,
)
import matplotlib.pyplot as plt

from ..torch_utils import dtype2str

LOGGER = logging.getLogger(__name__)


def check_correctness(
    test_specifier: str,
    baseline: torch.Tensor,
    target: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    vmax: float = None,
    max_num_batchhead_plots: int = -1,
    percentiles: list = [50, 90],
    savepath: str = None,
) -> bool:
    assert isinstance(baseline, torch.Tensor)
    assert isinstance(target, torch.Tensor)

    assert baseline.shape == target.shape

    dtype = target.dtype

    # closeness in highest precision
    baseline = baseline.to(dtype=torch.float64)
    target = target.to(dtype=torch.float64)

    result = torch.allclose(baseline, target, atol=atol, rtol=rtol)

    dtype_str = dtype2str(dtype)

    errors = (baseline.detach() - target.detach()).abs()
    errors_np = errors.cpu().numpy()

    error_percentiles = np.percentile(errors_np, percentiles)

    def make_percentile_str(error_percentiles: np.ndarray, percentiles: int) -> str:
        percentile_str = ""
        for i, percentile in enumerate(percentiles):
            percentile_str += f"p{percentile:<3}: {error_percentiles[i]:>5.5e}"
            if i < len(percentiles) - 1:
                percentile_str += "|"
        return percentile_str

    # title = f"{test_specifier:>20}|{dtype_str:>6}| max diff: {(baseline - target).abs().max():>25}| mean diff: {(baseline - target).abs().mean():25} | allclose(atol={atol},rtol={rtol}): {result}"
    title = f"{test_specifier:>20}|{dtype_str:>6}| diff: {make_percentile_str(error_percentiles, percentiles):>35} | maxdiff: {errors.max():25} | meandiff: {errors.mean():25} | allclose(atol={atol},rtol={rtol}): {result} | max abs bl: {baseline.abs().max():7.6f} | max abs tg: {target.abs().max():.5}"

    print(title)
    LOGGER.info(title)
    if savepath is not None:
        if vmax is None:
            vmax = atol
        savepath = Path(savepath) / f"{test_specifier}--{dtype_str}"
        savepath.mkdir(parents=True, exist_ok=True)

        figs = plot_numerical_diffs_per_batchhead(
            baseline=baseline,
            target=target,
            title=f"{test_specifier}|{dtype_str}",
            vmin=0,
            vmax=vmax,
            rtol=rtol,
            atol=atol,
            max_num_batchhead_plots=max_num_batchhead_plots,
        )
        for i, fig in enumerate(figs):
            fig.savefig(savepath / f"diff_imshow--batchhead_{i}.pdf")
            plt.close()

        figs = plot_error_statistics_over_time_per_batchhead(
            errors=compute_errors_per_batchhead(baseline=baseline, target=target),
            percentiles=[50, 90, 100],
            add_mean=True,
            title=f"{test_specifier}|{dtype_str}",
            ema_alpha=0.02,
            max_num_batchhead_plots=max_num_batchhead_plots,
        )

        for i, fig in enumerate(figs):
            fig.savefig(savepath / f"diff_lineplot--batchhead_{i}.pdf")
            plt.close()

    return result
