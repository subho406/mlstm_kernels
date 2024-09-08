from pathlib import Path
import torch
import logging

from .plot_utils import (
    plot_numerical_diffs_per_batchhead,
    plot_error_statistics_over_time_per_batchhead,
    compute_errors_per_batchhead,
)
from .components.ln import MultiHeadLayerNorm
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


def dtype2str(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.float64:
        return "fp64"
    elif dtype == torch.bfloat16:
        return "bf16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def check_correctness(
    test_specifier: str,
    baseline: torch.Tensor,
    target: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    vmax: float = None,
    max_num_batchhead_plots: int = -1,
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

    title = f"{test_specifier:>20}|{dtype_str:>6}| max diff: {(baseline - target).abs().max():>25}| mean diff: {(baseline - target).abs().mean():25} | allclose(atol={atol},rtol={rtol}): {result}"
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


def loss_layernorm_offset_quadratic(
    input_tensor: torch.Tensor, seed: int = 0, eps: float = 1e-6
) -> torch.Tensor:
    torch.manual_seed(seed)
    offset = torch.randn_like(input_tensor)
    assert len(input_tensor.shape) == 4
    ndim = input_tensor.shape[1] * input_tensor.shape[-1]  # NH * DHV
    mh_layernorm = MultiHeadLayerNorm(ndim=ndim, eps=eps).to(input_tensor.device)
    input_tensor_scaled = mh_layernorm(input_tensor)

    loss = ((input_tensor_scaled + offset) ** 2).sum()
    return loss
