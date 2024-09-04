from pathlib import Path
import torch
import logging

from .plot_utils import plot_numerical_diffs_per_batchhead
from .components.ln import MultiHeadLayerNorm
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)

def check_correctness(
    test_specifier: str,
    baseline: torch.Tensor,
    target=torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    vmax: float = None,
    savepath: str = None,
) -> bool:
    assert isinstance(baseline, torch.Tensor)
    assert isinstance(target, torch.Tensor)

    assert baseline.shape == target.shape
    assert baseline.dtype == target.dtype

    result = torch.allclose(baseline, target, atol=atol, rtol=rtol)
    dtype = baseline.dtype

    if dtype == torch.float32:
        dtype_str = "fp32"
    elif dtype == torch.float16:
        dtype_str = "fp16"
    elif dtype == torch.float64:
        dtype_str = "fp64"
    elif dtype == torch.bfloat16:
        dtype_str = "bf16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    title = f"{test_specifier:>20}|{dtype_str:>6}| max diff: {(baseline - target).abs().max():>25}| mean diff: {(baseline - target).abs().mean():25} | allclose(atol={atol},rtol={rtol}): {result}"
    print(title)
    LOGGER.info(title)
    if savepath is not None:
        if vmax is None:
            vmax = atol       
        figs = plot_numerical_diffs_per_batchhead(
            baseline=baseline,
            target=target,
            title=f"{test_specifier}|{dtype_str}",
            vmin=0,
            vmax=vmax,
            rtol=rtol, 
            atol=atol
        )
        savepath = Path(savepath) / f"{test_specifier}--{dtype_str}"
        savepath.mkdir(parents=True, exist_ok=True)
        for i, fig in enumerate(figs):
            fig.savefig(savepath / f"batchhead_{i}.pdf")
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

    loss = ((input_tensor_scaled + offset)**2).sum()
    return loss