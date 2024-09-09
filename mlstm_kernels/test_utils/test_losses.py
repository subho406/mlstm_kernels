import torch

from ..components.ln import MultiHeadLayerNorm


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
