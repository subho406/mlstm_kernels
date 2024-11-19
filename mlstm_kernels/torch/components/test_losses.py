import torch


def loss_layernorm_offset_quadratic(input_tensor: torch.Tensor, seed: int = 0, eps: float = 1e-5) -> torch.Tensor:
    torch.manual_seed(seed)
    offset = torch.randn_like(input_tensor)
    assert len(input_tensor.shape) == 4
    
    input_tensor_scaled = (input_tensor - input_tensor.mean(-1, keepdim=True)) / torch.sqrt((input_tensor.var(dim=-1, keepdim=True) + eps))

    loss = ((input_tensor_scaled + offset) ** 2).sum()
    return loss
