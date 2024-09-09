import torch


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
