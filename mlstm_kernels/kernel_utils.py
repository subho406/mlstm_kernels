# Partly adopted from https://github.com/sustcsonglin/flash-linear-attention

import functools

import torch
import triton.language as tl

_torch_to_triton_dtype = {
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}

def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(
            ctx,
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{
                k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
                for k, v in kwargs.items()
            },
        )

    return wrapper

def contiguous_noctx(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(
            *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
            **{
                k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
                for k, v in kwargs.items()
            },
        )

    return wrapper

def torch2triton_dtype(dtype):
    return _torch_to_triton_dtype[dtype]

def is_power_of_2(n):
    assert isinstance(n, int)
    return (n & (n - 1)) == 0