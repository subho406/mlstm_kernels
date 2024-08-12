# Partly adopted from https://github.com/sustcsonglin/flash-linear-attention

import functools

import torch


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