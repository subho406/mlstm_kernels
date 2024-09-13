import matplotlib.pyplot as plt
import torch
from mlstm_kernels.test_utils.test_fwbw import test_forward, test_backward

# from mlstm_kernels.mlstm.chunkwise.triton_fwbw_debug import chunk_mlstm
from mlstm_kernels.mlstm.chunkwise import mlstm_chunkwise_triton as mlstm_fwbw_chunk
from mlstm_kernels.mlstm.chunkwise import (
    mlstm_chunkwise_stable_triton as mlstm_fwbw_chunkstab,
)

# from mlstm_kernels.mlstm.parallel import mlstm
import matplotlib.pyplot as plt
from math import sqrt
import os


def shape_to_rect(shape):
    tot = 1
    for s in shape:
        tot *= s
    d1 = 1
    for s in shape[:-1]:
        d1 *= s
        if d1 > 0.25 * sqrt(tot):
            return (d1, tot // d1)
    return (d1, tot // d1)


def plot_diff(x, y, title=""):
    fig, ax = plt.subplots()

    im = ax.imshow(
        (x - y)
        .abs()
        .float()
        .cpu()
        .detach()
        .reshape(shape_to_rect(x.shape))
        .transpose(0, 1),
        vmin=0,
        # vmax=0.1,
    )
    fig.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    fig.show()

    n = 0
    while os.path.exists(f"testplot_{n}.png"):
        n += 1
    fig.savefig(f"testplot_{n}.png")


def layer_norm(x, ndim=16):
    return torch.nn.functional.layer_norm(x, normalized_shape=ndim)


if __name__ == "__main__":
    B, H, T, K, V = 3, 4, 2048, 256, 256
    device = "cuda"
    dtype = torch.bfloat16

    q = 1 + 0.0 * torch.randn([B, H, T, K], device=device, dtype=dtype)
    k = 1 + 0.0 * torch.randn([B, H, T, K], device=device, dtype=dtype)
    v = torch.randn([B, H, T, V], device=device, dtype=dtype)

    i = torch.randn([B, H, T], device=device, dtype=dtype)
    f = +3.0 + 0.5 * torch.randn([B, H, T], device=device, dtype=dtype)

    C_i = torch.randn([B, H, K, V], device=device, dtype=dtype)
    n_i = 1 + torch.randn([B, H, K], device=device, dtype=dtype)
    m_i = 0.0 * torch.ones([B, H], device=device, dtype=dtype)

    mask = torch.randn([B, H, T, V], device=device, dtype=dtype)
    # mask[:, :, ] = 0.
    # mask[:, :, 0] = 0.

    def baseline(*x):
        return layer_norm(
            mlstm_fwbw_chunk(*x, initial_n=n_i, initial_m=m_i, chunk_size=16),
            ndim=(V,),
        )

    def triton(*x):
        return layer_norm(
            mlstm_fwbw_chunkstab(*x, initial_n=n_i, initial_m=m_i, chunk_size=16),
            ndim=(V,),
        )

    # parallel = lambda *x: layer_norm(mlstm_fwbw(*x), ndim=(V,))
    # parallel_autograd = lambda *x: layer_norm(mlstm_torch_autograd(*x), ndim=(V,))

    backends = {
        "CW": baseline,
        "CWS": triton,
        # "PR": parallel,
        # "PA": parallel_autograd,
    }

    for bl_name, bl in backends.items():
        for cm_name, cm in backends.items():
            if cm == bl:
                break
            print(f"Test {cm_name} against {bl_name}")
            test_forward(
                bl,
                cm,
                (q, k, v, i, f),
                comp_func_kwargs={"atol": 0.000001, "rtol": 0.00001},
                show_diff_func=lambda x, y: plot_diff(
                    x, y, f"{bl_name}-{cm_name}-FW-B{B}H{H}T{T}K{K}V{V}"
                ),
            )
            test_backward(
                bl,
                cm,
                (q, k, v, i, f),
                mask=mask,
                comp_func_kwargs={"atol": 0.0000001, "rtol": 0.0000001},
                show_diff_func=lambda x, y: plot_diff(
                    x, y, f"{bl_name}-{cm_name}-BW-B{B}H{H}T{T}K{K}V{V}"
                ),
            )
