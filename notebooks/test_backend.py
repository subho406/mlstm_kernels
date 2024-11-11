import os
from math import sqrt

from mlstm_kernels.mlstm.chunkwise.triton_fwbw import mlstm_fwbw as mlstm_fwbw_chunk
from mlstm_kernels.mlstm.parallel.stable_torch_fwbw import mlstm_parallel_torch_ownbw
from mlstm_kernels.mlstm.chunkwise.triton_fwbw_stablef import (
    mlstm_fwbw as mlstm_fwbw_chunkstab,
)
from mlstm_kernels.mlstm.parallel.stable_torch_fwbw import mlstm_parallel_torch_ownbw
from mlstm_kernels.test_utils.test_fwbw import test_backward, test_forward

from math import sqrt
import os
import matplotlib.pyplot as plt
import torch

# options
_ = (mlstm_parallel_torch_ownbw, mlstm_fwbw_chunk, mlstm_fwbw_chunkstab)
# choose
baseline = mlstm_fwbw_chunk  # mlstm_parallel_torch_ownbw
comp = mlstm_fwbw_chunkstab


def shape_to_rect(shape):
    tot = 1
    for s in shape:
        tot *= s
    d1 = 1
    for s in shape[:-1]:
        d1 *= s
        if d1 > 0.25 * tot ** (1 / 2):
            return (d1, tot // d1)
    return (d1, tot // d1)


def plot_diff(x, y, title=""):
    dat = (
        (x - y)
        .abs()
        .float()
        .cpu()
        .detach()
        .reshape(shape_to_rect(x.shape))
        .transpose(0, 1)
    )
    fig, ax = plt.subplots(figsize=(20, 20))

    im = ax.imshow(
        dat,
    )
    fig.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    fig.show()

    n = 0
    while os.path.exists(f"testplot_{n}.png"):
        n += 1
    fig.savefig(f"testplot_{n}.png")

    fig, ax = plt.subplots(figsize=(20, 20))

    im = ax.imshow(
        dat.numpy()
        / (
            x.abs()
            .float()
            .cpu()
            .detach()
            .reshape(shape_to_rect(x.shape))
            .transpose(0, 1)
            + 1e-8
        ),
    )
    fig.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    fig.show()

    n = 0
    while os.path.exists(f"testplot_{n}_rel.png"):
        n += 1
    fig.savefig(f"testplot_{n}_rel.png")


def layer_norm(x, ndim=16):
    return torch.nn.functional.layer_norm(x, normalized_shape=ndim)


if __name__ == "__main__":
    import sys
    import numpy as np

    include_initial = True  # False
    B, H, T, K, V = 8, 8, 1024, 512, 512
    device = "cuda"
    dtype = torch.float32  # bfloat16

    q = 1 + 0.0 * torch.randn([B, H, T, K], device=device, dtype=dtype)
    k = 1 + 0.0 * torch.randn([B, H, T, K], device=device, dtype=dtype)
    v = torch.randn([B, H, T, V], device=device, dtype=dtype)
    i = torch.randn([B, H, T], device=device, dtype=dtype)
    f = +3.0 + 0.5 * torch.randn([B, H, T], device=device, dtype=dtype)

    # inps = np.load("test_failed_1729621406.npz")
    # print(inps)
    # q, k, v, i, f = [
    #     torch.from_numpy(inps[inp]).to(device=device, dtype=dtype)
    #     for inp in inps.files[:5]
    # ]

    B, H, T, K = q.shape
    V = v.shape[-1]

    if include_initial:
        C_i = torch.randn([B, H, K, V], device=device, dtype=torch.float32)
        n_i = 1 + torch.randn([B, H, K], device=device, dtype=torch.float32)
        m_i = 0.0 * torch.ones([B, H], device=device, dtype=torch.float32)
    else:
        C_i, n_i, m_i = None, None, None

    mask = torch.randn([B, H, T, V], device=device, dtype=dtype)
    # mask[:, :, ] = 0.
    # mask[:, :, 0] = 0.

    def baseline_f(*x):
        """
        Baseline function
        """
        return layer_norm(
            baseline(*x, initial_n=n_i, initial_m=m_i, chunk_size=16),
            ndim=(V,),
        )

    def comp_f(*x):
        """
        Function to be compared
        """
        return layer_norm(
            comp(*x, initial_n=n_i, initial_m=m_i, chunk_size=16),
            ndim=(V,),
        )

    # parallel = lambda *x: layer_norm(mlstm_fwbw(*x), ndim=(V,))
    # parallel_autograd = lambda *x: layer_norm(mlstm_torch_autograd(*x), ndim=(V,))

    backends = {
        "CW": baseline_f,
        "CWS": comp_f,
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
                (q, k, v, i, f, C_i, n_i, m_i),
                comp_func_kwargs={"atol": 0.01, "rtol": 0.01},
                show_diff_func=lambda x, y: plot_diff(
                    x, y, f"{bl_name}-{cm_name}-FW-B{B}H{H}T{T}K{K}V{V}"
                ),
            )
            if not "--skip-backward" in sys.argv:
                test_backward(
                    bl,
                    cm,
                    (q, k, v, i, f, C_i, n_i, m_i),
                    mask=mask,
                    comp_func_kwargs={"atol": 0.01, "rtol": 0.01},
                    show_diff_func=lambda x, y: plot_diff(
                        x, y, f"{bl_name}-{cm_name}-BW-B{B}H{H}T{T}K{K}V{V}"
                    ),
                )
