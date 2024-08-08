import torch
import triton

from mlstm_kernels import get_kernel

"""Benchmarks different kernels.

Enter in line vals the different kernels you want to benchmark.

Convention: <kernel>++<fw|fwbw>++<[compile]>

fw: forward pass only
fwbw: forward and backward pass
compile: apply torch.compile
"""

#! Parameters
BENCHMARK_NAME = "kernelbench2"
BATCH, N_HEADS = 1, 8
HEAD_DIMS = [64, 128, 256]
DTYPE = "float16"
#! =================

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
linestyle_mapping = {"fw": "--", "fwbw": "-"}


def create_kernel2style_mapping(kernel_names: list[str]) -> list[tuple[str, str]]:
    """This function maps a kernel name (as specified by the convention) to a color and a linestyle."""
    raw_kernel_names = [kernel_name.split("++")[0] for kernel_name in kernel_names]
    raw_kernel_names = list(set(raw_kernel_names))
    # map kernel name to color
    kernel2color = {
        kernel_name: color for kernel_name, color in zip(raw_kernel_names, colors)
    }
    # map kernel name to style
    kernel_names2style = []
    for kernel_name in kernel_names:
        raw_kernel_name = kernel_name.split("++")[0]
        fwbw_type = kernel_name.split("++")[1]
        kernel_names2style.append((kernel2color[raw_kernel_name], linestyle_mapping[fwbw_type]))
    return kernel_names2style


# mlstm only
configs = []
for HEAD_DIM in HEAD_DIMS:
    # kernels_to_benchmark = [
    #     "mlstm_parallel--triton++fwbw",
    #     "mlstm_parallel--triton++fw",
    #     "mlstm_parallel--torch_autograd++fwbw",
    #     "mlstm_parallel--torch_autograd++fw",
    #     "flash_attention--triton_flash++fwbw",
    #     "flash_attention--triton_flash++fw",
    #     "flash_attention--triton_tutorial++fwbw",
    #     "flash_attention--triton_tutorial++fw",
    # ]
    kernels_to_benchmark = [
        "mlstm_parallel--torch_autograd++fwbw++compile",
        "mlstm_parallel--triton++fwbw",
        "mlstm_chunkwise--triton++fwbw",
        # "mlstm_parallel--triton++fw",
        # "mlstm_parallel--torch_autograd++fw",
        "flash_attention--triton_flash++fwbw",
        # "flash_attention--triton_flash++fw",
        "flash_attention--triton_tutorial++fwbw",
        # "flash_attention--triton_tutorial++fw",
        "flash_linear_attention--triton_simple_gla++fwbw",
        # "flash_linear_attention--triton_simple_gla++fw",
        "flash_linear_attention--triton_fused_gla++fwbw",
        # "flash_linear_attention--triton_fused_gla++fw",
        # "flash_linear_attention--triton_fused_recurrent_gla++fwbw",
        # "flash_linear_attention--triton_fused_recurrent_gla++fw",
    ]

    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[256, 512, 1024, 2048, 4096],  # [2**i for i in range(10, 15)],
            line_arg="provider",
            line_vals=kernels_to_benchmark,
            line_names=kernels_to_benchmark,
            styles=create_kernel2style_mapping(kernels_to_benchmark),
            ylabel="ms",
            plot_name=f"{BENCHMARK_NAME}--batch-{BATCH}--head-{N_HEADS}--d-{HEAD_DIM}--dtype-{DTYPE}",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
            },
        )
    )


@triton.testing.perf_report(configs)
def bench_flash_mlstm_fwbw(BATCH, H, N_CTX, HEAD_DIM, provider, device="cuda"):
    warmup = 25
    rep = 100
    dtype = getattr(torch,DTYPE)

    # create input tensors
    q = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
    )
    k = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
    )
    v = torch.randn(
        (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
    )
    ig = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    fg = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=True)
    
    # only for flash linear attention
    gs = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

    # select kernel
    provider_split = provider.split("++")
    kernel_name = provider_split[0]
    fwbw_type = provider_split[1]

    if len(provider_split) > 2:
        use_torch_compile = "compile" in provider_split[2]
    else:
        use_torch_compile = False

    # prepare inputs
    if "mlstm" in kernel_name:
        inputs = (q, k, v, ig, fg)
    elif "flash_attention" in kernel_name:
        inputs = (q, k, v, None)
    elif "flash_linear_attention" in kernel_name:
        if "simple_gla" in kernel_name:
            inputs = (q, k, v, fg)
        else:
            inputs = (q, k, v, gs)

    # prepare kernel
    kernel_fn = get_kernel(kernel_name)

    if use_torch_compile:
        kernel_fn = torch.compile(kernel_fn)
    
    fw_fn = lambda: kernel_fn(*inputs)

    # fwbw
    if "fwbw" in fwbw_type:
        if "flash_linear_attention" in kernel_name:
            fn = lambda: fw_fn()[0].sum().backward()
        else:
            fn = lambda: fw_fn().sum().backward()
    else:
        fn = fw_fn
    print(f"Running benchmark for {provider}, with batch size {BATCH}, head size {H}, context size {N_CTX}, head dim {HEAD_DIM}, dtype {DTYPE}")
    try: 
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    except Exception as e:
        print(f"Error: {e}")
        ms = float("nan")
    return ms


if __name__ == "__main__":
    bench_flash_mlstm_fwbw.run(
        save_path=f"./outputs/{BENCHMARK_NAME}", print_data=True
    )
