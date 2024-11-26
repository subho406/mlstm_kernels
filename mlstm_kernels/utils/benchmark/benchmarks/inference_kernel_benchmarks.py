from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from ..param_handling import KernelSpec
from .interface import BenchmarkInterface


@dataclass
class mLSTMStepKernelBenchmark(BenchmarkInterface):
    batch_size: int = None
    num_heads: int = None
    head_dim_qk: int = None
    head_dim_v: int = None

    siz_b_DHQK: int = None
    siz_b_DHHV: int = None
    num_warps: int = None
    num_stages: int = None

    use_torch_compile: bool = False

    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        c_old = torch.zeros(
            (self.batch_size, self.num_heads, self.head_dim_qk, self.head_dim_v),
            dtype=torch.float32,
        )
        n_old = torch.zeros((self.batch_size, self.num_heads, self.head_dim_qk), dtype=torch.float32)
        m_old = torch.zeros((self.batch_size, self.num_heads, 1), dtype=torch.float32)

        q = torch.randn((self.batch_size, self.num_heads, self.head_dim_qk), dtype=torch.float32)
        k = torch.randn((self.batch_size, self.num_heads, self.head_dim_qk), dtype=torch.float32)
        v = torch.randn((self.batch_size, self.num_heads, self.head_dim_v), dtype=torch.float32)
        i = torch.randn((self.batch_size, self.num_heads, 1), dtype=torch.float32)
        f = torch.randn((self.batch_size, self.num_heads, 1), dtype=torch.float32) + 4.5

        return c_old, n_old, m_old, q, k, v, i, f

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        from mlstm_kernels.torch import get_mlstm_step_kernel

        kernel_fn = get_mlstm_step_kernel(self.kernel_name)
        if self.use_torch_compile:
            kernel_fn = torch.compile(kernel_fn)
        return kernel_fn

    def setup_benchmark(self) -> None:
        torch_dtype = getattr(torch, self.dtype)

        inputs = self._get_input_tensors()

        inputs = [x.to(device=self.device, dtype=torch_dtype) for x in inputs]
        self.kernel_inputs = inputs

        kernel_fn = self._get_kernel_fn()
        kernel_fn = partial(
            kernel_fn,
            num_warps=self.num_warps,
            siz_b_DHQK=self.siz_b_DHQK,
            siz_b_DHHV=self.siz_b_DHHV,
            num_stages=self.num_stages,
        )

        def benchmark_fn() -> None:
            output = kernel_fn(*self.kernel_inputs)

        self.benchmark_fn = benchmark_fn

    def available_kernels(self) -> list[str]:
        from mlstm_kernels.torch import get_available_mlstm_step_kernels

        return get_available_mlstm_step_kernels()


def create_inference_kernel_benchmark(kernel_spec: KernelSpec, param_dict: dict[str, Any]) -> BenchmarkInterface:
    mlstm_step_kernel_benchmark = mLSTMStepKernelBenchmark()

    if kernel_spec.kernel_name in mlstm_step_kernel_benchmark.available_kernels():
        benchmark = mlstm_step_kernel_benchmark
    else:
        raise ValueError(f"Unknown kernel name: {kernel_spec.kernel_name}")

    benchmark.kernel_name = kernel_spec.kernel_name
    benchmark.dtype = kernel_spec.dtype
    benchmark.fwbw = kernel_spec.fwbw
    benchmark.use_torch_compile = kernel_spec.use_torch_compile
    benchmark.set_params(param_dict)
    return benchmark
