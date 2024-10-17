import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch

from .do_bench import do_bench
from .param_handling import BenchmarkConfig, KernelSpec

LOGGER = logging.getLogger(__name__)


@dataclass
class BenchmarkInterface:
    warmup: int = 25
    """Warmup time (in ms) or warmup iterations."""
    rep: int = 1000
    """Repetition time (in ms) or repetition iterations."""

    warmup_and_rep_in_ms: bool = False
    """If true, the warmup and rep are in milliseconds, otherwise they are iterations."""

    device: str = "cuda"
    """The device to run the benchmark on."""

    dtype: Literal["float16", "float32", "float64", "bfloat16"] = "bfloat16"
    """The data type to use for the benchmark."""

    fwbw: bool = True
    """If true, the benchmark will run the forward and backward pass."""

    benchmark_fn: Callable = None
    """The benchmark function to run."""

    kernel_inputs: tuple[torch.Tensor, ...] = None
    """The input tensors to the benchmark function."""

    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        """Returns the kernel function to benchmark.
        The inputs to the kernel function are the tensors returned by `_get_input_tensors`."""
        raise NotImplementedError

    def _get_loss_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """By default use the sum of the output as loss."""

        def loss_fn(output: torch.Tensor) -> torch.Tensor:
            return torch.sum(output)

        return loss_fn

    def available_kernels(self) -> list[str]:
        """Returns the available kernel names for the benchmark."""
        raise NotImplementedError

    def set_params(self, param_dict) -> None:
        """Used to set all or multiple parameters of the benchmark at once."""
        for k, v in param_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown parameter: {k}")

    def setup_benchmark(self) -> None:
        torch_dtype = getattr(torch, self.dtype)

        inputs = self._get_input_tensors()

        inputs = [x.to(device=self.device, dtype=torch_dtype).requires_grad_(self.fwbw) for x in inputs]
        self.kernel_inputs = inputs

        kernel_fn = self._get_kernel_fn()

        loss_fn = self._get_loss_fn()

        def benchmark_fn() -> None:
            output = kernel_fn(*self.kernel_inputs)
            if self.fwbw:
                loss = loss_fn(output)
                loss.backward()

        self.benchmark_fn = benchmark_fn

    def run_benchmark(self, return_mode: Literal["mean", "median"] = "mean") -> int:
        """Runs the benchmark and returns the runtime in milliseconds."""

        if self.benchmark_fn is None:
            raise RuntimeError("The benchmark function has not been set up.")
        try:
            runtime = do_bench(
                self.benchmark_fn,
                warmup=self.warmup,
                rep=self.rep,
                warmup_and_rep_in_ms=self.warmup_and_rep_in_ms,
                return_mode=return_mode,
                grad_to_none=self.kernel_inputs,
            )
        except Exception as e:
            LOGGER.warning(f"Error: {e}")
            runtime = float("nan")
        return runtime


@dataclass
class mLSTMBenchmark(BenchmarkInterface):
    batch_size: int = None
    sequence_length: int = None
    num_heads: int = None
    head_dim_qk: int = None
    head_dim_v: int = None

    chunk_size: int = None

    kernel_name: str = None

    use_torch_compile: bool = False

    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        q = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_qk),
            dtype=torch.float32,
        )
        k = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_qk),
            dtype=torch.float32,
        )
        v = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_v),
            dtype=torch.float32,
        )
        ig = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length),
            dtype=torch.float32,
        )
        fg = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length),
            dtype=torch.float32,
        )
        return q, k, v, ig, fg

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        from functools import partial

        from ..mlstm import get_mlstm_kernel

        kernel_fn = get_mlstm_kernel(self.kernel_name)
        kernel_fn = partial(kernel_fn, chunk_size=self.chunk_size)
        if self.use_torch_compile:
            kernel_fn = torch.compile(kernel_fn)
        return kernel_fn

    def available_kernels(self) -> list[str]:
        from ..mlstm import get_available_mlstm_kernels

        return get_available_mlstm_kernels()


@dataclass
class FlashAttentionBenchmark(mLSTMBenchmark):
    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        q = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_qk),
            dtype=torch.float32,
        )
        k = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_qk),
            dtype=torch.float32,
        )
        v = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_v),
            dtype=torch.float32,
        )
        return q, k, v

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        from ..baselines.flash_attention import registry as flash_attention_registry

        kernel_fn = flash_attention_registry[self.kernel_name]

        return kernel_fn

    def available_kernels(self) -> list[str]:
        from ..baselines.flash_attention import registry as flash_attention_registry

        return list(flash_attention_registry.keys())


def get_benchmark(kernel_spec: KernelSpec, param_dict: dict[str, Any]) -> BenchmarkInterface:
    mlstm_benchmark = mLSTMBenchmark()
    flashattention_benchmark = FlashAttentionBenchmark()

    if kernel_spec.kernel_name in mlstm_benchmark.available_kernels():
        benchmark = mlstm_benchmark
    elif kernel_spec.kernel_name in flashattention_benchmark.available_kernels():
        benchmark = flashattention_benchmark
    else:
        raise ValueError(f"Unknown kernel name: {kernel_spec.kernel_name}")

    benchmark.kernel_name = kernel_spec.kernel_name
    benchmark.dtype = kernel_spec.dtype
    benchmark.fwbw = kernel_spec.fwbw
    benchmark.use_torch_compile = kernel_spec.use_torch_compile
    benchmark.set_params(param_dict)
    return benchmark


def run_benchmarks(benchmark_config: BenchmarkConfig, param_prefix: str = "P--") -> pd.DataFrame:
    """Runs the different kernel configurations and summarizes the results in a DataFrame.

    Args:
        benchmark_config: The benchmark configuration.
        param_prefix: The prefix to add to the parameter names in the DataFrame.
    """
    param_dicts = benchmark_config.get_param_dicts()
    kernel_specs = benchmark_config.kernel_specs
    results = []
    for param_dict in param_dicts:
        LOGGER.info(f"Running parameters: {param_dict}")
        # add a prefix to easily identify the parameters in the DataFrame
        result_dict = {f"{param_prefix}{k}": v for k, v in param_dict.items()}
        for kernel_spec in kernel_specs:
            LOGGER.info(f"Running kernel: {kernel_spec.to_string()}")
            benchmark = get_benchmark(kernel_spec, param_dict)
            benchmark.set_params(kernel_spec.additional_params)
            benchmark.setup_benchmark()
            runtime = benchmark.run_benchmark()
            result_dict[kernel_spec.to_string()] = runtime
        results.append(result_dict)

    return pd.DataFrame(results)
