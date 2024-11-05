import gc
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

    def set_params(self, param_dict: dict) -> None:
        """Used to set all or multiple parameters of the benchmark at once."""
        if param_dict is None:
            return
        for k, v in param_dict.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown parameter: {k}")

    def setup_benchmark(self) -> None:
        torch_dtype = getattr(torch, self.dtype)

        inputs = self._get_input_tensors()

        inputs = [
            x.to(device=self.device, dtype=torch_dtype).requires_grad_(self.fwbw)
            for x in inputs
        ]
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


@dataclass
class mLSTMXLChunkSizeTuneBenchmark(mLSTMBenchmark):
    chunk_size_inter: int = None
    chunk_size_intra: int = None
    siz_b_L_parallel: int = None
    siz_b_L_loop: int = None
    siz_b_DH_parallel: int = None
    siz_b_DH_loop: int = None
    num_warps_intra: int = None
    num_warps_inter: int = None
    num_stages_intra: int = None
    num_stages_inter: int = None
    output_dtype: Literal["bfloat16", "float32"] = "float32"

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

        assert (
            self.kernel_name == "max_triton_v5xlchunksize"
        ), "Only supports max_triton_v5xlchunksize kernel"

        from ..mlstm.chunkwise.max_triton_fwbw_v5xlchunksize.triton_fwbw import (
            mlstm_chunkwise_max_triton,
        )

        kernel_fn = mlstm_chunkwise_max_triton
        kernel_fn = partial(
            kernel_fn,
            chunk_size_inter=self.chunk_size_inter,
            chunk_size_intra=self.chunk_size_intra,
            siz_b_L_parallel=self.siz_b_L_parallel,
            siz_b_L_loop=self.siz_b_L_loop
            if self.siz_b_L_loop is not None
            else self.siz_b_L_parallel,
            siz_b_DH_parallel=self.siz_b_DH_parallel,
            siz_b_DH_loop=self.siz_b_DH_loop,
            num_warps_intra=self.num_warps_intra,
            num_warps_inter=self.num_warps_inter,
            num_stages_intra=self.num_stages_intra,
            num_stages_inter=self.num_stages_inter,
        )

        return kernel_fn

    def available_kernels(self) -> list[str]:
        return ["max_triton_v5xlchunksize"]


@dataclass
class mLSTMXLChunkSizeBackwardTuneBenchmark(mLSTMXLChunkSizeTuneBenchmark):
    # TODO: weird, when running this benchmark I get an illegal memory access error
    def _get_input_tensors(self) -> tuple[torch.Tensor, ...]:
        q = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_qk),
            dtype=torch.float32,
        )
        # k = torch.randn(
        #     (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_qk),
        #     dtype=torch.float32,
        # )
        v = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_v),
            dtype=torch.float32,
        )
        ig = torch.randn(
            (
                self.batch_size,
                self.num_heads,
                self.sequence_length // self.chunk_size_intra,
                self.chunk_size_intra,
            ),
            dtype=torch.float32,
        )
        # ag = torch.randn(
        #     (
        #         self.batch_size,
        #         self.num_heads,
        #         self.sequence_length // self.chunk_size_intra,
        #         self.chunk_size_intra,
        #     ),
        #     dtype=torch.float32,
        # )
        # bg = torch.randn(
        #     (
        #         self.batch_size,
        #         self.num_heads,
        #         self.sequence_length // self.chunk_size_intra,
        #         self.chunk_size_intra,
        #     ),
        #     dtype=torch.float32,
        # )
        matCstate_all = torch.randn(
            (
                self.batch_size,
                self.num_heads,
                self.chunk_size_intra + 1,
                self.head_dim_qk,
                self.head_dim_v,
            ),
            dtype=torch.float32,
        )
        vecNstate_all = torch.randn(
            (
                self.batch_size,
                self.num_heads,
                self.chunk_size_intra + 1,
                self.head_dim_qk,
            ),
            dtype=torch.float32,
        )
        scaMstate_all = torch.randn(
            (self.batch_size, self.num_heads, self.chunk_size_intra + 1),
            dtype=torch.float32,
        )
        # matH_out = torch.randn(
        #     (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_v),
        #     dtype=torch.float32,
        # )
        vecN_out = torch.randn(
            (self.batch_size, self.num_heads, self.sequence_length), dtype=torch.float32
        )
        # vecM_out = torch.randn(
        #     (self.batch_size, self.num_heads, self.sequence_length), dtype=torch.float32
        # )
        # matDeltaH_out = torch.randn(
        #     (self.batch_size, self.num_heads, self.sequence_length, self.head_dim_v),
        #     dtype=torch.float32,
        # )
        matDeltaC_states = torch.randn(
            (
                self.batch_size,
                self.num_heads,
                self.chunk_size_intra + 1,
                self.head_dim_qk,
                self.head_dim_v,
            ),
            dtype=torch.float32,
        )
        vecDeltaN_states = torch.randn(
            (
                self.batch_size,
                self.num_heads,
                self.chunk_size_intra + 1,
                self.head_dim_qk,
            ),
            dtype=torch.float32,
        )
        return (
            q,
            q,
            v,
            ig,
            ig,
            ig,
            matCstate_all,
            vecNstate_all,
            scaMstate_all,
            v,
            vecN_out,
            vecN_out,
            v,
            matDeltaC_states,
            vecDeltaN_states,
        )

    def _get_kernel_fn(self) -> Callable[[tuple[torch.Tensor, ...]], torch.Tensor]:
        from functools import partial

        assert (
            self.kernel_name == "max_triton_v5xlchunksize-bw-dV"
        ), "Only supports max_triton_v5xlchunksize-bw-dV kernel"

        from ..mlstm.chunkwise.max_triton_fwbw_v5xlchunksize._triton_parallel_bw_dV import (
            mlstm_chunkwise__parallel_bw_dV,
        )

        kernel_fn = mlstm_chunkwise__parallel_bw_dV
        kernel_fn = partial(
            kernel_fn,
            chunk_size=self.chunk_size_intra,
            siz_b_LQ=self.siz_b_LQ,
            siz_b_LKV=self.siz_b_LKV if self.siz_b_LKV is not None else self.siz_b_LQ,
            siz_b_DHQK=self.siz_b_DHQK,
            siz_b_DHHV=self.siz_b_DHHV,
            num_warps=self.num_warps_intra,
            output_dtype=torch.float32
            if self.output_dtype == "float32"
            else torch.bfloat16,
        )

        return kernel_fn

    def available_kernels(self) -> list[str]:
        return ["max_triton_v5xlchunksize-bw-dV"]


def get_benchmark(
    kernel_spec: KernelSpec, param_dict: dict[str, Any]
) -> BenchmarkInterface:
    mlstm_benchmark = mLSTMBenchmark()
    flashattention_benchmark = FlashAttentionBenchmark()
    mlstm_xl_chunk_size_tune_benchmark = mLSTMXLChunkSizeTuneBenchmark()
    mlstm_xl_chunk_size_tune_benchmark_backward = (
        mLSTMXLChunkSizeBackwardTuneBenchmark()
    )

    if kernel_spec.kernel_name in mlstm_benchmark.available_kernels():
        benchmark = mlstm_benchmark
    elif kernel_spec.kernel_name in flashattention_benchmark.available_kernels():
        benchmark = flashattention_benchmark
    elif (
        kernel_spec.kernel_name
        in mlstm_xl_chunk_size_tune_benchmark.available_kernels()
    ):
        benchmark = mlstm_xl_chunk_size_tune_benchmark
    elif (
        kernel_spec.kernel_name
        in mlstm_xl_chunk_size_tune_benchmark_backward.available_kernels()
    ):
        benchmark = mlstm_xl_chunk_size_tune_benchmark_backward
    else:
        raise ValueError(f"Unknown kernel name: {kernel_spec.kernel_name}")

    benchmark.kernel_name = kernel_spec.kernel_name
    benchmark.dtype = kernel_spec.dtype
    benchmark.fwbw = kernel_spec.fwbw
    benchmark.use_torch_compile = kernel_spec.use_torch_compile
    benchmark.set_params(param_dict)
    return benchmark


def run_benchmarks(
    benchmark_config: BenchmarkConfig,
    param_prefix: str = "P--",
    additional_param_name_short: bool = True,
) -> pd.DataFrame:
    """Runs the different kernel configurations and summarizes the results in a DataFrame.

    Args:
        benchmark_config: The benchmark configuration.
        param_prefix: The prefix to add to the parameter names in the DataFrame.
    """
    param_dicts = benchmark_config.get_param_dicts()
    kernel_specs = benchmark_config.kernel_specs
    results = []
    for i, param_dict in enumerate(param_dicts):
        LOGGER.info(f"Parameter combination ({i+1}/{len(param_dicts)}): {param_dict}")
        # add a prefix to easily identify the parameters in the DataFrame
        result_dict = {f"{param_prefix}{k}": v for k, v in param_dict.items()}
        for k, kernel_spec in enumerate(kernel_specs):
            benchmark = get_benchmark(kernel_spec, param_dict)
            benchmark.set_params(kernel_spec.additional_params)
            benchmark.setup_benchmark()
            runtime = benchmark.run_benchmark()
            result_dict[
                kernel_spec.to_string(short_param_name=additional_param_name_short)
            ] = runtime
            LOGGER.info(
                f"Kernel ({k+1}/{len(kernel_specs)}): {kernel_spec.to_string()} finished. Runtime: {runtime} ms"
            )
            gc.collect()
            torch.cuda.empty_cache()
        results.append(result_dict)

    return pd.DataFrame(results)
