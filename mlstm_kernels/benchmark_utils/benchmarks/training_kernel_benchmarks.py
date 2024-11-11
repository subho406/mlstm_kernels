import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from ..param_handling import KernelSpec
from .interface import BenchmarkInterface

LOGGER = logging.getLogger(__name__)


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

        from ...mlstm import get_mlstm_kernel

        kernel_fn = get_mlstm_kernel(self.kernel_name)
        kernel_fn = partial(kernel_fn, chunk_size=self.chunk_size)
        if self.use_torch_compile:
            kernel_fn = torch.compile(kernel_fn)
        return kernel_fn

    def available_kernels(self) -> list[str]:
        from ...mlstm import get_available_mlstm_kernels

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
        from ...baselines.flash_attention import registry as flash_attention_registry

        kernel_fn = flash_attention_registry[self.kernel_name]

        return kernel_fn

    def available_kernels(self) -> list[str]:
        from ...baselines.flash_attention import registry as flash_attention_registry

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

        assert self.kernel_name == "max_triton_v5xlchunksize", "Only supports max_triton_v5xlchunksize kernel"

        from ...mlstm.chunkwise.max_triton_fwbw_v5xlchunksize.triton_fwbw import mlstm_chunkwise_max_triton

        kernel_fn = mlstm_chunkwise_max_triton
        kernel_fn = partial(
            kernel_fn,
            chunk_size_inter=self.chunk_size_inter,
            chunk_size_intra=self.chunk_size_intra,
            siz_b_L_parallel=self.siz_b_L_parallel,
            siz_b_L_loop=self.siz_b_L_loop if self.siz_b_L_loop is not None else self.siz_b_L_parallel,
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
        vecN_out = torch.randn((self.batch_size, self.num_heads, self.sequence_length), dtype=torch.float32)
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

        from ...mlstm.chunkwise.max_triton_fwbw_v5xlchunksize._triton_parallel_bw_dV import (
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
            output_dtype=torch.float32 if self.output_dtype == "float32" else torch.bfloat16,
        )

        return kernel_fn

    def available_kernels(self) -> list[str]:
        return ["max_triton_v5xlchunksize-bw-dV"]


def create_training_kernel_benchmark(kernel_spec: KernelSpec, param_dict: dict[str, Any]) -> BenchmarkInterface:
    mlstm_benchmark = mLSTMBenchmark()
    flashattention_benchmark = FlashAttentionBenchmark()
    mlstm_xl_chunk_size_tune_benchmark = mLSTMXLChunkSizeTuneBenchmark()
    mlstm_xl_chunk_size_tune_benchmark_backward = mLSTMXLChunkSizeBackwardTuneBenchmark()

    if kernel_spec.kernel_name in mlstm_benchmark.available_kernels():
        benchmark = mlstm_benchmark
    elif kernel_spec.kernel_name in flashattention_benchmark.available_kernels():
        benchmark = flashattention_benchmark
    elif kernel_spec.kernel_name in mlstm_xl_chunk_size_tune_benchmark.available_kernels():
        benchmark = mlstm_xl_chunk_size_tune_benchmark
    elif kernel_spec.kernel_name in mlstm_xl_chunk_size_tune_benchmark_backward.available_kernels():
        benchmark = mlstm_xl_chunk_size_tune_benchmark_backward
    else:
        raise ValueError(f"Unknown kernel name: {kernel_spec.kernel_name}")

    benchmark.kernel_name = kernel_spec.kernel_name
    benchmark.dtype = kernel_spec.dtype
    benchmark.fwbw = kernel_spec.fwbw
    benchmark.use_torch_compile = kernel_spec.use_torch_compile
    benchmark.set_params(param_dict)
    return benchmark
