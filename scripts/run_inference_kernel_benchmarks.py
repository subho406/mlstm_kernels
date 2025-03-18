#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import argparse
from pathlib import Path

from dacite import from_dict
from omegaconf import OmegaConf

from mlstm_kernels.utils.benchmark.benchmarks.inference_kernel_benchmarks import (
    create_inference_kernel_benchmark,
)
from mlstm_kernels.utils.benchmark.param_handling import BenchmarkConfig
from mlstm_kernels.utils.benchmark.run_benchmark import run_and_record_benchmarks
from mlstm_kernels.utils.benchmark.utils import setup_output_folder

DEBUG = True


def _head_dim_benchmark(
    output_folder: Path, half_qkdim=False, batch_size: int = 1, debug: bool = False
):
    ### head dimension benchmark 7B
    head_dims_v = [64, 128, 256, 512, 1024]  # , 2048]
    embedding_dim = 4096
    num_heads = [embedding_dim // head_dim for head_dim in head_dims_v]
    if half_qkdim:
        head_dims_qk = [head_dim // 2 for head_dim in head_dims_v]
        bench_name = "head_dim_half_qk_7B"
    else:
        head_dims_qk = head_dims_v
        bench_name = "head_dim_7B"

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  num_heads: {num_heads}
  head_dim_qk: {head_dims_qk}
  head_dim_v: {head_dims_v}
fixed_params:
  batch_size: {batch_size}
  rep: {2500 if not debug else 10}
  warmup: {500 if not debug else 10}

x_axis_param: "head_dim_v"

kernel_specs:
  - kernel_name: "triton_fused"
    dtype: bfloat16
  - kernel_name: "triton_fused"
    dtype: float32
  # - kernel_name: "native"
  #   dtype: bfloat16
  #   use_torch_compile: True
  # - kernel_name: "native"
  #   dtype: float32
  #   use_torch_compile: True
  - kernel_name: "native"
    dtype: bfloat16
    use_torch_compile: False
  - kernel_name: "native"
    dtype: float32
    use_torch_compile: False
  - kernel_name: "mamba"
    dtype: bfloat16
    use_torch_compile: False
  - kernel_name: "mamba2"
    dtype: bfloat16
    use_torch_compile: False
  - kernel_name: "fused_recurrent_gla"
    dtype: bfloat16
    use_torch_compile: False
  - kernel_name: "fused_recurrent_simple_gla"
    dtype: bfloat16
    use_torch_compile: False


benchmark_name: {bench_name}
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_inference_kernel_benchmark, output_folder)


def _batch_size_benchmark(output_folder: Path, debug: bool = False):
    ### head dimension benchmark 7B
    embedding_dim = 4096
    num_heads = 8
    head_dim_v = embedding_dim // num_heads
    head_dim_qk = embedding_dim // num_heads // 2

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  batch_size: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
fixed_params:

  rep: {250 if not debug else 10}
  warmup: {50 if not debug else 10}

x_axis_param: "batch_size"

kernel_specs:
  - kernel_name: "triton_fused"
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}
  - kernel_name: "triton_fused"
    dtype: float32
    additional_params:
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}
  # - kernel_name: "native"
  #   dtype: bfloat16
  #   use_torch_compile: True
  # - kernel_name: "native"
  #   dtype: float32
  #   use_torch_compile: True
  - kernel_name: "native"
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_v}
  - kernel_name: "native"
    dtype: float32
    use_torch_compile: False
    additional_params:
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_v}
  - kernel_name: "mamba"
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      num_heads: 1
      head_dim_v: {2*embedding_dim}
      head_dim_qk: 16
  - kernel_name: "mamba2"
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      num_heads: {2*embedding_dim//64}
      head_dim_v: 64
      head_dim_qk: 64
  # the following do only work up to batch_size 16
  # - kernel_name: "fused_recurrent_gla"
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_v: {head_dim_v}
  #     head_dim_qk: {head_dim_qk}
  # - kernel_name: "fused_recurrent_simple_gla"
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_v: {head_dim_v}
  #     head_dim_qk: {head_dim_qk}


benchmark_name: constant_tokens_sequence
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_inference_kernel_benchmark, output_folder)


def run_multiple_benchmarks(
    output_dir: str = "./outputs_kernel_benchmarks",
    output_folder_suffix: str = "",
    debug: bool = False,
):
    output_folder = setup_output_folder(output_dir, name_suffix=output_folder_suffix)

    # _batch_size_benchmark(output_folder, num_heads=8, head_dim_qk=256, head_dim_v=512, debug=debug)

    # _head_dim_benchmark(output_folder, half_qkdim=False, batch_size=8, debug=debug)
    # _head_dim_benchmark(output_folder, half_qkdim=True, batch_size=8, debug=debug)
    _batch_size_benchmark(output_folder, debug=debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_suffix",
        type=str,
        required=False,
        help="Suffix that is appended to the output folder of the benchmark results.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )

    args = parser.parse_args()
    print(args)
    run_multiple_benchmarks(output_folder_suffix=args.folder_suffix, debug=args.debug)
