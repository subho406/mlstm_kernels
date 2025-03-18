#  Copyright (c) NXAI GmbH.
#  This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import argparse
from pathlib import Path

from dacite import from_dict
from omegaconf import OmegaConf

from mlstm_kernels.utils.benchmark.benchmarks.training_kernel_benchmarks import (
    create_training_kernel_benchmark,
)
from mlstm_kernels.utils.benchmark.param_handling import BenchmarkConfig
from mlstm_kernels.utils.benchmark.run_benchmark import run_and_record_benchmarks
from mlstm_kernels.utils.benchmark.utils import setup_output_folder

mlstm_triton_kernel_specs = """
  #? this kernel supports only head dimensions up to 256
  # and head dim qk must be equal to head dim v
  # - kernel_name: "parallel--triton_limit_headdim"
  #   fwbw: {fwbw}
  #   dtype: bfloat16
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_v: {head_dim_v}
  #     head_dim_qk: {head_dim_v}


  #### limit_chunk
  #? chunk size 64 is optimal
  - kernel_name: "chunkwise--triton_limit_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 64
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  # - kernel_name: "chunkwise--triton_limit_chunk"
  #   dtype: bfloat16
  #   fwbw: {fwbw}
  #   additional_params:
  #     chunk_size: 128
  # - kernel_name: "chunkwise--triton_limit_chunk"
  #   dtype: bfloat16
  #   fwbw: {fwbw}
  #   additional_params:
  #     chunk_size: 32
  #### xl_chunk
  - kernel_name: "chunkwise--triton_xl_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 64
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  - kernel_name: "chunkwise--triton_xl_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 128
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  - kernel_name: "chunkwise--triton_xl_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 256
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  - kernel_name: "chunkwise--triton_xl_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 512
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  - kernel_name: "chunkwise--triton_xl_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 1024
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  - kernel_name: "chunkwise--triton_xl_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 2048
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  - kernel_name: "chunkwise--triton_xl_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 4096
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  ### xl_chunk_siging normalize=False
  - kernel_name: "chunkwise--triton_xl_chunk_siging"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 64
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}
      normalize: False

  - kernel_name: "chunkwise--triton_xl_chunk_siging"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 128
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}
      normalize: False

  - kernel_name: "chunkwise--triton_xl_chunk_siging"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 256
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}
      normalize: False

  - kernel_name: "chunkwise--triton_xl_chunk_siging"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 512
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}
      normalize: False

  - kernel_name: "chunkwise--triton_xl_chunk_siging"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 1024
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}
      normalize: False

  - kernel_name: "chunkwise--triton_xl_chunk_siging"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 2048
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}
      normalize: False

  - kernel_name: "chunkwise--triton_xl_chunk_siging"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 4096
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}
      normalize: False
"""

fla_kernel_specs = """
  - kernel_name: "chunk_gla"
    dtype: bfloat16
    use_torch_compile: False
    fwbw: {fwbw}
    additional_params:
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  - kernel_name: "fused_chunk_gla"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}


  - kernel_name: "chunk_simple_gla"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}
"""


def headdim_benchmark_mlstm_triton(
    output_folder: Path,
    fwbw: bool = True,
    half_qkdim=False,
    seq_len: int = 8192,
    batch_size: int = 4,
    debug: bool = False,
):
    head_dims_v = [32, 64, 128, 256, 512, 1024]  # 2048 yields illegal memory access
    embedding_dim = 4096
    batch_size = batch_size if batch_size is not None else 4
    num_heads = [embedding_dim // head_dim for head_dim in head_dims_v]
    name = "mlstm_triton"
    if half_qkdim:
        head_dims_qk = [head_dim // 2 for head_dim in head_dims_v]
        bench_name = f"{name}_headdim_half_qk_7B_{'fwbw' if fwbw else 'fw'}"
    else:
        head_dims_qk = head_dims_v
        bench_name = f"{name}_headdim_7B_{'fwbw' if fwbw else 'fw'}"

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  num_heads: {num_heads}
  head_dim_qk: {head_dims_qk}
  head_dim_v: {head_dims_v}
fixed_params:
  sequence_length: {seq_len}
  batch_size: {batch_size}
  rep: {100 if not debug else 10}
  warmup: {25 if not debug else 3}
  result_aggregation: "median"

x_axis_param: "head_dim_v"

kernel_specs:
  #### limit_chunk
  #? chunk size 64 is optimal
  - kernel_name: "chunkwise--triton_limit_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 64

  # - kernel_name: "chunkwise--triton_limit_chunk"
  #   dtype: bfloat16
  #   fwbw: {fwbw}
  #   additional_params:
  #     chunk_size: 128
  # - kernel_name: "chunkwise--triton_limit_chunk"
  #   dtype: bfloat16
  #   fwbw: {fwbw}
  #   additional_params:
  #     chunk_size: 32
  #### xl_chunk
  - kernel_name: "chunkwise--triton_xl_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 64

  - kernel_name: "chunkwise--triton_xl_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 128

  - kernel_name: "chunkwise--triton_xl_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 256

  - kernel_name: "chunkwise--triton_xl_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 512

  ### xl_chunk_siging normalize=False
  - kernel_name: "chunkwise--triton_xl_chunk_siging"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 64
      normalize: False

  - kernel_name: "chunkwise--triton_xl_chunk_siging"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 128
      normalize: False

  - kernel_name: "chunkwise--triton_xl_chunk_siging"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 256
      normalize: False

  - kernel_name: "chunkwise--triton_xl_chunk_siging"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 512
      normalize: False


benchmark_name: {bench_name}
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_training_kernel_benchmark, output_folder)


def headdim_benchmark_fla(
    output_folder: Path,
    fwbw: bool = True,
    half_qkdim=False,
    seq_len: int = 8192,
    batch_size: int | None = 4,
    debug: bool = False,
):
    head_dims_v = [
        128,
        256,
        512,
        1024,
    ]  # [256, 512, 1024, 2048] #[32, 64, 128, 256, 512, 1024, 2048]
    embedding_dim = 4096
    batch_size = batch_size if batch_size is not None else 4
    num_heads = [embedding_dim // head_dim for head_dim in head_dims_v]
    name = "fla"
    if half_qkdim:
        head_dims_qk = [head_dim // 2 for head_dim in head_dims_v]
        bench_name = f"{name}_headdim_half_qk_7B_{'fwbw' if fwbw else 'fw'}"
    else:
        head_dims_qk = head_dims_v
        bench_name = f"{name}_headdim_7B_{'fwbw' if fwbw else 'fw'}"

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  num_heads: {num_heads}
  head_dim_qk: {head_dims_qk}
  head_dim_v: {head_dims_v}
fixed_params:
  sequence_length: {seq_len}
  batch_size: {batch_size}
  rep: {100 if not debug else 10}
  warmup: {25 if not debug else 3}
  result_aggregation: "median"

x_axis_param: "head_dim_v"

kernel_specs:
  - kernel_name: "chunk_gla"
    dtype: bfloat16
    use_torch_compile: False
    fwbw: {fwbw}

  - kernel_name: "fused_chunk_gla"
    dtype: bfloat16
    fwbw: {fwbw}

  - kernel_name: "chunk_simple_gla"
    dtype: bfloat16
    fwbw: {fwbw}

benchmark_name: {bench_name}
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_training_kernel_benchmark, output_folder)


def consttoken_benchmark_mlstm_triton(
    output_folder: Path,
    fwbw: bool = True,
    debug: bool = False,
    sequence_length_limits=[9, 17],
    num_heads: int | None = 8,
    half_qkdim: bool = True,
):
    """
    Const token benchmark for the mlstm chunkwise triton kernels.

    This benchmark uses head dimensions as they would be used in a 7B model (i.e. embedding dimension 4096).

    It uses 8 heads and the qk head dimension is half of the v head dimension.
    """
    embedding_dim = 4096
    num_heads = num_heads if num_heads is not None else 8
    head_dim_v = embedding_dim // num_heads
    head_dim_qk = embedding_dim // num_heads
    if half_qkdim:
        head_dim_qk = head_dim_qk // 2
    sequence_lengths = list(map(lambda i: 1 << i, range(*sequence_length_limits)))
    batch_sizes = list(
        map(
            lambda i: 1 << i,
            reversed(range(sequence_length_limits[1] - sequence_length_limits[0])),
        )
    )

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  sequence_length: {sequence_lengths}
  batch_size: {batch_sizes}
fixed_params:
  fwbw: {fwbw}
  rep: {30 if not debug else 10}
  warmup: {10 if not debug else 3}
  result_aggregation: "median"

x_axis_param: "sequence_length"

kernel_specs: {mlstm_triton_kernel_specs.format(fwbw=fwbw, num_heads=num_heads, head_dim_v=head_dim_v, head_dim_qk=head_dim_qk)}

benchmark_name: mlstm_triton_constant_tokens_sequence_{"fwbw" if fwbw else "fw"}
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_training_kernel_benchmark, output_folder)


def consttoken_benchmark_fla(
    output_folder: Path,
    fwbw: bool = True,
    debug: bool = False,
    sequence_length_limits=[9, 17],
    num_heads: int = 8,
):
    """
    Const token benchmark for the flash linear attention triton kernels.

    This benchmark uses head dimensions as they would be used in a 7B model (i.e. embedding dimension 4096).

    It uses 8 heads and the qk head dimension is half of the v head dimension.
    """
    embedding_dim = 4096
    num_heads = num_heads if num_heads is not None else 8
    head_dim_v = embedding_dim // num_heads
    head_dim_qk = embedding_dim // num_heads // 2
    sequence_lengths = list(map(lambda i: 1 << i, range(*sequence_length_limits)))
    batch_sizes = list(
        map(
            lambda i: 1 << i,
            reversed(range(sequence_length_limits[1] - sequence_length_limits[0])),
        )
    )

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  sequence_length: {sequence_lengths}
  batch_size: {batch_sizes}
fixed_params:
  fwbw: {fwbw}
  rep: {30 if not debug else 10}
  warmup: {10 if not debug else 3}
  result_aggregation: "median"

x_axis_param: "sequence_length"

kernel_specs: {fla_kernel_specs.format(fwbw=fwbw, num_heads=num_heads, head_dim_v=head_dim_v, head_dim_qk=head_dim_qk)}

benchmark_name: fla_constant_tokens_sequence_{"fwbw" if fwbw else "fw"}
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_training_kernel_benchmark, output_folder)


def consttoken_benchmark_lightning_attn2(
    output_folder: Path,
    fwbw: bool = True,
    debug: bool = False,
    sequence_length_limits=[9, 17],
    num_heads: int = 32,
):
    """
    Const token benchmark for the lightning attention 2 triton kernels.

    This benchmark uses head dimensions as they would be used in a 7B model (i.e. embedding dimension 4096).

    Notes:
    - Lightning attention does not support different head dimensions for qk and v.
    - It only supports head dimensions up to 128, i.e. num_heads = 32 or 64
    """
    embedding_dim = 4096
    num_heads = num_heads if num_heads is not None else 32
    head_dim_v = embedding_dim // num_heads
    head_dim_qk = embedding_dim // num_heads
    sequence_lengths = list(map(lambda i: 1 << i, range(*sequence_length_limits)))
    batch_sizes = list(
        map(
            lambda i: 1 << i,
            reversed(range(sequence_length_limits[1] - sequence_length_limits[0])),
        )
    )

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  sequence_length: {sequence_lengths}
  batch_size: {batch_sizes}
fixed_params:
  fwbw: {fwbw}
  rep: {30 if not debug else 10}
  warmup: {10 if not debug else 3}
  result_aggregation: "median"

x_axis_param: "sequence_length"

kernel_specs:
  - kernel_name: "lightning_attn2"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

benchmark_name: lightning_attn2_constant_tokens_sequence_{"fwbw" if fwbw else "fw"}
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_training_kernel_benchmark, output_folder)


def consttoken_benchmark_mamba(
    output_folder: Path,
    fwbw: bool = True,
    debug: bool = False,
    sequence_length_limits=[9, 17],
):
    """
    Const token benchmark for the Mamba kernels.

    This benchmark uses head dimensions as they would be used in a 7B model (i.e. embedding dimension 4096).
    It uses different number of heads and state / qk dimensions for Mamba and Mamba2 as used in their respective
    default settings for this model size.
    """
    embedding_dim = 4096
    sequence_lengths = list(map(lambda i: 1 << i, range(*sequence_length_limits)))
    batch_sizes = list(
        map(
            lambda i: 1 << i,
            reversed(range(sequence_length_limits[1] - sequence_length_limits[0])),
        )
    )

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  sequence_length: {sequence_lengths}
  batch_size: {batch_sizes}
fixed_params:
  fwbw: {fwbw}
  rep: {30 if not debug else 10}
  warmup: {10 if not debug else 3}
  result_aggregation: "median"

x_axis_param: "sequence_length"

kernel_specs:
  - kernel_name: "mamba"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      num_heads: 1
      head_dim_v: {2*embedding_dim}
      head_dim_qk: 16

  - kernel_name: "mamba"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      num_heads: 1
      head_dim_v: {embedding_dim}
      head_dim_qk: 16

  - kernel_name: "mamba2"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      num_heads: {2*embedding_dim//64}
      head_dim_v: 64
      head_dim_qk: 64
      chunk_size: 256
  - kernel_name: "mamba2"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      num_heads: {embedding_dim//64}
      head_dim_v: 64
      head_dim_qk: 64
      chunk_size: 256

  # - kernel_name: "mamba2"
  #   dtype: bfloat16
  #   fwbw: {fwbw}
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {2*embedding_dim//64}
  #     head_dim_v: 64
  #     head_dim_qk: 64
  #     chunk_size: 128

  # - kernel_name: "mamba2"
  #   dtype: bfloat16
  #   fwbw: {fwbw}
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {2*embedding_dim//64}
  #     head_dim_v: 64
  #     head_dim_qk: 64
  #     chunk_size: 64

  - kernel_name: "mamba2_noconv"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      num_heads: {2*embedding_dim//64}
      head_dim_v: 64
      head_dim_qk: 64
      chunk_size: 256
  - kernel_name: "mamba2_noconv"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      num_heads: {embedding_dim//64}
      head_dim_v: 64
      head_dim_qk: 64
      chunk_size: 256
  # - kernel_name: "mamba2_noconv"
  #   dtype: bfloat16
  #   fwbw: {fwbw}
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {2*embedding_dim//64}
  #     head_dim_v: 64
  #     head_dim_qk: 64
  #     chunk_size: 128
  # - kernel_name: "mamba2_noconv"
  #   dtype: bfloat16
  #   fwbw: {fwbw}
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {2*embedding_dim//64}
  #     head_dim_v: 64
  #     head_dim_qk: 64
  #     chunk_size: 64

benchmark_name: mamba_constant_tokens_sequence_{"fwbw" if fwbw else "fw"}
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_training_kernel_benchmark, output_folder)


def consttoken_benchmark_flashattn(
    output_folder: Path,
    fwbw: bool = True,
    debug: bool = False,
    sequence_length_limits=[9, 17],
):
    """
    This benchmark uses head dimensions as they would be used in a 7B model (i.e. embedding dimension 4096).

    For the attention kernels we use head dim 128 and 32 heads as the most common setting.
    Smaller head dimensions are usually favorable in terms of compute.
    """
    embedding_dim = 4096
    num_heads = 32
    head_dim_v = embedding_dim // num_heads  # 128
    head_dim_qk = embedding_dim // num_heads  # 128
    sequence_lengths = list(map(lambda i: 1 << i, range(*sequence_length_limits)))
    batch_sizes = list(
        map(
            lambda i: 1 << i,
            reversed(range(sequence_length_limits[1] - sequence_length_limits[0])),
        )
    )

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  sequence_length: {sequence_lengths}
  batch_size: {batch_sizes}
fixed_params:
  fwbw: {fwbw}
  warmup: {25 if not debug else 3}
  rep: {100 if not debug else 5}
  result_aggregation: "median"

x_axis_param: "sequence_length"

kernel_specs:
  - kernel_name: "torch_flash"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim_qk}
      head_dim_v: {head_dim_v}

  - kernel_name: "torch_cudnn"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim_qk}
      head_dim_v: {head_dim_v}

  - kernel_name: "flashattn3"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim_qk}
      head_dim_v: {head_dim_v}

benchmark_name: flashattn_constant_tokens_sequence_{"fwbw" if fwbw else "fw"}
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_training_kernel_benchmark, output_folder)


def run_multiple_benchmarks(
    output_dir: str = "./outputs_kernel_benchmarks",
    output_folder_suffix: str | None = None,
    debug: bool = False,
    consttoken_benchmark: str | None = None,
    headdim_benchmark: str | None = None,
    num_heads: int | None = None,
    half_qkdim: bool = True,
    batch_size: int | None = None,
):
    if consttoken_benchmark is not None:
        name = f"consttok_{consttoken_benchmark}"
        output_folder = setup_output_folder(
            output_dir,
            name_suffix=f"{name}_{output_folder_suffix}"
            if output_folder_suffix is not None
            else name,
        )

        if consttoken_benchmark == "mlstm_triton":
            consttoken_benchmark_mlstm_triton(
                output_folder,
                fwbw=True,
                debug=debug,
                num_heads=num_heads,
                half_qkdim=half_qkdim,
            )
            consttoken_benchmark_mlstm_triton(
                output_folder,
                fwbw=False,
                debug=debug,
                num_heads=num_heads,
                half_qkdim=half_qkdim,
            )
        elif consttoken_benchmark == "fla":
            consttoken_benchmark_fla(
                output_folder, fwbw=True, debug=debug, num_heads=num_heads
            )
            consttoken_benchmark_fla(
                output_folder, fwbw=False, debug=debug, num_heads=num_heads
            )
        elif consttoken_benchmark == "lightning_attn2":
            consttoken_benchmark_lightning_attn2(
                output_folder, fwbw=True, debug=debug, num_heads=num_heads
            )
            consttoken_benchmark_lightning_attn2(
                output_folder, fwbw=False, debug=debug, num_heads=num_heads
            )
        elif consttoken_benchmark == "mamba":
            consttoken_benchmark_mamba(output_folder, fwbw=True, debug=debug)
            consttoken_benchmark_mamba(output_folder, fwbw=False, debug=debug)
        elif consttoken_benchmark == "flashattn":
            consttoken_benchmark_flashattn(output_folder, fwbw=True, debug=debug)
            consttoken_benchmark_flashattn(output_folder, fwbw=False, debug=debug)
        else:
            raise ValueError(f"Unknown consttoken benchmark: {consttoken_benchmark}")

    if headdim_benchmark is not None:
        name = f"headdim_{headdim_benchmark}"
        output_folder = setup_output_folder(
            output_dir,
            name_suffix=f"{name}_{output_folder_suffix}"
            if output_folder_suffix is not None
            else name,
        )

        if headdim_benchmark == "mlstm_triton":
            headdim_benchmark_mlstm_triton(
                output_folder,
                fwbw=True,
                debug=debug,
                half_qkdim=half_qkdim,
                batch_size=batch_size,
            )
            headdim_benchmark_mlstm_triton(
                output_folder,
                fwbw=False,
                debug=debug,
                half_qkdim=half_qkdim,
                batch_size=batch_size,
            )
        elif headdim_benchmark == "fla":
            headdim_benchmark_fla(
                output_folder,
                fwbw=True,
                debug=debug,
                half_qkdim=half_qkdim,
                batch_size=batch_size,
            )
            headdim_benchmark_fla(
                output_folder,
                fwbw=False,
                debug=debug,
                half_qkdim=half_qkdim,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Unknown headdim benchmark: {headdim_benchmark}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_suffix",
        type=str,
        required=False,
        help="Suffix that is appended to the output folder of the benchmark results.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--consttoken_benchmark",
        type=str,
        required=False,
        default=None,
        help="Name of the const token benchmark.",
    )
    parser.add_argument("--num_heads", type=int, required=False)
    parser.add_argument(
        "--headdim_benchmark",
        type=str,
        required=False,
        default=None,
        help="Name of the head dim benchmark.",
    )
    parser.add_argument("--half_qkdim", type=int, default=1, required=False)
    parser.add_argument("--batch_size", type=int, required=False)

    args = parser.parse_args()
    print(args)

    run_multiple_benchmarks(
        output_folder_suffix=args.folder_suffix,
        debug=args.debug,
        consttoken_benchmark=args.consttoken_benchmark,
        num_heads=args.num_heads,
        headdim_benchmark=args.headdim_benchmark,
        half_qkdim=bool(args.half_qkdim),
        batch_size=args.batch_size,
    )

# Commands:
# python scripts/run_training_kernel_benchmarks.py --consttoken_benchmark fla --debug
# python scripts/run_training_kernel_benchmarks.py --consttoken_benchmark mlstm_triton --debug
# python scripts/run_training_kernel_benchmarks.py --consttoken_benchmark mamba --debug
# python scripts/run_training_kernel_benchmarks.py --consttoken_benchmark flashattn --debug
# python scripts/run_training_kernel_benchmarks.py --headdim_benchmark mlstm_triton --folder_suffix "v0"
