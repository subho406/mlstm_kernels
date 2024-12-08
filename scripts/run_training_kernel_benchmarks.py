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


def _head_dim_benchmark(
    output_folder: Path, half_qkdim=False, seq_len: int = 8192, batch_size: int = 1,
    debug: bool = False,
):
    ### head dimension benchmark 7B
    head_dims_v = [64, 128, 256, 512, 1024, 2048]
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
  sequence_length: {seq_len}
  batch_size: {batch_size}
  rep: {100 if not debug else 10}
  warmup: {25 if not debug else 10}

x_axis_param: "head_dim_v"

kernel_specs:
  - kernel_name: "parallel--triton_limit_headdim"
    fwbw: True
    dtype: bfloat16
  ####
  #? chunk size 64 is optimal
  - kernel_name: "chunkwise--triton_limit_chunk"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 64

  # - kernel_name: "chunkwise--triton_limit_chunk"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     chunk_size: 128
  # - kernel_name: "chunkwise--triton_limit_chunk"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     chunk_size: 32
  ####
  - kernel_name: "chunkwise--triton_xl_chunk"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 128
  ####
  # - kernel_name: "chunkwise--native_custbw"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     chunk_size: 64

  - kernel_name: "chunkwise--native_custbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 128
  - kernel_name: "chunkwise--native_custbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 256
  - kernel_name: "chunkwise--native_custbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 512
  - kernel_name: "chunkwise--native_custbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 1024

  # - kernel_name: "chunkwise--native_autograd"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  ####
  - kernel_name: "chunkwise--native_custbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 64

  - kernel_name: "chunkwise--native_custbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 128
  - kernel_name: "chunkwise--native_custbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 256
  - kernel_name: "chunkwise--native_custbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 512
  - kernel_name: "chunkwise--native_custbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 1024

  - kernel_name: "chunkwise--native_autograd"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 256
  ####
  - kernel_name: "chunk_gla"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
  - kernel_name: "fused_chunk_gla"
    fwbw: True
    dtype: bfloat16
  - kernel_name: "chunk_simple_gla"
    fwbw: True
    dtype: bfloat16
  - kernel_name: "mamba"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
  - kernel_name: "mamba2"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
  - kernel_name: "mamba2_noconv"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
  - kernel_name: "flashattn3"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
  


benchmark_name: {bench_name}
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_training_kernel_benchmark, output_folder)


def _sequence_length_benchmark(
    output_folder: Path,
    batch_size: int = 1,
    num_heads: int = 16,
    head_dim: int = 256,
    debug: bool = False,
):
    ### sequence length benchmark 7B
    sequence_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]

    bench_name_params = f"nh_{num_heads}_hd_{head_dim}"

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  sequence_length: {sequence_lengths}
fixed_params:
  batch_size: {batch_size}
  rep: {100 if not debug else 10}
  warmup: {25 if not debug else 10}

x_axis_param: "sequence_length"

kernel_specs:
  - kernel_name: "torch_flash"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: 32
      head_dim_qk: 128
      head_dim_v: 128

  # - kernel_name: "parallel--triton"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     num_heads: 32
  #     head_dim_qk: 128
  #     head_dim_v: 128

  - kernel_name: "chunkwise--triton_limit_chunk"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim}
      head_dim_v: {head_dim}
      chunk_size: 64

  - kernel_name: "chunkwise--triton_xl_chunk"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim}
      head_dim_v: {head_dim}

      chunk_size: 128

  - kernel_name: "chunkwise--triton_limit_chunk"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim//2}
      head_dim_v: {head_dim}
      chunk_size: 64

  - kernel_name: "chunkwise--triton_xl_chunk"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim//2}
      head_dim_v: {head_dim}
      chunk_size: 128
  
  # - kernel_name: "chunk_gla"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: {head_dim//2}
  #     head_dim_v: {head_dim}

  #   use_torch_compile: False
  # - kernel_name: "fused_chunk_gla"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: {head_dim}
  #     head_dim_v: {head_dim}

  # - kernel_name: "chunk_simple_gla"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: {head_dim}
  #     head_dim_v: {head_dim}

  # - kernel_name: "mamba"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: 16  # standard mamba setting
  #     head_dim_v: {head_dim}

  # - kernel_name: "mamba2"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: {head_dim}
  #     head_dim_v: {head_dim}

  # - kernel_name: "mamba2_noconv"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: {head_dim}
  #     head_dim_v: {head_dim}

  # - kernel_name: "flashattn3"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: {head_dim}
  #     head_dim_v: {head_dim}

benchmark_name: "sequence_length_7B--{bench_name_params}"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_training_kernel_benchmark, output_folder)


def _batch_size_benchmark(
    output_folder: Path,
    seq_len: int = 8192,
    num_heads: int = 16,
    head_dim: int = 256,
    fwbw: bool = True,
    debug: bool = False,
):
    bench_name_params = f"nh_{num_heads}_hd_{head_dim}_seq_{seq_len}"

    ### batch size benchmark 7B
    cfg_yaml = f"""
vary_type: sequence
vary_params:
  batch_size: [1, 2, 4, 8] #  16, 32]
fixed_params:
  sequence_length: {seq_len}
  fwbw: {fwbw}
  rep: {100 if not debug else 10}
  warmup: {25 if not debug else 10}

x_axis_param: "batch_size"

kernel_specs:
  - kernel_name: "torch_flash"
    dtype: bfloat16
    additional_params:
      num_heads: 32
      head_dim_qk: 128
      head_dim_v: 128

  # - kernel_name: "parallel--triton"
  #   fwbw: {fwbw}
  #   dtype: bfloat16
  #   additional_params:
  #     num_heads: 32
  #     head_dim_qk: 128
  #     head_dim_v: 128

  - kernel_name: "chunkwise--triton_limit_chunk"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim}
      head_dim_v: {head_dim}
      chunk_size: 64

  - kernel_name: "chunkwise--triton_xl_chunk"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim}
      head_dim_v: {head_dim}

      chunk_size: 128

      # siz_b_L_parallel: 64
      # siz_b_L_loop: 64
      # siz_b_DH_parallel: 128
      # siz_b_DH_loop: 64

      # num_warps_intra: 4
      # num_warps_inter: 4
      # num_stages_intra: 1
      # num_stages_inter: 1

      # chunk_size_intra: 128
      # chunk_size_inter: 128

  - kernel_name: "chunkwise--triton_limit_chunk"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim//2}
      head_dim_v: {head_dim}
      chunk_size: 64

  - kernel_name: "chunkwise--triton_xl_chunk"
    fwbw: {fwbw}
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim//2}
      head_dim_v: {head_dim}

      chunk_size: 128
      # siz_b_L_parallel: 64
      # siz_b_L_loop: 64
      # siz_b_DH_parallel: 128
      # siz_b_DH_loop: 64

      # num_warps_intra: 4
      # num_warps_inter: 4
      # num_stages_intra: 1
      # num_stages_inter: 1

      # chunk_size_intra: 128
      # chunk_size_inter: 128
  
  # - kernel_name: "chunk_gla"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: {head_dim}
  #     head_dim_v: {head_dim}

  # - kernel_name: "fused_chunk_gla"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: {head_dim}
  #     head_dim_v: {head_dim}

  # - kernel_name: "chunk_simple_gla"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: {head_dim}
  #     head_dim_v: {head_dim}

  # - kernel_name: "mamba"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: 16  # standard mamba setting
  #     head_dim_v: {head_dim}

  # - kernel_name: "mamba2"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: {head_dim}
  #     head_dim_v: {head_dim}

  # - kernel_name: "mamba2_noconv"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: {head_dim}
  #     head_dim_v: {head_dim}

  # - kernel_name: "flashattn3"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     num_heads: {num_heads}
  #     head_dim_qk: {head_dim}
  #     head_dim_v: {head_dim}


benchmark_name: "batch_size_7B--{bench_name_params}"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_training_kernel_benchmark, output_folder)



def _consttoken_benchmark(
    output_folder: Path,
    fwbw: bool = True,
    debug: bool = False,
    sequence_length_limits = [9, 17]
):
    """
    This benchmark uses head dimensions as they would be used in a 7B model (i.e. embedding dimension 4096).

    It uses different number of heads and state / qk dimensions for Mamba and Mamba2 as used in their respective
    default settings for this model size.
    For the attention kernels we use head dim 128 and 32 heads as the most common setting.
    Smaller head dimensions are usually favorable in terms of compute.
    """
    embedding_dim = 4096
    num_heads = 8
    head_dim_v = embedding_dim // num_heads
    head_dim_qk = embedding_dim // num_heads //2
    sequence_lengths = list(map(lambda i: 1<<i, range(*sequence_length_limits)))
    batch_sizes = list(map(lambda i: 1<<i, reversed(range(sequence_length_limits[1] - sequence_length_limits[0]))))

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  sequence_length: {sequence_lengths}
  batch_size: {batch_sizes}
fixed_params:
  fwbw: {fwbw}
  rep: {100 if not debug else 10}
  warmup: {25 if not debug else 10}

x_axis_param: "sequence_length"

kernel_specs:
  - kernel_name: "torch_flash"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      num_heads: 32
      head_dim_qk: 128
      head_dim_v: 128
  - kernel_name: "torch_cudnn"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      num_heads: 32
      head_dim_qk: 128
      head_dim_v: 128
  - kernel_name: "parallel--triton_limit_headdim"
    fwbw: {fwbw}
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_v}

    
  ####
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
  ####
  - kernel_name: "chunkwise--triton_xl_chunk"
    dtype: bfloat16
    fwbw: {fwbw}
    additional_params:
      chunk_size: 128
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  ####
  # - kernel_name: "chunkwise--native_custbw"
  #   dtype: bfloat16
  #   fwbw: {fwbw}
  #   use_torch_compile: False
  #   additional_params:
  #     chunk_size: 64

  - kernel_name: "chunkwise--native_custbw"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      chunk_size: 128
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  - kernel_name: "chunkwise--native_custbw"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      chunk_size: 256
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  # - kernel_name: "chunkwise--native_custbw"
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     chunk_size: 512
  #     num_heads: {num_heads}
  #     head_dim_v: {head_dim_v}
  #     head_dim_qk: {head_dim_qk}

  # - kernel_name: "chunkwise--native_custbw"
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     chunk_size: 1024
  #     num_heads: {num_heads}
  #     head_dim_v: {head_dim_v}
  #     head_dim_qk: {head_dim_qk}


  # - kernel_name: "chunkwise--native_autograd"
  #   dtype: bfloat16
  #   use_torch_compile: False
  ####
  - kernel_name: "chunkwise--native_custbw"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      chunk_size: 64
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}


  - kernel_name: "chunkwise--native_custbw"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      chunk_size: 128
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  - kernel_name: "chunkwise--native_custbw"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      chunk_size: 256
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  # - kernel_name: "chunkwise--native_custbw"
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     chunk_size: 512
  #     num_heads: {num_heads}
  #     head_dim_v: {head_dim_v}
  #     head_dim_qk: {head_dim_qk}
      
  # - kernel_name: "chunkwise--native_custbw"
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     chunk_size: 1024
  #     head_dim_v: {head_dim_v}
  #     head_dim_qk: {head_dim_qk}


  - kernel_name: "chunkwise--native_autograd"
    dtype: bfloat16
    use_torch_compile: False
    fwbw: {fwbw}
    additional_params:
      chunk_size: 256
      num_heads: {num_heads}
      head_dim_v: {head_dim_v}
      head_dim_qk: {head_dim_qk}

  ####
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


  - kernel_name: "mamba"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      num_heads: 1
      head_dim_v: {2*embedding_dim}
      head_dim_qk: 16

  - kernel_name: "mamba2"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      num_heads: {2*embedding_dim//64}
      head_dim_v: 64
      head_dim_qk: 64

  - kernel_name: "mamba2_noconv"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      num_heads: {2*embedding_dim//64}
      head_dim_v: 64
      head_dim_qk: 64

  - kernel_name: "flashattn3"
    dtype: bfloat16
    fwbw: {fwbw}
    use_torch_compile: False
    additional_params:
      num_heads: {embedding_dim//128}
      head_dim_v: 128
      head_dim_qk: 128

benchmark_name: constant_tokens_sequence_{"fwbw" if fwbw else "fw"}
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
):
    output_folder = setup_output_folder(output_dir, name_suffix=output_folder_suffix)

    # _sequence_length_benchmark(output_folder, batch_size=1, num_heads=16, head_dim=256, debug=debug,)
    # _batch_size_benchmark(output_folder, seq_len=8192, num_heads=16, head_dim=256, debug=debug,)
    # _sequence_length_benchmark(output_folder, batch_size=1, num_heads=8, head_dim=512, debug=debug,)
    # _batch_size_benchmark(
    #     output_folder, seq_len=128, num_heads=8, head_dim=512, fwbw=False, debug=debug,
    # )
    # _batch_size_benchmark(
    #     output_folder, seq_len=512, num_heads=8, head_dim=512, fwbw=False, debug=debug,
    # )
    # _batch_size_benchmark(
    #     output_folder, seq_len=1024, num_heads=8, head_dim=512, fwbw=False, debug=debug,
    # )
    # _batch_size_benchmark(
    #     output_folder, seq_len=2048, num_heads=8, head_dim=512, fwbw=False, debug=debug,
    # )
    # _batch_size_benchmark(
    #     output_folder, seq_len=4096, num_heads=8, head_dim=512, fwbw=False, debug=debug,
    # )
    # _head_dim_benchmark(output_folder, half_qkdim=False, seq_len=8192, batch_size=1, debug=debug)
    # _head_dim_benchmark(output_folder, half_qkdim=True, seq_len=8192, batch_size=1, debug=debug)

    # debug:
    # _head_dim_benchmark(output_folder, half_qkdim=False, seq_len=2048, batch_size=1, debug=debug)

    _consttoken_benchmark(output_folder, fwbw=True, debug=debug)
    _consttoken_benchmark(output_folder, fwbw=False, debug=debug)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_suffix",
        type=str,
        required=False,
        help="Suffix that is appended to the output folder of the benchmark results.",
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    print(args)

    run_multiple_benchmarks(output_folder_suffix=args.folder_suffix, debug=args.debug)
