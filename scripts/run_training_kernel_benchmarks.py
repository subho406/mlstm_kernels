from pathlib import Path

from mlstm_kernels.benchmark_utils.benchmarks.training_kernel_benchmarks import (
    create_training_kernel_benchmark,
    mLSTMBenchmark,
)
from mlstm_kernels.benchmark_utils.param_handling import BenchmarkConfig
from mlstm_kernels.benchmark_utils.run_benchmark import run_and_record_benchmarks
from mlstm_kernels.benchmark_utils.utils import setup_output_folder

from dacite import from_dict
from omegaconf import OmegaConf


def debug_single_benchmark():
    print("Running benchmark...")
    mlstm_benchmark = mLSTMBenchmark(
        batch_size=1,
        sequence_length=256,
        num_heads=8,
        head_dim_qk=64,
        kernel_name="chunkwise--max_triton_v3",
    )
    mlstm_benchmark.setup_benchmark()
    time = mlstm_benchmark.run_benchmark()

    print(f"Benchmark finished. time: {time} ms")


def _head_dim_benchmark(output_folder: Path, half_qkdim=False, seq_len: int = 8192, batch_size: int = 1):
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

x_axis_param: "head_dim_v"

kernel_specs:
  - kernel_name: "parallel--triton"
    fwbw: True
    dtype: bfloat16
  ####
  #? chunk size 64 is optimal
  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 64

  # - kernel_name: "chunkwise--max_triton_v3"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     chunk_size: 128
  # - kernel_name: "chunkwise--max_triton_v3"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     chunk_size: 32
  ####
  - kernel_name: "max_triton_v5xlchunksize"
    fwbw: True
    dtype: bfloat16
    additional_params:
      siz_b_L_parallel: 64
      siz_b_L_loop: 64
      siz_b_DH_parallel: 128
      siz_b_DH_loop: 64

      num_warps_intra: 4
      num_warps_inter: 4
      num_stages_intra: 1
      num_stages_inter: 1

      chunk_size_intra: 128
      chunk_size_inter: 128
  ####
  # - kernel_name: "chunkwise--torch_ownbw"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     chunk_size: 64

  # - kernel_name: "chunkwise--torch_ownbw"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     chunk_size: 128
  # - kernel_name: "chunkwise--torch_ownbw"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     chunk_size: 256
  # - kernel_name: "chunkwise--torch_ownbw"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     chunk_size: 512
  # - kernel_name: "chunkwise--torch_ownbw"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  #   additional_params:
  #     chunk_size: 1024

  # - kernel_name: "chunkwise--torch_autograd"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False
  ####

benchmark_name: {bench_name}
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_training_kernel_benchmark, output_folder)


def _head_dim_benchmark_no_slicing(output_folder: Path):
    ### head dimension benchmark 7B
    head_dims = [64, 128, 256, 512, 1024, 2048]
    embedding_dim = 4096
    num_heads = [embedding_dim // head_dim for head_dim in head_dims]

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  num_heads: {num_heads}
  head_dim_qk: {head_dims}
  head_dim_v: {head_dims}
fixed_params:
  sequence_length: 8192
  batch_size: 1

x_axis_param: "head_dim_v"

kernel_specs:

  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 64
  # - kernel_name: "chunkwise--max_triton_v3"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     chunk_size: 128
  # - kernel_name: "chunkwise--max_triton_v3"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     chunk_size: 32

  - kernel_name: "chunkwise--max_triton_v3noslice"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 64
  # - kernel_name: "chunkwise--max_triton_v3noslice"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     chunk_size: 128
  # - kernel_name: "chunkwise--max_triton_v3noslice"
  #   fwbw: True
  #   dtype: bfloat16
  #   additional_params:
  #     chunk_size: 32



benchmark_name: "head_dim_no_slicing_7B"
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

  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim}
      head_dim_v: {head_dim}
      chunk_size: 64

  - kernel_name: "max_triton_v5xlchunksize"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim}
      head_dim_v: {head_dim}

      siz_b_L_parallel: 64
      siz_b_L_loop: 64
      siz_b_DH_parallel: 128
      siz_b_DH_loop: 64

      num_warps_intra: 4
      num_warps_inter: 4
      num_stages_intra: 1
      num_stages_inter: 1

      chunk_size_intra: 128
      chunk_size_inter: 128

  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim//2}
      head_dim_v: {head_dim}
      chunk_size: 64

  - kernel_name: "max_triton_v5xlchunksize"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim//2}
      head_dim_v: {head_dim}

      siz_b_L_parallel: 64
      siz_b_L_loop: 64
      siz_b_DH_parallel: 128
      siz_b_DH_loop: 64

      num_warps_intra: 4
      num_warps_inter: 4
      num_stages_intra: 1
      num_stages_inter: 1

      chunk_size_intra: 128
      chunk_size_inter: 128

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
):
    bench_name_params = f"nh_{num_heads}_hd_{head_dim}"

    ### batch size benchmark 7B
    cfg_yaml = f"""
vary_type: sequence
vary_params:
  batch_size: [1, 2, 4, 8]
fixed_params:
  sequence_length: {seq_len}

x_axis_param: "batch_size"

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

  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim}
      head_dim_v: {head_dim}
      chunk_size: 64

  - kernel_name: "max_triton_v5xlchunksize"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim}
      head_dim_v: {head_dim}

      siz_b_L_parallel: 64
      siz_b_L_loop: 64
      siz_b_DH_parallel: 128
      siz_b_DH_loop: 64

      num_warps_intra: 4
      num_warps_inter: 4
      num_stages_intra: 1
      num_stages_inter: 1

      chunk_size_intra: 128
      chunk_size_inter: 128

  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim//2}
      head_dim_v: {head_dim}
      chunk_size: 64

  - kernel_name: "max_triton_v5xlchunksize"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: {num_heads}
      head_dim_qk: {head_dim//2}
      head_dim_v: {head_dim}

      siz_b_L_parallel: 64
      siz_b_L_loop: 64
      siz_b_DH_parallel: 128
      siz_b_DH_loop: 64

      num_warps_intra: 4
      num_warps_inter: 4
      num_stages_intra: 1
      num_stages_inter: 1

      chunk_size_intra: 128
      chunk_size_inter: 128


benchmark_name: "batch_size_7B--{bench_name_params}"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    run_and_record_benchmarks(cfg, create_training_kernel_benchmark, output_folder)


def run_multiple_benchmarks(output_dir: str = "./outputs_kernel_benchmarks"):
    output_folder = setup_output_folder(output_dir)

    # _sequence_length_benchmark(output_folder, batch_size=1, num_heads=16, head_dim=256)
    # _batch_size_benchmark(output_folder, seq_len=8192, num_heads=16, head_dim=256)
    # _sequence_length_benchmark(output_folder, batch_size=1, num_heads=8, head_dim=512)
    # _batch_size_benchmark(output_folder, seq_len=8192, num_heads=8, head_dim=512)

    _head_dim_benchmark(output_folder, half_qkdim=False, seq_len=8192, batch_size=1)
    _head_dim_benchmark(output_folder, half_qkdim=True, seq_len=8192, batch_size=1)
    # _head_dim_benchmark_no_slicing(output_folder)


if __name__ == "__main__":
    run_multiple_benchmarks()

if __name__ == "__main__":
    run_multiple_benchmarks()
