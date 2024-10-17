from dataclasses import asdict
from pathlib import Path

from mlstm_kernels.benchmark_utils.benchmark import mLSTMBenchmark, run_benchmarks
from mlstm_kernels.benchmark_utils.param_handling import BenchmarkConfig
from mlstm_kernels.benchmark_utils.plot_results import plot_benchmark_result_table

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


def setup_output_folder(output_dir: str = "./outputs_kernel_benchmarks") -> Path:
    import logging
    import sys
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_folder = Path(output_dir) / timestamp

    output_folder.mkdir(parents=True, exist_ok=False)

    logfile = output_folder / "benchmark.log"
    file_handler = logging.FileHandler(filename=logfile)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        handlers=[file_handler, stdout_handler],
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        force=True,
    )
    LOGGER = logging.getLogger(__name__)
    LOGGER.info(f"Logging to {logfile}")
    return output_folder


def perform_single_benchmark(benchmark_config: BenchmarkConfig, output_folder: Path):
    import logging

    import tabulate

    LOGGER = logging.getLogger(__name__)

    LOGGER.info(f"Running benchmark: {benchmark_config.benchmark_name}")

    benchmark_folder = output_folder / benchmark_config.benchmark_name
    benchmark_folder.mkdir(parents=True, exist_ok=False)

    OmegaConf.save(OmegaConf.create(asdict(benchmark_config)), benchmark_folder / "config.yaml")

    result_df = run_benchmarks(benchmark_config)

    LOGGER.info(f"Results:\n{tabulate.tabulate(result_df, headers='keys', tablefmt='pretty')}")
    LOGGER.info(f"Saving results to {benchmark_folder}")
    result_df.to_csv(benchmark_folder / "results.csv")

    fig = plot_benchmark_result_table(result_df, benchmark_config.x_axis_param, title=benchmark_config.get_plot_title())
    fig.savefig(benchmark_folder / f"plot_{benchmark_config.benchmark_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(benchmark_folder / f"plot_{benchmark_config.benchmark_name}.pdf", bbox_inches="tight")
    fig.savefig(benchmark_folder / f"plot_{benchmark_config.benchmark_name}.svg", bbox_inches="tight")


def _head_dim_benchmark(output_folder: Path):
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
  - kernel_name: "parallel--triton"
    fwbw: True
    dtype: bfloat16
    
  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 64
  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 128
  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 32
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 64
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 128
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 256
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 512
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 1024

  # - kernel_name: "chunkwise--torch_autograd"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False

benchmark_name: "head_dim_7B"
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    perform_single_benchmark(cfg, output_folder)


def _head_dim_benchmark_seqlen1024(output_folder: Path):
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
  sequence_length: 1024
  batch_size: 1

x_axis_param: "head_dim_v"
  
kernel_specs:
  - kernel_name: "parallel--triton"
    fwbw: True
    dtype: bfloat16
    
  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 64
  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 128
  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 32
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 64
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 128
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 256
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 512
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 1024

  # - kernel_name: "chunkwise--torch_autograd"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False

benchmark_name: "head_dim_seqlen1024_7B"
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    perform_single_benchmark(cfg, output_folder)


def _head_dim_benchmark_half_qkdim(output_folder: Path):
    ### head dimension benchmark 7B
    head_dims_v = [64, 128, 256, 512, 1024, 2048]
    embedding_dim = 4096
    num_heads = [embedding_dim // head_dim for head_dim in head_dims_v]
    head_dims_qk = [head_dim // 2 for head_dim in head_dims_v]

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  num_heads: {num_heads}
  head_dim_qk: {head_dims_qk}
  head_dim_v: {head_dims_v}
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
  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 128
  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 32
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 64
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 128
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 256
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 512
  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 1024

  # - kernel_name: "chunkwise--torch_autograd"
  #   fwbw: True
  #   dtype: bfloat16
  #   use_torch_compile: False

benchmark_name: "head_dim_half_qk_7B"
"""

    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    perform_single_benchmark(cfg, output_folder)


def _sequence_length_benchmark(output_folder: Path):
    ### sequence length benchmark 7B
    sequence_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]

    cfg_yaml = f"""
vary_type: sequence
vary_params:
  sequence_length: {sequence_lengths}
fixed_params:
  batch_size: 1

x_axis_param: "sequence_length"
  
kernel_specs:
  - kernel_name: "torch_flash"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: 32
      head_dim_qk: 128
      head_dim_v: 128
  
  - kernel_name: "parallel--triton"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: 32
      head_dim_qk: 128
      head_dim_v: 128

  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 64
      num_heads: 8
      head_dim_qk: 512
      head_dim_v: 512

  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 64
      num_heads: 8
      head_dim_qk: 256
      head_dim_v: 512

benchmark_name: "sequence_length_7B"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    perform_single_benchmark(cfg, output_folder)


def _batch_size_benchmark(output_folder: Path):
    ### batch size benchmark 7B
    cfg_yaml = f"""
vary_type: sequence
vary_params:
  batch_size: [1, 2, 4, 8]
fixed_params:
  sequence_length: 8192

x_axis_param: "batch_size"

kernel_specs:
  - kernel_name: "torch_flash"
    fwbw: True
    dtype: bfloat16
    additional_params:
      num_heads: 32
      head_dim_qk: 128
      head_dim_v: 128
  
  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 64
      num_heads: 8
      head_dim_qk: 512
      head_dim_v: 512

  - kernel_name: "chunkwise--max_triton_v3"
    fwbw: True
    dtype: bfloat16
    additional_params:
      chunk_size: 64
      num_heads: 8
      head_dim_qk: 256
      head_dim_v: 512

  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 1024
      num_heads: 8
      head_dim_qk: 256
      head_dim_v: 512

  - kernel_name: "chunkwise--torch_ownbw"
    fwbw: True
    dtype: bfloat16
    use_torch_compile: False
    additional_params:
      chunk_size: 1024
      num_heads: 8
      head_dim_qk: 256
      head_dim_v: 512

benchmark_name: "batch_size_7B"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )

    perform_single_benchmark(cfg, output_folder)


def run_multiple_benchmarks(output_dir: str = "./outputs_kernel_benchmarks"):
    output_folder = setup_output_folder(output_dir)

    _head_dim_benchmark_seqlen1024(output_folder)
    # _head_dim_benchmark(output_folder)
    # _head_dim_benchmark_half_qkdim(output_folder)
    # _sequence_length_benchmark(output_folder)
    # _batch_size_benchmark(output_folder)


if __name__ == "__main__":
    run_multiple_benchmarks()
