import argparse
from pathlib import Path

from dacite import from_dict
from omegaconf import OmegaConf

from mlstm_kernels.utils.benchmark.benchmarks.model_benchmarks import (
    create_mlstm_model_benchmark,
)
from mlstm_kernels.utils.benchmark.param_handling import BenchmarkConfig
from mlstm_kernels.utils.benchmark.run_benchmark import run_and_record_benchmarks
from mlstm_kernels.utils.benchmark.utils import setup_output_folder


def _throughput_benchmark(output_folder: Path, generation_length: int = 100):
    cfg_yaml = """
vary_type: grid
vary_params:
  batch_size: [1, 2, 4]#, 8] #[8, 16, 32, 64, 128, 256]
fixed_params:
  prefill_length: 0
  generation_length: 100

  embedding_dim: 4096
  num_heads: 8
  num_blocks: 32
  vocab_size: 50304

  chunkwise_kernel: chunkwise--triton_xl_chunk
  sequence_kernel: native_sequence__triton_step_fused
  step_kernel: triton_fused

  chunk_size: 128
  autocast_kernel_dtype: bfloat16

  rep: 1
  warmup: 1

x_axis_param: "batch_size"

kernel_specs:
  - model_name: "mLSTM"
    amp_enabled: True
    amp_dtype: bfloat16
    weight_dtype: float32
    use_torch_compile_model: False
    additional_params:
      use_torch_compile_generate: False
      inference_state_dtype: float32

  - model_name: "mLSTM"
    amp_enabled: True
    amp_dtype: bfloat16
    weight_dtype: bfloat16
    use_torch_compile_model: False
    additional_params:
      use_torch_compile_generate: False
      inference_state_dtype: bfloat16

#! torch.compile does not work atm
#   - model_name: "mLSTM"
#     amp_enabled: True
#     amp_dtype: bfloat16
#     weight_dtype: float32
#     use_torch_compile_model: True
#     additional_params:
#       use_torch_compile_generate: False
#       inference_state_dtype: float32
#   - model_name: "mLSTM"
#     amp_enabled: True
#     amp_dtype: bfloat16
#     weight_dtype: float32
#     use_torch_compile_model: True
#     additional_params:
#       use_torch_compile_generate: True
#       inference_state_dtype: float32

#   - model_name: "mLSTM"
#     amp_enabled: True
#     amp_dtype: bfloat16
#     weight_dtype: bfloat16
#     use_torch_compile_model: True
#     additional_params:
#       use_torch_compile_generate: True
#       inference_state_dtype: bfloat16

benchmark_name: "mlstm_7B_throughput"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )
    run_and_record_benchmarks(cfg, create_mlstm_model_benchmark, output_folder)


def run_multiple_benchmarks(
    output_dir: str = "./outputs_kernel_benchmarks",
    output_folder_suffix: str | None = None,
):
    output_folder = setup_output_folder(output_dir, name_suffix=output_folder_suffix)

    _throughput_benchmark(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_suffix",
        type=str,
        required=False,
        help="Suffix that is appended to the output folder of the benchmark results.",
    )

    args = parser.parse_args()
    print(args)

    run_multiple_benchmarks(output_folder_suffix=args.folder_suffix)
