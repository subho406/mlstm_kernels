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
  batch_size: [1, 4, 8, 16, 32, 64, 128, 256, 512]
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
  warmup: 0 #1

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


def _find_torchcompile_throughput_benchmark(
    output_folder: Path, generation_length: int = 100
):
    amp_enabled = True
    cfg_yaml = f"""
vary_type: grid
vary_params:
  batch_size: [64] #[1, 4, 8, 16, 32, 64]
fixed_params:
  prefill_length: 0
  generation_length: 100 #100

  embedding_dim: 4096
  num_heads: 8
  num_blocks: 32 #3 #32
  vocab_size: 50304

  chunkwise_kernel: chunkwise--triton_xl_chunk
  sequence_kernel: native_sequence__triton_step_fused
  step_kernel: triton_fused

  chunk_size: 128
  inference_state_dtype: float32
  autocast_kernel_dtype: bfloat16

  rep: 1
  warmup: 0

x_axis_param: "batch_size"

kernel_specs:
  - model_name: "mLSTM"
    amp_enabled: {amp_enabled}
    amp_dtype: bfloat16
    weight_dtype: float32
    use_torch_compile_model: False
    additional_params:
      use_torch_compile_generate: False
      inference_state_dtype: float32

  - model_name: "mLSTM"
    amp_enabled: {amp_enabled}
    amp_dtype: bfloat16
    weight_dtype: float32
    use_torch_compile_model: False
    additional_params:
      use_torch_compile_generate: True
      inference_state_dtype: float32

  - model_name: "mLSTM"
    amp_enabled: {amp_enabled}
    amp_dtype: bfloat16
    weight_dtype: float32
    use_torch_compile_model: True
    additional_params:
      use_torch_compile_generate: False
      inference_state_dtype: float32

  - model_name: "mLSTM"
    amp_enabled: {amp_enabled}
    amp_dtype: bfloat16
    weight_dtype: float32
    use_torch_compile_model: True
    additional_params:
      use_torch_compile_generate: True
      inference_state_dtype: float32

benchmark_name: "mlstm_torchcompile_throughput"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )
    run_and_record_benchmarks(cfg, create_mlstm_model_benchmark, output_folder)


def _find_weight_dtype_throughput_benchmark(
    output_folder: Path,
    generation_length: int = 100,
    batch_size: int = 64,
    use_torch_compile: bool = False,
    amp_enabled: bool = False,
):
    amp_dtype = "bfloat16"
    cfg_yaml = f"""
vary_type: grid
vary_params:
  batch_size: [{batch_size}] #[1, 4, 8, 16, 32, 64]
fixed_params:
  prefill_length: 0
  generation_length: {generation_length} #100

  embedding_dim: 4096
  num_heads: 8
  num_blocks: 32 #3 #32
  vocab_size: 50304

  chunkwise_kernel: chunkwise--triton_xl_chunk
  sequence_kernel: native_sequence__triton_step_fused
  step_kernel: native #triton_fused

  chunk_size: 128
  inference_state_dtype: float32
  autocast_kernel_dtype: bfloat16

  rep: 1
  warmup: 0

x_axis_param: "batch_size"

kernel_specs:
  - model_name: "mLSTM"
    amp_enabled: {amp_enabled}
    amp_dtype: {amp_dtype}
    weight_dtype: float32
    use_torch_compile_model: {use_torch_compile}
    additional_params:
      use_torch_compile_generate: {use_torch_compile}
      inference_state_dtype: float32

  - model_name: "mLSTM"
    amp_enabled: {amp_enabled}
    amp_dtype: {amp_dtype}
    weight_dtype: float32
    use_torch_compile_model: {use_torch_compile}
    additional_params:
      use_torch_compile_generate: {use_torch_compile}
      inference_state_dtype: bfloat16

  - model_name: "mLSTM"
    amp_enabled: {amp_enabled}
    amp_dtype: {amp_dtype}
    weight_dtype: bfloat16
    use_torch_compile_model: {use_torch_compile}
    additional_params:
      use_torch_compile_generate: {use_torch_compile}
      inference_state_dtype: bfloat16

  - model_name: "mLSTM"
    amp_enabled: {amp_enabled}
    amp_dtype: {amp_dtype}
    weight_dtype: bfloat16
    use_torch_compile_model: {use_torch_compile}
    additional_params:
      use_torch_compile_generate: {use_torch_compile}
      inference_state_dtype: float32

benchmark_name: "mlstm_torchsetup_throughput__amp{amp_enabled}-torchcompile{use_torch_compile}-genlength{generation_length}-batchsize{batch_size}"
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

    # _throughput_benchmark(output_folder)

    # Finding the correct setup:
    # _find_torchcompile_throughput_benchmark(output_folder)
    _find_weight_dtype_throughput_benchmark(
        output_folder, use_torch_compile=True, batch_size=1
    )


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
