import argparse
import logging
import pprint
import typing
from pathlib import Path
from typing import Literal

from dacite import from_dict
from omegaconf import OmegaConf

from mlstm_kernels.utils.benchmark.benchmarks.huggingface_model_benchmark import (
    create_hf_model_benchmark,
)
from mlstm_kernels.utils.benchmark.param_handling import BenchmarkConfig
from mlstm_kernels.utils.benchmark.run_benchmark import run_and_record_benchmarks
from mlstm_kernels.utils.benchmark.utils import setup_output_folder

LOGGER = logging.getLogger(__name__)


def _time_to_first_token_benchmark(
    output_folder: Path,
    batch_size: int = 8,
    generation_length: int = 1,
    use_torch_compile_model: bool = True,
    prefill_length: int = 0,
    weight_dtype: str = "bfloat16",
):
    cfg_yaml = f"""
vary_type: grid
vary_params:
  prefill_length: [0, 2048] #[0, 128, 512, 2048]
fixed_params:
#   prefill_length: {prefill_length}
  batch_size: {batch_size}
  generation_length: {generation_length}

  rep: 1
  warmup: 1 #1
  benchmark_fn_context_manager: "inference_mode"
  #

x_axis_param: "prefill_length"

kernel_specs:
#   - model_name: "xlstm"
#     weight_dtype: {weight_dtype}
#     use_torch_compile_model: True #{use_torch_compile_model}
#     additional_params:
#       use_cuda_graphs_model: False
#       use_cuda_graphs_generate: True

#       inference_state_dtype: bfloat16
#       embedding_dim: 4096
#       num_heads: 8
#       num_blocks: 32 #3 #32
#       vocab_size: 50304
#       weight_mode: "fused" # or "single"

#       chunkwise_kernel: chunkwise--triton_xl_chunk
#       sequence_kernel: native_sequence__triton_step_fused
#       step_kernel: triton_fused

#       chunk_size: 128
#       autocast_kernel_dtype: bfloat16


#   - model_name: "mlstm_simple"
#     weight_dtype: {weight_dtype}
#     use_torch_compile_model: True #{use_torch_compile_model}
#     additional_params:
#       use_cuda_graphs_model: False
#       use_cuda_graphs_generate: True

#       inference_state_dtype: bfloat16
#       embedding_dim: 4096
#       num_heads: 8
#       num_blocks: 32 #3 #32
#       vocab_size: 50304
#       weight_mode: "fused" # or "single"

#       chunkwise_kernel: chunkwise--triton_xl_chunk
#       sequence_kernel: native_sequence__triton_step_fused
#       step_kernel: triton_fused

#       chunk_size: 128
#       autocast_kernel_dtype: bfloat16

  # - model_name: "ministral8b"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: True #{use_torch_compile_model}
  #   additional_params:
  #     num_blocks: 2
  #     use_torch_compile_generate: False
  #     apply_overrides_to_hf_model: True

  # - model_name: "ministral8b"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False #{use_torch_compile_model}
  #   additional_params:
  #     num_blocks: 2
  #     use_torch_compile_generate: False
  #     apply_overrides_to_hf_model: True

#   - model_name: "llama3"
#     weight_dtype: {weight_dtype}
#     use_torch_compile_model: False #{use_torch_compile_model}
#     additional_params:
#       use_torch_compile_generate: False
#       use_cuda_graphs_generate: True
#       use_cuda_graphs_model: False

#   - model_name: "llama3"
#     weight_dtype: {weight_dtype}
#     use_torch_compile_model: True #{use_torch_compile_model}
#     additional_params:
#       use_torch_compile_generate: False
#       use_cuda_graphs_generate: True
#       use_cuda_graphs_model: False

#   - model_name: "llama2"
#     weight_dtype: {weight_dtype}
#     use_torch_compile_model: True #{use_torch_compile_model}
#     additional_params:
#       use_cuda_graphs_generate: True
#       use_cuda_graphs_model: False



  # - model_name: "codestral_mamba"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: True #{use_torch_compile_model}
  #   additional_params:
  #     num_blocks: 2
  #     use_torch_compile_generate: False
  #     apply_overrides_to_hf_model: True

  - model_name: "codestral_mamba"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: False #{use_torch_compile_model}
    additional_params:
      num_blocks: 2
      use_torch_compile_generate: False
      use_cuda_graphs_generate: False
      use_cuda_graphs_model: False

      apply_overrides_to_hf_model: True


#   - model_name: "falcon_mamba"
#     weight_dtype: {weight_dtype}
#     use_torch_compile_model: False #{use_torch_compile_model}
#     additional_params:
#       use_cuda_graphs_generate: True
#       use_cuda_graphs_model: False


  # - model_name: "falcon_mamba"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False #{use_torch_compile_model}
  #   additional_params:
  #     num_blocks: 2
  #     use_torch_compile_generate: False
  #     apply_overrides_to_hf_model: True


#   - model_name: "zamba2"
#     weight_dtype: {weight_dtype}
#     use_torch_compile_model: False #{use_torch_compile_model}
#     # additional_params:
#     #   num_blocks: 2
#     #   use_torch_compile_generate: False
#     #   apply_overrides_to_hf_model: True

  # - model_name: "zamba2"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False #{use_torch_compile_model}
  #   additional_params:
  #     num_blocks: 2
  #     use_torch_compile_generate: False
  #     apply_overrides_to_hf_model: True

############## NO TORCH COMPILE ####################
  # - model_name: "mlstm_simple"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False #{use_torch_compile_model}
  #   additional_params:
  #     use_torch_compile_generate: False
  #     inference_state_dtype: bfloat16
  #     embedding_dim: 4096
  #     num_heads: 8
  #     num_blocks: 32 #3 #32
  #     vocab_size: 50304

  #     chunkwise_kernel: chunkwise--triton_xl_chunk
  #     sequence_kernel: native_sequence__triton_step_fused
  #     step_kernel: triton_fused

  #     chunk_size: 128
  #     autocast_kernel_dtype: bfloat16

  # - model_name: "xlstm"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False #{use_torch_compile_model}
  #   additional_params:
  #     inference_state_dtype: bfloat16
  #     embedding_dim: 4096
  #     num_heads: 8
  #     num_blocks: 32 #3 #32
  #     vocab_size: 50304

  #     chunkwise_kernel: chunkwise--triton_xl_chunk
  #     sequence_kernel: native_sequence__triton_step_fused
  #     step_kernel: triton_fused

  #     chunk_size: 128
  #     autocast_kernel_dtype: bfloat16

  # - model_name: "llama3"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False #{use_torch_compile_model}

  # - model_name: "falcon_mamba"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False #{use_torch_compile_model}

  # - model_name: "zamba2"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False #{use_torch_compile_model}

benchmark_name: "hf_7B_timtofirsttok__bs{batch_size}_gl{generation_length}_tc{use_torch_compile_model}_weightdtype{weight_dtype}"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )
    LOGGER.info(f"Running benchmark with config:\n{pprint.pformat(cfg)}")
    run_and_record_benchmarks(
        cfg,
        create_hf_model_benchmark,
        output_folder,
        benchmark_type="model",
        setup_model_on_every_param_combination=False,
    )


BenchmarkType = Literal["ttft", "gen_time", "throughput"]


def run_multiple_benchmarks(
    output_dir: str = "./outputs_kernel_benchmarks_debug",
    benchmark_type: BenchmarkType = "ttft",
    output_folder_suffix: str | None = None,
    batch_size: int = 1,
    generation_length: int = 100,
    prefill_length: int = 0,
):
    full_folder_suffix = (
        f"{benchmark_type}__{output_folder_suffix}"
        if output_folder_suffix
        else benchmark_type
    )
    full_folder_suffix = f"{output_folder_suffix}__bs{batch_size}_gl{generation_length}_pl{prefill_length}"
    output_folder = setup_output_folder(output_dir, name_suffix=full_folder_suffix)

    _time_to_first_token_benchmark(
        output_folder,
        batch_size=batch_size,
        generation_length=generation_length,
        prefill_length=prefill_length,
        use_torch_compile_model=True,
        weight_dtype="bfloat16",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        required=False,
        help=f"The benchmark type. One of {typing.get_args(BenchmarkType)}",
    )
    parser.add_argument(
        "--folder_suffix",
        type=str,
        required=False,
        help="Suffix that is appended to the output folder of the benchmark results.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1,
        help="Batch size for the benchmark.",
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        required=False,
        default=10,
        help="Generation length for the benchmark.",
    )
    parser.add_argument(
        "--prefill_length",
        type=int,
        required=False,
        default=64,
        help="Prefill length for the benchmark.",
    )

    args = parser.parse_args()
    print(args)

    run_multiple_benchmarks(
        output_folder_suffix=args.folder_suffix,
        benchmark_type=args.benchmark,
        batch_size=args.batch_size,
        generation_length=args.generation_length,
        prefill_length=args.prefill_length,
    )
