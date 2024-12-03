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


def _throughput_benchmark(
    output_folder: Path,
    prefill_length: int = 0,
    generation_length: int = 100,
    use_torch_compile_model: bool = True,
    weight_dtype: str = "bfloat16",
):
    cfg_yaml = f"""
vary_type: grid
vary_params:
  batch_size: [1, 4, 8, 16, 32, 64, 128, 256]
fixed_params:
  prefill_length: {prefill_length}
  generation_length: {generation_length}

  rep: 1
  warmup: 1 #1

x_axis_param: "batch_size"

kernel_specs:
  - model_name: "mlstm_simple"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}
    additional_params:
      use_torch_compile_generate: False
      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused #!!! with triton_fused I get compiler errors triton_fused #native #triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "xlstm"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}
    additional_params:
      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "llama3"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}

  - model_name: "falcon_mamba"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}

  - model_name: "mlstm_simple"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: False #{use_torch_compile_model}
    additional_params:
      use_torch_compile_generate: False
      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused #!!! with triton_fused I get compiler errors triton_fused #native #triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "xlstm"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: False #{use_torch_compile_model}
    additional_params:
      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "llama3"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: False #{use_torch_compile_model}

  - model_name: "falcon_mamba"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: False #{use_torch_compile_model}


benchmark_name: "hf_7B_throughput__pfl{prefill_length}_gl{generation_length}_tc{use_torch_compile_model}_weightdtype{weight_dtype}"
"""
    cfg = from_dict(
        data_class=BenchmarkConfig,
        data=OmegaConf.to_container(OmegaConf.create(cfg_yaml)),
    )
    run_and_record_benchmarks(
        cfg,
        create_hf_model_benchmark,
        output_folder,
        benchmark_type="model",
        setup_model_on_every_param_combination=False,
    )


def _generation_time_benchmark(
    output_folder: Path,
    prefill_length: int = 0,
    batch_size: int = 1,
    use_torch_compile_model: bool = True,
    weight_dtype: str = "bfloat16",
):
    cfg_yaml = f"""
vary_type: grid
vary_params:
  generation_length: [64, 128, 512, 1024, 2048, 4096, 8192, 16384]
fixed_params:
  batch_size: {batch_size}
  prefill_length: {prefill_length}

  rep: 1
  warmup: 1
  benchmark_fn_context_manager: "inference_mode"

x_axis_param: "generation_length"

kernel_specs:
  - model_name: "mlstm_simple"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}
    additional_params:
      use_torch_compile_generate: False
      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "xlstm"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}
    additional_params:
      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "llama2"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}

  - model_name: "llama3"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}

  - model_name: "ministral8b"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}

  - model_name: "codestral_mamba"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}

  - model_name: "falcon_mamba"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}

  - model_name: "zamba2"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}

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


benchmark_name: "hf_7B_generation_time__pfl{prefill_length}_bs{batch_size}_tc{use_torch_compile_model}_weightdtype{weight_dtype}"
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


def _time_to_first_token_benchmark(
    output_folder: Path,
    batch_size: int = 8,
    generation_length: int = 1,
    use_torch_compile_model: bool = True,
    weight_dtype: str = "bfloat16",
):
    cfg_yaml = f"""
vary_type: grid
vary_params:
  prefill_length: [128, 512, 1024, 2048] # , 1024, 2048, 4096, 8192, 16384]
fixed_params:
  batch_size: {batch_size}
  generation_length: {generation_length}

  rep: 5
  warmup: 2 #1
  benchmark_fn_context_manager: "inference_mode"

x_axis_param: "prefill_length"

kernel_specs:
  # - model_name: "mlstm_simple"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: {use_torch_compile_model}
  #   additional_params:
  #     use_torch_compile_generate: False
  #     use_cuda_graphs_generate: True
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

  - model_name: "xlstm"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}
    additional_params:
      use_cuda_graphs_generate: True
      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "llama2"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}
    additional_params:
      use_cuda_graphs_generate: True

  - model_name: "llama3"
    weight_dtype: {weight_dtype}
    use_torch_compile_model: {use_torch_compile_model}
    additional_params:
      use_cuda_graphs_generate: True

  # - model_name: "ministral8b"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: {use_torch_compile_model}

  # - model_name: "codestral_mamba"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: {use_torch_compile_model}

  # - model_name: "falcon_mamba"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: {use_torch_compile_model}

  # - model_name: "zamba2"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: {use_torch_compile_model}

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

  # - model_name: "llama2"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False #{use_torch_compile_model}

  # - model_name: "llama3"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False #{use_torch_compile_model}

  # - model_name: "ministral8b"
  #   weight_dtype: {weight_dtype}
  #   use_torch_compile_model: False #{use_torch_compile_model}

  # - model_name: "codestral_mamba"
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


def _time_to_first_token_benchmark_mlstm_simple(
    output_folder: Path,
    batch_size: int = 8,
    generation_length: int = 1,
    weight_dtype: str = "bfloat16",
):
    cfg_yaml = f"""
vary_type: grid
vary_params:
  prefill_length: [128, 512, 1024, 2048, 4096, 8192, 16384]
fixed_params:
  batch_size: {batch_size}
  generation_length: {generation_length}

  rep: 10
  warmup: 1 #1
  benchmark_fn_context_manager: "inference_mode"

x_axis_param: "prefill_length"

kernel_specs:
  - model_name: "mlstm_simple"
    weight_dtype: {weight_dtype}
    additional_params:
      use_torch_compile_model: False
      use_torch_compile_generate: True
      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "mlstm_simple"
    weight_dtype: {weight_dtype}
    additional_params:
      use_torch_compile_model: True
      use_torch_compile_generate: False
      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16

  - model_name: "mlstm_simple"
    weight_dtype: {weight_dtype}
    additional_params:
      use_torch_compile_model: False
      use_torch_compile_generate: False
      inference_state_dtype: bfloat16
      embedding_dim: 4096
      num_heads: 8
      num_blocks: 32 #3 #32
      vocab_size: 50304

      chunkwise_kernel: chunkwise--triton_xl_chunk
      sequence_kernel: native_sequence__triton_step_fused
      step_kernel: triton_fused

      chunk_size: 128
      autocast_kernel_dtype: bfloat16



benchmark_name: "mlstm_simple_7B_timtofirsttok__bs{batch_size}_gl{generation_length}_weightdtype{weight_dtype}"
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


BenchmarkType = Literal["ttft", "gen_time", "throughput", "ttft_mlstm_simple"]


def run_multiple_benchmarks(
    output_dir: str = "./outputs_kernel_benchmarks",
    benchmark_type: BenchmarkType = "ttft",
    output_folder_suffix: str | None = None,
):
    full_folder_suffix = (
        f"{benchmark_type}__{output_folder_suffix}"
        if output_folder_suffix
        else benchmark_type
    )
    output_folder = setup_output_folder(output_dir, name_suffix=full_folder_suffix)

    if benchmark_type == "throughput":
        _throughput_benchmark(
            output_folder,
            prefill_length=0,
            generation_length=100,
            use_torch_compile_model=True,
            weight_dtype="bfloat16",
        )
    elif benchmark_type == "ttft":
        batch_sizes = [1]  # [1, 4, 8]
        generation_lengths = [10]  # [1, 10, 100]
        for batch_size in batch_sizes:
            for generation_length in generation_lengths:
                _time_to_first_token_benchmark(
                    output_folder,
                    batch_size=batch_size,
                    generation_length=generation_length,
                    use_torch_compile_model=True,
                    weight_dtype="bfloat16",
                )
    elif benchmark_type == "gen_time":
        batch_sizes = [1, 8]
        prefill_lengths = [256, 0]
        for batch_size in batch_sizes:
            for prefill_length in prefill_lengths:
                _generation_time_benchmark(
                    output_folder,
                    prefill_length=prefill_length,
                    batch_size=batch_size,
                    use_torch_compile_model=True,
                    weight_dtype="bfloat16",
                )
    elif benchmark_type == "ttft_mlstm_simple":
        batch_sizes = [1, 4, 8]
        generation_lengths = [100, 1, 10]
        for batch_size in batch_sizes:
            for generation_length in generation_lengths:
                _time_to_first_token_benchmark_mlstm_simple(
                    output_folder,
                    batch_size=batch_size,
                    generation_length=generation_length,
                    weight_dtype="bfloat16",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help=f"The benchmark type. One of {typing.get_args(BenchmarkType)}",
    )
    parser.add_argument(
        "--folder_suffix",
        type=str,
        required=False,
        help="Suffix that is appended to the output folder of the benchmark results.",
    )

    args = parser.parse_args()
    print(args)

    run_multiple_benchmarks(
        output_folder_suffix=args.folder_suffix, benchmark_type=args.benchmark
    )

# Run commands:
# PYTHONPATH=. python scripts/run_hf_model_benchmark.py --benchmark ttft --folder_suffix timetofirsttoken_final_v2
