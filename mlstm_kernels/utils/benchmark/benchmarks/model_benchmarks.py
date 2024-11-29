import logging
from dataclasses import dataclass
from typing import Any

import torch

from ..param_handling import ModelSpec
from .interface import BenchmarkInterface

LOGGER = logging.getLogger(__name__)


@dataclass
class mLSTMSimpleModelBenchmark(BenchmarkInterface):
    model_name: str = "mLSTM"

    embedding_dim: int = 256
    num_heads: int = 4
    num_blocks: int = 2
    vocab_size: int = 1000
    # feedforward
    ffn_proj_factor: float = 2.6667
    ffn_round_up_to_multiple_of: int = 64
    # mlstm
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    # mlstm backend
    chunkwise_kernel: str = "chunkwise--triton_xl_chunk"
    sequence_kernel: str = "native_sequence__native"
    step_kernel: str = "native"
    mode: str = "inference"
    chunk_size: int = 128

    inference_state_dtype: str = "float32"
    autocast_kernel_dtype: str = "bfloat16"

    # benchmark
    amp_enabled: bool = True
    amp_dtype: str = "bfloat16"
    weight_dtype: str = "float32"

    use_torch_compile_model: bool = False
    use_torch_compile_generate: bool = False

    batch_size: int = 1
    prefill_length: int = 128
    generation_length: int = 1

    def setup_benchmark(self) -> None:
        from mlstm_simple.generate import generate_tokens
        from mlstm_simple.model import mLSTM, mLSTMConfig

        mlstm_config = mLSTMConfig(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            vocab_size=self.vocab_size,
            ffn_proj_factor=self.ffn_proj_factor,
            ffn_round_up_to_multiple_of=self.ffn_round_up_to_multiple_of,
            qk_dim_factor=self.qk_dim_factor,
            v_dim_factor=self.v_dim_factor,
            chunkwise_kernel=self.chunkwise_kernel,
            sequence_kernel=self.sequence_kernel,
            step_kernel=self.step_kernel,
            mode=self.mode,
            chunk_size=self.chunk_size,
            inference_state_dtype=self.inference_state_dtype,
            autocast_kernel_dtype=self.autocast_kernel_dtype,
            return_last_states=True,
        )

        self.model = mLSTM(mlstm_config).to(
            dtype=getattr(torch, self.weight_dtype), device=torch.device(self.device)
        )

        LOGGER.info(f"Setting up model: {self.model}")

        if self.use_torch_compile_model:
            self.model = torch.compile(self.model)

        # setup prefill inputs
        if self.prefill_length > 0:
            self.prefill_tokens = torch.randint(
                low=0, high=self.vocab_size, size=(self.batch_size, self.prefill_length)
            ).to(device=torch.device(self.device))
        else:
            self.prefill_tokens = None

        # setup generation function
        def llm_forward(tokens, state):
            return self.model(
                x=tokens,
                state=state,
            )

        self.generate_fn = generate_tokens
        if self.use_torch_compile_generate:
            self.generate_fn = torch.compile(self.generate_fn)

        def benchmark_fn():
            with torch.autocast(
                device_type=torch.device(self.device).type, enabled=self.amp_enabled
            ):
                generated_tokens, state = self.generate_fn(
                    llm_forward=llm_forward,
                    prefill_tokens=self.prefill_tokens,
                    max_length=self.generation_length,
                    batch_size_no_prefill=self.batch_size,
                    device=self.device,
                )

        self.benchmark_fn = benchmark_fn


def create_mlstm_model_benchmark(
    model_spec: ModelSpec, param_dict: dict[str, Any]
) -> BenchmarkInterface:
    mlstm_model_benchmark = mLSTMSimpleModelBenchmark()

    mlstm_model_benchmark.model_name = model_spec.model_name
    mlstm_model_benchmark.amp_enabled = model_spec.amp_enabled
    mlstm_model_benchmark.amp_dtype = model_spec.amp_dtype
    mlstm_model_benchmark.weight_dtype = model_spec.weight_dtype
    mlstm_model_benchmark.use_torch_compile_model = model_spec.use_torch_compile_model

    mlstm_model_benchmark.set_params(param_dict)

    return mlstm_model_benchmark
