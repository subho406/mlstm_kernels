import logging
from dataclasses import dataclass

from .interface import BenchmarkInterface

LOGGER = logging.getLogger(__name__)


@dataclass
class MLSTMSimpleModelBenchmark(BenchmarkInterface):
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

    batch_size: int = 1
    prefill_length: int = 128
    generation_length: int = 1

    def setup_benchmark(self) -> None:
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
        )

        self.model = mLSTM(mlstm_config)

        LOGGER.info(f"Setting up model: {self.model}")

        # setup prefill inputs

        def benchmark_fn():
            pass
            # consume prefill

            # generate
