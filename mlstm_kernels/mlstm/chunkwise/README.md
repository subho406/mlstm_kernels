# mLSTM Chunkwise Kernels

## Training Kernels

### Full Chunkwise Kernel

As implemented in flash-linear-attention and Korbinian.

The kernel consists of two "sub-"kernels. A recurrent part and a parallel part.
The recurrent part computes the C, n, and m states for all chunks.
The parallel part computes the outputs within the chunks.

Forward pass contains:
- `chunkwise__recurrent_fw`
- `chunkwise__parallel_fw`
Backward pass contains:
- `chunkwise__recurrent_bw`
- `chunkwise__parallel_bw`

## Inference Kernels

### Chunkwise Recurrent

This is only the `chunkwise__recurrent_fw` part. 
On a high level this is a "chunk step function". Given an inital state and a sequence of inputs
it computes the next state without computing every intermediate state. 

During inference this might be used to consume long prompts and produce a hidden state from which decoding starts.