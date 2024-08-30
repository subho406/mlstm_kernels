# mLSTM Kernels

In this repository we collect clean implementations of the different mLSTM formulations.

## External kernel interface with names

```python
def mlstm_interface(
    q: torch.Tensor, # (B, NH, S, DHQK)
    k: torch.Tensor, # (B, NH, S, DHQK)
    v: torch.Tensor, # (B, NH, S, DHV)
    i: torch.Tensor, # (B, NH, S)
    f: torch.Tensor, # (B, NH, S)
    c_initial: torch.Tensor = None, # (B, NH, DHQK, DHV)
    n_initial: torch.Tensor = None, # (B, NH, DHQK)
    m_initial: torch.Tensor = None, # (B, NH)
    return_last_states: bool = False,
    eps: float = 1e-6,
    autocast_kernel_dtype: torch.dtype = torch.float16,
    chunk_size: int = 64,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    # (B, NH, S, DHV) | ((B, NH, S, DHV), ((B, NH, DHQK, DHV), (B, NH, DHQK), (B, NH)))
    """
    Returns:
        torch.Tensor: matH outputs (no n and m values, no last states)
        tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: matH, (matC_last, vecN_last, scaM_last)
    """
    
    pass

```

## Kernel variants

The mLSTM repo contains the following kernel variants:
- `chunkwise`: chunkwise kernels like flash linear attention (sub-quadratic)
- `parallel`: parallel kernels like flash attention (quadratic)
- `recurrent`: recurrent kernels (mostly for inference) (linear)

Not all variants support all features of the interface. Only the chunkwise and recurrent support passing the initial states and returning the last states. 

### Kernel naming

#### External names of kernel functions in chunkwise, parallel and recurrent modules:
- Python function: `mlstm_[recurrent|parallel|chunkwise]_[specifier]_[triton|torch]_[[autograd|ownbw]]`
- Registry name (within module): `[specifier]_[triton|torch]_[[autograd|ownbw]]`

---
---
---
# Working Notes:

## Next steps

- [ ] integrate memory tracker for kernels for measuring GPU memory during speed tests



## Questions about Nsight Systems & Nsight Compute

- How can I organize the workflow efficiently in project. 
- How can I compare to baselines efficiently.
