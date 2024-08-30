# mLSTM Kernels

In this repository we collect clean implementations of the different mLSTM formulations.

## External kernel interface with names

```python
def mlstm_interface(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    f: torch.Tensor,
    c_initial: torch.Tensor = None,
    n_initial: torch.Tensor = None,
    m_initial: torch.Tensor = None,
    return_last_states: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
