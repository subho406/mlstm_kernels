# mLSTM Kernels

In this repository we collect clean implementations of the different mLSTM formulations.

## Kernel variants

The mLSTM repo contains the following kernel variants:
- `chunkwise`: chunkwise kernels like flash linear attention (sub-quadratic)
- `parallel`: parallel kernels like flash attention (quadratic)
- `recurrent`: recurrent kernels (mostly for inference) (linear)


# Working Notes:

## Next steps

- [ ] integrate memory tracker for kernels for measuring GPU memory during speed tests



## Questions about Nsight Systems & Nsight Compute

- How can I organize the workflow efficiently in project. 
- How can I compare to baselines efficiently.
