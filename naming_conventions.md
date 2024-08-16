# Naming variables in kernels

## Common types of variables
Where they exist: Triton (T), CUDA (C), both (TC)


Pointer (TC): Points to HBM memory

Shared Memory Pointer (C): Points to Shared Memory

Data Block (T): Contains a Block 1D, 2D of values

<!-- Scalar (TC): Contains a number

Stride Global (TC): Stride along one axis

Index Global (TC): Index along one axis, multiplied by stride to be added to pointer

Offset Global (TC): Global Index without Block internal index, one level below, i.e. Index Global = Offset Global + Block-Internal Index

Shared Memory Stride (C): Stride along one axis in shared memory

Block-Internal Index (TC): Index within a block (i.e. also relevant for shared memory)

Block-Internal Offset (C): Warp-Offset within a block (also relevant for shared memory)

Warp-Internal Index (C): Index within a warp -->

### Main naming conventions:

What defines a point in memory: __Pointer__ (ptr) + where it lies (_shd shared, _glb global, _reg register, can be omitted in triton) 

What is added to a pointer: __Offset__ (off)
An offset is tied to a single pointer (unless one does pointwise operations of exactly same-sized objects)

How does one walk in one direction in memory? __Stride__
A stride is tied to a memory pointer + one direction. It defines how an offset is incremented for the next element

How do we define a position in a virtual matrix? __Index__
An index is virtually defined and tells you the position in a Tensor along one dimension / direction.
One dimension / direction can be divided into multiple levels for patches (i.e. HBM / Shared Memory / Warp ...)
There is a __LevelIndex__ and a __LevelSize__, the lowest __LevelSize__ is typically 1. The __FullIndex__ is defined as:

__FullIndex__ = \sum_{levels} __LevelIndex__ * __LevelSize__ 

The FullIndex is partly dependent on where something is stored, because it might only describe a position within a Patch instead of a full matrix.
An offset is now defined as the sum of products of __FullIndices__ times __Strides__ along multiple directions:

__Offset__ = \sum_{directions} __StrideDir__ * __FullIndexDir__

Offsets can be used, but ultimately one can also update pointers right away. Then everything that needs to be handled is one-dimensional, with other dimensions are handled with the pointer-specific strides.

Cutlass Concept: Coordinate - Tuple of __FullIndex__ describing the full position in the (virtual) matrix

Otherwise naming is mostly aligned, offsets there are actually per-dimension.

See: [https://github.com/NVIDIA/cutlass/blob/main/media/docs/terminology.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/terminology.md)


### Orthogonal to these dimensions - what does it point to / contain:
Dimensionality:
- Scalar, Vector, Matrix
- Fragment (Small Matrix Patch)

Function:
- State
- Input
- Output
- Gate

mLSTM specific:
- input: q, k, v  (or in fused version, Wq, Wk, Wv)
- original input: x
- gates: f, i
- output: h
- matrix state: C
- normalizer state: n
- stabilizer state: m
- gating matrix: G (?)
