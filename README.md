# GPU Kernels

Learning how to write fast GPU programs.

## Programming Massively Parallel Programs 

| Chapter | Kernel |
|---------|--------|
| Ch. 2   | [vector_addition.cu](/pmpp/vector_addition/vector_addition.cu) |
| Ch. 3   | [greyscale.cu](/pmpp/greyscale/greyscale.cu) |
| Ch. 3   | [blur.cu](/pmpp/blur/blur.cu) |

## GPU-Puzzles in CUDA C

| Puzzle  | Kernel |
|---------|--------|
| 1 - Map | [map.cu](/gpu-puzzles/map.cu) |
| 2 - Zip | [zip.cu](/gpu-puzzles/zip.cu) |
| 3 - Guard | [guard.cu](/gpu-puzzles/guard.cu) |
| 4 - Map 2D | [map2D.cu](/gpu-puzzles/map2D.cu) |
| 5 - Broadcast | [broadcast.cu](/gpu-puzzles/broadcast.cu) |
| 6 - Blocks | [blocks.cu](/gpu-puzzles/blocks.cu) |

## Metal Kernels

| Kernel |
|--------|
| [greyscale_metal.py](/metal/greyscale_metal.py) |
| [blur_metal.py](/metal/blur_metal.py) |
| [matmul_metal.py](/metal/matmul_metal.py) |

## Notes

[Metal vs. CUDA reference](notes/metal-vs-cuda.md)

## References

- [Nvidia Tesla V100 GPU Architecture](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)

## Papers/Blogs

- [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/pdf/1804.06826)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) 