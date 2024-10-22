# Metal vs. CUDA

Basic overview of Metals and CUDA C thread management models and associated variables:

| Concept                | Metal                       | CUDA                 |
|------------------------|-----------------------------|----------------------|
| Thread Organization    | Grid                        | Grid                     |
| Thread Grouping        | Threadgroup                 | Block                    |
| Thread Identifier      | `thread_position_in_grid`   | `threadIdx`              |
| Threadgroup Identifier | `threadgroup_position_in_grid` | `blockIdx`            |
| Threadgroup Size       | `threadgroup_size`          | `blockDim`               |
| Grid Size              | `grid_size`                 | `gridDim`                |
| Shared Memory          | `threadgroup_memory`        | `shared memory`          |
| Synchronization        | `threadgroup_barrier`       | `__syncthreads`          |
| Memory Hierarchy       | Unified Memory              | Global, Shared, Local    |
