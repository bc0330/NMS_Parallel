# NMS Parallelization Project

This project implements Non-Maximum Suppression (NMS) using various parallelization strategies to improve performance.

## Strategies

### 1. Sequential (`nms_seq.cpp`)
- **Description**: The baseline implementation. It processes images one by one. For each image, it sorts bounding boxes by score and iteratively suppresses overlapping boxes using a standard nested loop.
- **Performance**: Baseline. Time complexity is roughly O(N^2) per image where N is the number of boxes.

### 2. SIMD - AVX2 (`nms_simd.cpp`)
- **Description**: Optimizes the inner loop of the NMS algorithm using AVX2 intrinsics.
    - Converts the array of structures (AoS) to a structure of arrays (SoA) for better cache locality and vectorization.
    - Computes IoU for 8 boxes simultaneously using 256-bit registers (`__m256`).
- **Expected Performance**: Significant speedup (approx. 4-8x) for the IoU computation part, especially for images with many bounding boxes.

### 3. OpenMP (`nms_omp.cpp`)
- **Description**: Parallelizes the processing of *multiple images* using OpenMP.
    - Uses `#pragma omp parallel for schedule(dynamic)` to distribute the list of image files across available CPU threads.
    - Each thread processes a different image independently using the sequential NMS logic.
- **Expected Performance**: Near-linear speedup with the number of physical cores, assuming enough images to keep all threads busy. Ideal for batch processing.

### 4. CUDA Implementations
These implementations offload the NMS computation to the GPU (NVIDIA GTX 1060).

- **Naive (`nms_cuda_naive.cu`)**:
    - Sorts boxes on CPU.
    - Copies boxes to GPU.
    - Kernel: Each thread computes IoU for one pair of boxes and writes to a collision mask.
    - CPU: Reads mask and performs suppression.
    - *Note*: High overhead for small number of boxes per image.

- **Optimized (`nms_cuda_opt.cu`)**:
    - **Memory Reuse**: Reuses GPU memory buffers across images to avoid expensive `cudaMalloc`/`cudaFree` calls.
    - **Shared Memory Tiling**: Threads load a block of boxes into shared memory to reduce global memory bandwidth usage during IoU comparisons.

- **Ultimate (`nms_cuda_ultimate.cu`)**:
    - **Hybrid Strategy**: Automatically falls back to **CPU SIMD** execution for images with few boxes (< 1000) where GPU overhead outweighs the benefits.
    - **Thrust Sorting**: Uses the Thrust library to sort boxes directly on the GPU.
    - **Bitmask Optimization**: Uses `unsigned long long` bitmasks for compact storage and efficient checking.
    - **Memory Reuse**: Reuses GPU memory buffers.

**Performance Note**: For this dataset (COCO validation), the images often have a small number of boxes. Consequently, the overhead of data transfer and kernel launching makes the GPU implementations slower than the multi-core OpenMP CPU implementation. The "Ultimate" version mitigates this by using the CPU for small workloads.

### 5. MPI (`nms_mpi.cpp`) (Planned/Remote)
- **Description**: Distributed memory parallelization.
    - Distributes the workload (images) across multiple nodes in a cluster.
    - Uses MPI to coordinate and gather results.
- **Expected Performance**: Scales beyond a single machine, allowing processing of massive datasets by utilizing multiple nodes.

## Remote Execution

You can run these implementations on the cluster using the provided Makefile targets:

- `make run_remote_seq`: Run Sequential version
- `make run_remote_simd`: Run SIMD version
- `make run_remote_omp`: Run OpenMP version
- `make run_remote_cuda_naive`: Run CUDA Naive version
- `make run_remote_cuda_opt`: Run CUDA Optimized version
- `make run_remote_cuda_ultimate`: Run CUDA Ultimate version

**Note**: TBB implementations are currently disabled due to library unavailability on the cluster.
