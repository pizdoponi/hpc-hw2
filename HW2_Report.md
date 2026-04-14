# Assignment 2: Lenia

## 1. Implementation

### 1.1 CUDA Parallelization Strategy

The core Lenia algorithm consists of two operations per timestep:

1. **2D Convolution**: Each cell's state is convolved with a 26×26 ring kernel using toroidal boundary conditions.
2. **Growth Update**: Cell values are updated based on a Gaussian growth function applied to the convolution result.

We parallelize both operations using thread-per-thread-output mapping:

- Grid dimensions are computed dynamically based on world size: `gridSize = ((cols-1)/16 + 1, (rows-1)/16 + 1)`.
- Block dimensions are passed as runtime (CLI) arguments to benchmark different block shapes.
- Each thread independently computes one output cell, performing full kernel operations for convolution.

### 1.2 Memory Management and Transfer Optimization

We minimize host-device bandwidth by:

- **Initialising state on host**: The kernel and initial world are initialised on host because it is more convenient.
- **Single first transfer**: The kernel and initial world state are transferred to device once at the start.
- **On device computation**: All 100 timesteps execute entirely on GPU with no intermediate transfers.
- **Single final transfer**: Only the final world state is copied back to host.

This approach reduces redundant transfers between host and device compared to per step transfers. Furthermore, there is no real reason to transfer any data during the computation back to host. For this reason, and since the requirements were to implement it as efficiently as possible, we did not separately benchmark the speedup gain no transfers vs. transfer per time step.

The measured timings include all three components:

$$T_{\text{total}} = T_{\text{t_host_to_device}} + T_{\text{execution}} + T_{\text{t_device_to_host}}$$

### 1.3 Thread Block Optimization

To test how block sizes affect performance, we measured execution times for different configurations. Measured average times (s) were:

| Grid Size | 8×32 | 16×16 | 32×8 | 32×16 | Best |
|-----------|------|-------|------|-------|------|
| 512×512   | 0.051213 | 0.050904 | 0.050590 | 0.048100 | 32×16 |
| 1024×1024 | 0.152106 | 0.158307 | 0.149676 | 0.150233 | 32×8 |
| 2048×2048 | 0.604062 | 0.594140 | 0.594851 | 0.595997 | 16×16 |
| 4096×4096 | 2.423494 | 2.375615 | 2.377747 | 2.378126 | 16×16 |

Thus, 16×16 was the best-performing configuration for larger grids (2048 and 4096), while 32×16 / 32×8 were faster on smaller grids (512 and 1024).


For the shared-vs-global memory comparison we used the default 16×16 configuration.

### 1.4 Shared Memory Implementation

We implemented two GPU convolution modes:
- **GLOBAL**: baseline convolution reads directly from global memory.
- **SHARED**: each thread block first loads an extended tile (including halo) into dynamic shared memory, then computes convolution from shared memory.

Both modes use identical update logic and differ only in convolution data access strategy. This allows direct speedup comparison for memory optimization.

## 2. Experimental Results

### 2.1 Benchmark Setup

- **Hardware**: NVIDIA GPU on Arnes cluster
- **Grid sizes**: 256×256, 512×512, 1024×1024, 2048×2048, 4096×4096
- **Simulation steps**: 100
- **Repetitions**: 5 runs for all sizes except 4096×4096 (1 run due to sequential baseline exceeding 1 hour)
- **Measurement method**: Wall-clock time via `omp_get_wtime()`, averaged across multiple runs

### 2.2 Performance Metrics

| Grid Size | CPU Time (s) | GPU Time (s) | Speedup |
|-----------|--------------|--------------|---------|
| 256×256   | 20.32        | 0.017        | 1196×   |
| 512×512   | 81.31        | 0.052        | 1565×   |
| 1024×1024 | 325.08       | 0.154        | 2105×   |
| 2048×2048 | 1300.87      | 0.605        | 2152×   |
| 4096×4096 | 5210.30      | 2.417        | 2156×   |

**Table 1: Average execution times and speedups on Arnes cluster.**

### 2.3 Speedup Analysis

Speedup increases with grid size:

1. **256×256**: 1196× speedup
2. **512×512**: 1565× speedup
3. **1024×1024**: 2105× speedup
4. **2048×2048**: 2152× speedup
5. **4096×4096**: 2156× speedup

Speedup grows from small to large grids, with diminishing returns visible between 2048 and 4096 (from 2152× to 2156×). The slowing improvement at larger grid sizes reflects reduced computation-to-overhead ratio and increased memory bandwidth pressure relative to arithmetic operations.

### 2.4 Shared vs Global Memory (16×16)

Using the same GPU setup (100 steps, same grid sizes), we benchmarked convolution with global-memory reads versus shared-memory tiling.

| Grid Size | Global Avg (s) | Shared Avg (s) | Speedup (Global/Shared) |
|-----------|----------------|----------------|-------------------------|
| 256×256   | 0.016992       | 0.009588       | 1.772×                  |
| 512×512   | 0.050779       | 0.027361       | 1.856×                  |
| 1024×1024 | 0.154643       | 0.087660       | 1.764×                  |
| 2048×2048 | 0.594342       | 0.354649       | 1.676×                  |
| 4096×4096 | 2.368430       | 1.413549       | 1.676×                  |

**Table 2: Shared-memory speedup over global-memory baseline (16×16 block).**

The shared-memory version is consistently faster for all tested sizes, with measured speedup between 1.676× and 1.856×.

## 3. Conclusions

1. **Effective parallelization**: CUDA implementation achieves large CPU-to-GPU speedups, reaching 2156× at 4096×4096.
2. **Block-shape benchmarking**: Runtime-configurable block sizes enabled empirical verification of launch-shape performance across grid sizes.
3. **Shared-memory optimization**: Shared-memory convolution delivers consistent measured gains over global-memory convolution (1.676× to 1.856×).
4. **Memory efficiency**: Single-transfer strategy minimizes host-device data movement during simulation timesteps.
5. **Optimization opportunities**: Further improvements could leverage:
   - Asynchronous data transfers overlapped with computation
   - Double buffering for multi-GPU scenarios
   - Additional kernel-level optimizations (e.g., further memory access tuning)