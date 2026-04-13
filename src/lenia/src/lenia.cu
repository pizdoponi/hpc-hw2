#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "lenia.h"
#include "orbium.h"
#include "gifenc.h"

// Include CUDA headers
#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

// Uncomment to generate gif animation
//#define GENERATE_GIF

// For prettier indexing syntax
#define w(r, c) (w[(r) * w_cols + (c)])
#define input(r, c) (input[((r) % rows) * cols + ((c) % cols)])

__host__ __device__ inline int wrap_index(int index, int size)
{
    int wrapped = index % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

// Function to calculate Gaussian
__host__ __device__ inline double gauss(double x, double mu, double sigma)
{
    return exp(-0.5 * pow((x - mu) / sigma, 2));
}

// Function for growth criteria
double growth_lenia(double u)
{
    double mu = 0.15;
    double sigma = 0.015;
    return -1 + 2 * gauss(u, mu, sigma); // Baseline -1, peak +1
}

// Function to generate convolution kernel
double *generate_kernel(double *K, const unsigned int size)
{
    // Construct ring convolution filter
    double mu = 0.5;
    double sigma = 0.15;
    int r = size / 2;
    double sum = 0;
    if (K != NULL)
    {
        for (int y = -r; y < r; y++)
        {
            for (int x = -r; x < r; x++)
            {
                double distance = sqrt((1 + x) * (1 + x) + (1 + y) * (1 + y)) / r;
                K[(y + r) * size + x + r] = gauss(distance, mu, sigma);
                if (distance > 1)
                {
                    K[(y + r) * size + x + r] = 0; // Cut at d=1
                }
                sum += K[(y + r) * size + x + r];
            }
        }
        // Normalize
        for (unsigned int y = 0; y < size; y++)
        {
            for (unsigned int x = 0; x < size; x++)
            {
                K[y * size + x] /= sum;
            }
        }
    }
    return K;
}

// Function to perform convolution on input using kernel w
// Note that the kernel is flipped for convolution as per definition, and we use modular indexing for toroidal world
inline double *convolve2d(double *result, const double *input, const double *w, const unsigned int rows, const unsigned int cols, const unsigned int w_rows, const unsigned int w_cols)
{
    if (result != NULL && input != NULL && w != NULL)
    {
        for (unsigned int i = 0; i < rows; i++)
        {
            for (unsigned int j = 0; j < cols; j++)
            {
                double sum = 0;
                for (int ki = w_rows - 1, kri = 0; ki >= 0; ki--, kri++)
                {
                    for (int kj = w_cols - 1, kcj = 0; kj >= 0; kj--, kcj++)
                    {
                        sum += w(ki, kj) * input((i - w_rows / 2 + rows + kri), (j - w_cols / 2 + cols + kcj));
                    }
                }
                result[i * cols + j] = sum;
            }
        }
    }
    return result;
}

// Function to perform convolution on input using kernel w on cuda
// Note that the kernel is flipped for convolution as per definition, and we use modular indexing for toroidal world
__global__ void convolve2d_cuda_global(double *result, const double *input, const double *w, const unsigned int rows, const unsigned int cols, const unsigned int w_rows, const unsigned int w_cols)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        double sum = 0;
        for (int ki = w_rows - 1, kri = 0; ki >= 0; ki--, kri++)
        {
            for (int kj = w_cols - 1, kcj = 0; kj >= 0; kj--, kcj++)
            {
                int r = (y - w_rows / 2 + rows + kri) % rows;
                int c = (x - w_cols / 2 + cols + kcj) % cols;
                sum += w[ki * w_cols + kj] * input[r * cols + c];
            }
        }
        result[y * cols + x] = sum;
    }
}

__global__ void convolve2d_cuda_shared(double *result, const double *input, const double *w, const unsigned int rows, const unsigned int cols, const unsigned int w_rows, const unsigned int w_cols)
{
    extern __shared__ double shared_block_memory[];

    const unsigned int local_x = threadIdx.x; // inside the block.
    const unsigned int local_y = threadIdx.y;
    const unsigned int global_x = blockIdx.x * blockDim.x + local_x; // in the world / grid.
    const unsigned int global_y = blockIdx.y * blockDim.y + local_y;

    // The current convolution uses offsets [-w_rows/2, ..., +w_rows/2-1].
    // For a 26x26 kernel, that means 13 cells of halo on the "negative" side (left/top),
    // and 12 cells on the "positive" side (right/bottom).
    const unsigned int left_halo = w_cols / 2;
    const unsigned int top_halo = w_rows / 2;
    const unsigned int right_halo = w_cols - left_halo - 1;
    const unsigned int bottom_halo = w_rows - top_halo - 1;

    const unsigned int shared_memory_width = blockDim.x + left_halo + right_halo;
    const unsigned int shared_memory_height = blockDim.y + top_halo + bottom_halo;
    const unsigned int shared_memory_elements = shared_memory_width * shared_memory_height;

    // The top left element of the shared tile corresponds to the block's top left
    // output cell shifted by the halo size.
    // This is where we need to start loading from global memory.
    const int shared_memory_global_start_x = (int)(blockIdx.x * blockDim.x) - (int)left_halo;
    const int shared_memory_global_start_y = (int)(blockIdx.y * blockDim.y) - (int)top_halo;

    // Each thread loads several elements, striding by the total number of threads in the block.
    const unsigned int thread_linear_index = local_y * blockDim.x + local_x;
    const unsigned int threads_per_block = blockDim.x * blockDim.y;

    for (unsigned int shared_index = thread_linear_index; shared_index < shared_memory_elements; shared_index += threads_per_block)
    {
        // Recover the 2D coordinates from the linear index.
        const unsigned int shared_y = shared_index / shared_memory_width;
        const unsigned int shared_x = shared_index % shared_memory_width;

        const int wrapped_global_y = wrap_index(shared_memory_global_start_y + (int)shared_y, (int)rows);
        const int wrapped_global_x = wrap_index(shared_memory_global_start_x + (int)shared_x, (int)cols);

        shared_block_memory[shared_index] = input[wrapped_global_y * cols + wrapped_global_x];
    }

    // All threads must wait until the shared tile is fully populated before any thread starts reading from it for the convolution.
    __syncthreads();

    if (global_x >= cols || global_y >= rows)
    {
        return;
    }

    double sum = 0.0;

    // The thread's local output position inside the block maps to the top-left
    // corner of its stencil window inside the shared tile.
    for (unsigned int stencil_row = 0; stencil_row < w_rows; stencil_row++)
    {
        const unsigned int kernel_row = w_rows - 1 - stencil_row;
        const unsigned int shared_row = local_y + stencil_row;

        for (unsigned int stencil_col = 0; stencil_col < w_cols; stencil_col++)
        {
            const unsigned int kernel_col = w_cols - 1 - stencil_col;
            const unsigned int shared_col = local_x + stencil_col;

            const double kernel_value = w[kernel_row * w_cols + kernel_col];
            const double world_value = shared_block_memory[shared_row * shared_memory_width + shared_col];
            sum += kernel_value * world_value;
        }
    }

    result[global_y * cols + global_x] = sum;
}

__global__ void growth_lenia_cuda(double* d_world, double* d_tmp_world, unsigned int rows, unsigned int cols, double dt)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        double mu = 0.15;
        double sigma = 0.015;
        double u = d_tmp_world[y * cols + x];
        double growth = -1 + 2 * gauss(u, mu, sigma); // Baseline -1, peak +1

        // d_world[y * cols + x] += dt * growth;
        // d_world[y * cols + x] = fmin(1, fmax(0, d_world[y * cols + x])); // Clip between 0 and 1

        // The above does not work on device because it is for host code. We need to do the clipping manually.
        double val = d_world[y * cols + x] + dt * growth;

        if (val < 0.0) val = 0.0;
        if (val > 1.0) val = 1.0;

        d_world[y * cols + x] = val;
    }
}

// Function to evolve Lenia
LeniaResult evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums, const Device device, const unsigned int block_x, const unsigned int block_y, const MemoryMode memory_mode)
{

#ifdef GENERATE_GIF
    ge_GIF *gif = ge_new_gif(
        "lenia.gif",     /* file name */
        cols, rows,      /* canvas size */
        inferno_pallete, /*pallete*/
        8,               /* palette depth == log2(# of colors) */
        -1,              /* no transparency */
        0                /* infinite loop */
    );
#endif

    Times times = {0, 0, 0};

    if (device == GPU)
    {
        // Allocate memory on host
        double *h_w = (double *)calloc(kernel_size * kernel_size, sizeof(double));
        double *h_world = (double *)calloc(rows * cols, sizeof(double));

        // Generate convolution kernel
        h_w=generate_kernel(h_w,kernel_size);

        // Place orbiums
        for (unsigned int o = 0; o < num_orbiums; o++)
        {
            h_world = place_orbium(h_world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
        }

        // Allocate memory on device
        double *d_w, *d_world, *d_tmp_world;
        checkCudaErrors(cudaMalloc((void **)&d_w, kernel_size * kernel_size * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&d_world, rows * cols * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **)&d_tmp_world, rows * cols * sizeof(double)));

        // Transfer data: device <-- host
        double start_time = omp_get_wtime();

        checkCudaErrors(cudaMemcpy(d_w, h_w, kernel_size * kernel_size * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_world, h_world, rows * cols * sizeof(double), cudaMemcpyHostToDevice));
        // No need to transfer tmp_world as it will be computed on device and not used for anything else.

        double end_time = omp_get_wtime();
        times.t_copy_to_device = end_time - start_time;

        // Compute on device
        dim3 blockSize(block_x, block_y);
        dim3 gridSize((cols - 1)/blockSize.x + 1, (rows - 1)/blockSize.y + 1); // gridSize is more than enough, no need for striding.

        // Compute shared-memory size per block only for shared-memory convolution.
        const unsigned int shared_memory_width = blockSize.x + kernel_size - 1;
        const unsigned int shared_memory_height = blockSize.y + kernel_size - 1;
        const size_t shared_memory_bytes = (size_t)shared_memory_width * shared_memory_height * sizeof(double);

        // Lenia Simulation
        // Each time step is still sequential, so this for loop is needed.
        start_time = omp_get_wtime();

        for (unsigned int step = 0; step < steps; step++)
        {
            // Convolution
            if (memory_mode == MEMORY_SHARED)
            {
                convolve2d_cuda_shared<<<gridSize, blockSize, shared_memory_bytes>>>(d_tmp_world, d_world, d_w, rows, cols, kernel_size, kernel_size);
            }
            else
            {
                convolve2d_cuda_global<<<gridSize, blockSize>>>(d_tmp_world, d_world, d_w, rows, cols, kernel_size, kernel_size);
            }
            checkCudaErrors(cudaGetLastError());

            // Evolution
            growth_lenia_cuda<<<gridSize, blockSize>>>(d_world, d_tmp_world, rows, cols, dt);
            checkCudaErrors(cudaGetLastError());

            // gif generation
//             for (unsigned int i = 0; i < rows; i++)
//             {
//                 for (unsigned int j = 0; j < cols; j++)
//                 {
// #ifdef GENERATE_GIF
//                     gif->frame[i * rows + j] = world[i * rows + j] * 255;
// #endif
//                 }
//             }
// #ifdef GENERATE_GIF
//             ge_add_frame(gif, 5);
// #endif
        }

    // Kernel launches are asynchronous; synchronize so execution timing includes real GPU compute time.
    checkCudaErrors(cudaDeviceSynchronize());

        end_time = omp_get_wtime();
        times.t_execution = end_time - start_time;
// #ifdef GENERATE_GIF
//         ge_close_gif(gif);
// #endif

        // Transfer data: device --> host
        start_time = omp_get_wtime();
        checkCudaErrors(cudaMemcpy(h_world, d_world, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));
        end_time = omp_get_wtime();
        times.t_copy_to_host = end_time - start_time;

        // free space: device
        checkCudaErrors(cudaFree(d_w));
        checkCudaErrors(cudaFree(d_world));
        checkCudaErrors(cudaFree(d_tmp_world));

        // Free space: host
        free(h_w);

        return LeniaResult{h_world, times};
    }
    else if (device == CPU)
    {
        // Allocate memory
        double *w = (double *)calloc(kernel_size * kernel_size, sizeof(double));
        double *world = (double *)calloc(rows * cols, sizeof(double));
        double *tmp = (double *)calloc(rows * cols, sizeof(double));

        // Generate convolution kernel
        w=generate_kernel(w,kernel_size);

        // Place orbiums
        for (unsigned int o = 0; o < num_orbiums; o++)
        {
            world = place_orbium(world, rows, cols, orbiums[o].row, orbiums[o].col, orbiums[o].angle);
        }

        // Lenia Simulation
        double start_time = omp_get_wtime();

        for (unsigned int step = 0; step < steps; step++)
        {
            // Convolution
            tmp = convolve2d(tmp, world, w, rows, cols, kernel_size, kernel_size);

            // Evolution
            for (unsigned int i = 0; i < rows; i++)
            {
                for (unsigned int j = 0; j < cols; j++)
                {
                    world[i * rows + j] += dt * growth_lenia(tmp[i * rows + j]);
                    world[i * rows + j] = fmin(1, fmax(0, world[i * rows + j])); // Clip between 0 and 1
#ifdef GENERATE_GIF
                    gif->frame[i * rows + j] = world[i * rows + j] * 255;
#endif
                }
            }
#ifdef GENERATE_GIF
            ge_add_frame(gif, 5);
#endif
        }
        double end_time = omp_get_wtime();
        times.t_execution = end_time - start_time;

#ifdef GENERATE_GIF
        ge_close_gif(gif);
#endif
        free(w);
        free(tmp);
        return LeniaResult{world, times};
    }
    else
    {
        fprintf(stderr, "Invalid device specified. Use 'GPU' or 'CPU'.\n");
        return LeniaResult{NULL, times};
    }
}
