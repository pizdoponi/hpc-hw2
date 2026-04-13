#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "lenia.h"

#define N 256
#define NUM_STEPS 100
#define DT 0.1
#define KERNEL_SIZE 26
#define NUM_ORBIUMS 2

// Place two orbiums in the world with different angles. (y, x, angle)
// Orbiums size is 20x20, supproted angles are 0, 90, 180 and 270 degrees.
struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, N / 3, 0}, {N / 3, 0, 180}};

// Write the final world state to a PGM file
static void write_pgm(const char *path, const double *world, unsigned int rows, unsigned int cols)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL)
    {
        fprintf(stderr, "Failed to open output file '%s'.\n", path);
        return;
    }

    fprintf(f, "P5\n%u %u\n255\n", cols, rows);
    for (unsigned int i = 0; i < rows; i++)
    {
        for (unsigned int j = 0; j < cols; j++)
        {
            double value = world[i * cols + j];
            if (value < 0.0)
            {
                value = 0.0;
            }
            if (value > 1.0)
            {
                value = 1.0;
            }
            unsigned char pixel = (unsigned char)(value * 255.0);
            fwrite(&pixel, sizeof(unsigned char), 1, f);
        }
    }
    fclose(f);
}

int main(int argc, char **argv)
{
    Device device = CPU;
    MemoryMode memory_mode = MEMORY_SHARED;
    unsigned int n = N;
    unsigned int steps = NUM_STEPS;
    const char *output_path = NULL;
    unsigned int block_x = 16;
    unsigned int block_y = 16;

    if (argc > 1)
    {
        if (strcmp(argv[1], "gpu") == 0 || strcmp(argv[1], "GPU") == 0)
        {
            device = GPU;
        }
        else if (strcmp(argv[1], "cpu") == 0 || strcmp(argv[1], "CPU") == 0)
        {
            device = CPU;
        }
        else
        {
            fprintf(stderr, "Unknown device '%s'. Use 'cpu' or 'gpu'.\n", argv[1]);
            return 1;
        }
    }

    if (argc > 2)
    {
        n = (unsigned int)strtoul(argv[2], NULL, 10);
    }

    if (argc > 3)
    {
        steps = (unsigned int)strtoul(argv[3], NULL, 10);
    }

    if (argc > 4)
    {
        output_path = argv[4];
        if (output_path[0] == '\0')
        {
            output_path = NULL;
        }
    }

    if (argc > 5)
    {
        block_x = (unsigned int)strtoul(argv[5], NULL, 10);
    }

    if (argc > 6)
    {
        block_y = (unsigned int)strtoul(argv[6], NULL, 10);
    }

    if (argc > 7)
    {
        if (strcmp(argv[7], "shared") == 0 || strcmp(argv[7], "SHARED") == 0)
        {
            memory_mode = MEMORY_SHARED;
        }
        else if (strcmp(argv[7], "global") == 0 || strcmp(argv[7], "GLOBAL") == 0)
        {
            memory_mode = MEMORY_GLOBAL;
        }
        else
        {
            fprintf(stderr, "Unknown memory mode '%s'. Use 'shared' or 'global'.\n", argv[7]);
            return 1;
        }
    }

    if (block_x == 0 || block_y == 0)
    {
        fprintf(stderr, "Block dimensions must be positive integers.\n");
        return 1;
    }

    if ((unsigned long long)block_x * (unsigned long long)block_y > 1024ULL)
    {
        fprintf(stderr, "Block dimensions must satisfy block_x * block_y <= 1024.\n");
        return 1;
    }

    struct orbium_coo orbiums[NUM_ORBIUMS] = {{0, (int)(n / 3), 0}, {(int)(n / 3), 0, 180}};

    LeniaResult result = evolve_lenia(n, n, steps, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS, device, block_x, block_y, memory_mode);

    double t_total = result.times.t_execution + result.times.t_copy_to_device + result.times.t_copy_to_host;

    printf("DEVICE=%s\n", device == GPU ? "GPU" : "CPU");
    printf("SIZE=%u\n", n);
    printf("STEPS=%u\n", steps);
    printf("BLOCK_X=%u\n", block_x);
    printf("BLOCK_Y=%u\n", block_y);
    printf("MEMORY_MODE=%s\n", memory_mode == MEMORY_SHARED ? "SHARED" : "GLOBAL");
    printf("T_EXEC=%.6f\n", result.times.t_execution);
    printf("T_H2D=%.6f\n", result.times.t_copy_to_device);
    printf("T_D2H=%.6f\n", result.times.t_copy_to_host);
    printf("T_TOTAL=%.6f\n", t_total);

    if (output_path != NULL)
    {
        write_pgm(output_path, result.world, n, n);
    }

    free(result.world);
    return 0;
}
