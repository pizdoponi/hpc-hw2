#include <stdio.h>
#include <stdlib.h>
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

int main()
{
    // Run the simulation
    LeniaResult result = evolve_lenia(N, N, NUM_STEPS, DT, KERNEL_SIZE, orbiums, NUM_ORBIUMS, CPU);
    printf("Execution time: %.3f\n", result.times.t_execution);
    printf("Time device <-- host: %.3f, Time device --> host: %.3f\n", result.times.t_copy_to_device, result.times.t_copy_to_host);
    free(result.world);
    return 0;
}
