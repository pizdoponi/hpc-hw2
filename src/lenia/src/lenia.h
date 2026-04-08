#ifndef LENIA_H
#define LENIA_H

#ifdef __cplusplus
extern "C" {
#endif

struct orbium_coo {
    int row;
    int col;
    int angle;
};


typedef enum {
    CPU,
    GPU
} Device;


typedef struct {
    double t_copy_to_device;
    double t_execution;
    double t_copy_to_host;
} Times;


typedef struct {
    double *world;
    Times times;
} LeniaResult;


LeniaResult *evolve_lenia(const unsigned int rows, const unsigned int cols, const unsigned int steps, const double dt, const unsigned int kernel_size, const struct orbium_coo *orbiums, const unsigned int num_orbiums, const Device device);

#ifdef __cplusplus
}
#endif

#endif


