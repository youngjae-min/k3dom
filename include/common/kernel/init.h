#pragma once

#include <curand_kernel.h>
#include <device_launch_parameters.h>

namespace dom
{

struct GridCell;
struct MeasurementCell;
struct Particle;

__global__ void setupRandomStatesKernel(curandState* __restrict__ states, unsigned long long seed, int count);

__global__ void initGridCellsKernel(GridCell* __restrict__ grid_cell_array,
                                    int grid_size, int cell_count, float prior_f, float prior_s, float prior_d, float prior_o);

__global__ void reinitGridParticleIndices(GridCell* __restrict__ grid_cell_array, int cell_count);

} /* namespace dom */
