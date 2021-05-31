#pragma once

#include <curand_kernel.h>
#include <device_launch_parameters.h>

namespace dom
{
struct GridCell;
struct ParticlesSoA;

__global__ void moveParticlesKernel(ParticlesSoA particle_array, int x_move, int y_move, int z_move, int particle_count);

__global__ void moveMapKernel(GridCell* __restrict__ grid_cell_array, const GridCell* __restrict__ old_grid_cell_array,
                              int x_move, int y_move, int z_move, int grid_size, int grid_size_z, int cell_count);

} /* namespace dom */
