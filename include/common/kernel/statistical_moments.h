#pragma once

#include <device_launch_parameters.h>

namespace dom
{
struct GridCell;
struct Particle;

__global__ void statisticalMomentsKernel1(const ParticlesSoA particle_array,
                                          float* __restrict__ vel_x_array, float* __restrict__ vel_y_array,
                                          float* __restrict__ vel_z_array,
                                          int particle_count);

__global__ void statisticalMomentsKernel2(GridCell* __restrict__ grid_cell_array,
                                          const float* __restrict__ vel_x_array_accum,
                                          const float* __restrict__ vel_y_array_accum,
                                          const float* __restrict__ vel_z_array_accum,
                                          int cell_count);

} /* namespace dom */
