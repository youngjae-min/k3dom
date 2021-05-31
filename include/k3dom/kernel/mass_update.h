#pragma once

#include <device_launch_parameters.h>

namespace dom
{

struct GridCell;
struct Particle;

__global__ void gridCellPredictionUpdateKernel(GridCell* __restrict__ grid_cell_array, ParticlesSoA particle_array,
                                               float* __restrict__ born_masses_array, float p_B, int cell_count,
                                               const float* __restrict__ meas_array_x,
                                               const float* __restrict__ meas_array_y,
                                               const float* __restrict__ meas_array_z, int meas_len, float dt,
                                               int grid_size, int grid_size_z, float resolution,
                                               float sigma, float ls, float gamma, float mass_scale,
                                               float sensor_x, float sensor_y, float sensor_z);

} /* namespace dom */
