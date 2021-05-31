#pragma once

#include <device_launch_parameters.h>

namespace dom
{

struct GridCell;
struct MeasurementCell;
struct Particle;

__global__ void gridCellPredictionUpdateKernel(GridCell* __restrict__ grid_cell_array, ParticlesSoA particle_array,
                                               const MeasurementCell* __restrict__ meas_cell_array,
                                               float* __restrict__ born_masses_array, float p_B, int cell_count);

} /* namespace dom */
