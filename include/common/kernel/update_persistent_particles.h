#pragma once

#include <device_launch_parameters.h>

namespace dom
{

struct GridCell;
struct MeasurementCell;
struct Particle;

__global__ void updatePersistentParticlesKernel1(const ParticlesSoA particle_array,
                                                 const GridCell* __restrict__ grid_cell_array,
                                                 int particle_count);

__global__ void updatePersistentParticlesKernel2(GridCell* __restrict__ grid_cell_array,
                                                 const float* __restrict__ weight_array_accum, int cell_count);

__global__ void updatePersistentParticlesKernel3(const ParticlesSoA particle_array,
                                                 const GridCell* __restrict__ grid_cell_array,
                                                 int particle_count);

} /* namespace dom */
