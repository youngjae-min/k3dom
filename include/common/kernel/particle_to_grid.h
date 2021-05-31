#pragma once

#include <device_launch_parameters.h>

namespace dom
{

struct GridCell;
struct Particle;

__global__ void particleToGridKernel(const ParticlesSoA particle_array, GridCell* __restrict__ grid_cell_array,
                                      int particle_count);

} /* namespace dom */
