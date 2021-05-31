#pragma once

#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

namespace dom
{

struct Particle;

__global__ void predictKernel(ParticlesSoA particle_array, curandState* __restrict__ global_state, float velocity,
                              int grid_size, int grid_size_z, float p_S, const float dt,
                              float process_noise_position, float process_noise_velocity, 
                              float init_max_velocity, int particle_count);

} /* namespace dom */
