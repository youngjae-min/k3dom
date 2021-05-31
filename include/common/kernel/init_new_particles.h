#pragma once

#include <curand_kernel.h>
#include <device_launch_parameters.h>

namespace dom
{

struct GridCell;
struct Particle;

void normalize_particle_orders(float* particle_orders_array_accum, int particle_orders_count, int v_B);

__global__ void initParticlesKernel1(GridCell* __restrict__ grid_cell_array,
                                     ParticlesSoA particle_array,
                                     const float* __restrict__ particle_orders_array_accum, int cell_count);

__global__ void initParticlesKernel2(ParticlesSoA particle_array, const GridCell* __restrict__ grid_cell_array,
                                     curandState* __restrict__ global_state, float velocity, float min_vel, int grid_size,
                                     float new_weight, int particle_count);

__global__ void initNewParticlesKernel1(GridCell* __restrict__ grid_cell_array,
                                        const float* __restrict__ born_masses_array, ParticlesSoA birth_particle_array,
                                        const float* __restrict__ particle_orders_array_accum, int cell_count);

__global__ void initNewParticlesKernel2(ParticlesSoA birth_particle_array, const GridCell* __restrict__ grid_cell_array,
                                        curandState* __restrict__ global_state, float stddev_velocity,
                                        float max_velocity, float min_vel, int grid_size, int particle_count);
                                        
} /* namespace dom */