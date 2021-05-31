#pragma once

#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <curand_kernel.h>

namespace dom
{

struct Particle;

__global__ void penalizeStaticParticlesKernel(const ParticlesSoA particle_array,
                                            const ParticlesSoA birth_particle_array,
                                            float min_speed, int particle_count, int birt_particle_count);

__global__ void resamplingGenerateRandomNumbersKernel(float* __restrict__ rand_array,
                                                      curandState* __restrict__ global_state, float max,
                                                      int particle_count);

void calc_resampled_indices(thrust::device_vector<float>& weight_accum, thrust::device_vector<float>& rand_array,
                            thrust::device_vector<int>& indices, float accum_max);

__global__ void resamplingKernel(const ParticlesSoA particle_array, ParticlesSoA particle_array_next,
                                 const ParticlesSoA birth_particle_array, const int* __restrict__ idx_array_resampled,
                                 float new_weight, int particle_count);

} /* namespace dom */
