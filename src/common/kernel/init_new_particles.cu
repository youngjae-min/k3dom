#include "common/common.h"
#include "common/cuda_utils.h"
#include "common/dom_types.h"
#include "common/kernel/init_new_particles.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace dom
{

__device__ int calc_start_idx(const float* __restrict__ particle_orders_array_accum, int index)
{
    if (index == 0)
    {
        return 0;
    }

    return static_cast<int>(particle_orders_array_accum[index - 1]);
}

__device__ int calc_end_idx(const float* __restrict__ particle_orders_array_accum, int index)
{
    return static_cast<int>(particle_orders_array_accum[index]) - 1;
}

__device__ int calc_num_assoc(int num_new_particles, float p_A)
{
    return static_cast<int>(roundf(num_new_particles * p_A));
}

__device__ float calc_weight_assoc(int nu_A, float p_A, float born_mass)
{
    return nu_A > 0 ? (p_A * born_mass) / nu_A : 0.0;
}

__device__ float calc_weight_unassoc(int nu_UA, float p_A, float born_mass)
{
    return nu_UA > 0 ? ((1.0 - p_A) * born_mass) / nu_UA : 0.0;
}

void normalize_particle_orders(float* particle_orders_array_accum, int particle_orders_count, int v_B)
{
    thrust::device_ptr<float> particle_orders_accum(particle_orders_array_accum);

    float max = 1.0f;
    cudaMemcpy(&max, &particle_orders_array_accum[particle_orders_count - 1], sizeof(float), cudaMemcpyDeviceToHost);
    thrust::transform(particle_orders_accum, particle_orders_accum + particle_orders_count, particle_orders_accum,
                      GPU_LAMBDA(float x) { return x * (v_B / max); });
}

__global__ void initParticlesKernel1(GridCell* __restrict__ grid_cell_array,
                                     ParticlesSoA particle_array,
                                     const float* __restrict__ particle_orders_array_accum, int cell_count)
{
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < cell_count; j += blockDim.x * gridDim.x)
    {
        int start_idx = calc_start_idx(particle_orders_array_accum, j);
        int end_idx = calc_end_idx(particle_orders_array_accum, j);

        for (int i = start_idx; i < end_idx + 1; i++)
        {
            particle_array.grid_cell_idx[i] = j;
        }
    }
}

__global__ void initParticlesKernel2(ParticlesSoA particle_array, const GridCell* __restrict__ grid_cell_array,
                                     curandState* __restrict__ global_state, float velocity, float min_vel, int grid_size,
                                     float new_weight, int particle_count)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState local_state = global_state[thread_id];

    for (int i = thread_id; i < particle_count; i += stride)
    {
        int cell_idx = particle_array.grid_cell_idx[i];

        float x = cell_idx % grid_size + 0.5f;
        float y = (cell_idx % (grid_size * grid_size)) / grid_size + 0.5f;
        float z = cell_idx / (grid_size * grid_size) + 0.5f;
        float vel_x = curand_uniform(&local_state, -velocity, velocity);
        float vel_y = curand_uniform(&local_state, -velocity, velocity);
        float vel_z = curand_uniform(&local_state, -velocity, velocity);

        float vel_sq = vel_x * vel_x + vel_y * vel_y + vel_z * vel_z;
        if (vel_sq < min_vel * min_vel && vel_sq != 0.0f) {
            float rate = min_vel / sqrtf(vel_sq);
            vel_x *= rate;
            vel_y *= rate;
            vel_z *= rate;
        }

        particle_array.weight[i] = new_weight;
        particle_array.state_pos[i] = glm::vec3(x, y, z);
        particle_array.state_vel[i] = glm::vec3(vel_x, vel_y, vel_z);
    }

    global_state[thread_id] = local_state;
}

__global__ void initNewParticlesKernel1(GridCell* __restrict__ grid_cell_array,
                                        const float* __restrict__ born_masses_array, ParticlesSoA birth_particle_array,
                                        const float* __restrict__ particle_orders_array_accum, int cell_count)
{
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < cell_count; j += blockDim.x * gridDim.x)
    {
        int start_idx = calc_start_idx(particle_orders_array_accum, j);
        int end_idx = calc_end_idx(particle_orders_array_accum, j);

        int num_new_particles = start_idx <= end_idx ? end_idx - start_idx + 1 : 0;
        float p_A = grid_cell_array[j].p_A;
        int nu_A = calc_num_assoc(num_new_particles, p_A);
        int nu_UA = num_new_particles - nu_A;
        grid_cell_array[j].w_A = calc_weight_assoc(nu_A, p_A, born_masses_array[j]);
        grid_cell_array[j].w_UA = calc_weight_unassoc(nu_UA, p_A, born_masses_array[j]);

        for (int i = start_idx; i < start_idx + nu_A; i++)
        {
            birth_particle_array.grid_cell_idx[i] = j;
            birth_particle_array.associated[i] = true;
        }

        for (int i = start_idx + nu_A; i < end_idx + 1; i++)
        {
            birth_particle_array.grid_cell_idx[i] = j;
            birth_particle_array.associated[i] = false;
        }
    }
}

__global__ void initNewParticlesKernel2(ParticlesSoA birth_particle_array, const GridCell* __restrict__ grid_cell_array,
                                        curandState* __restrict__ global_state, float stddev_velocity,
                                        float max_velocity, float min_vel, int grid_size, int particle_count)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState local_state = global_state[thread_id];

    for (int i = thread_id; i < particle_count; i += stride)
    {
        int cell_idx = birth_particle_array.grid_cell_idx[i];
        const GridCell& grid_cell = grid_cell_array[cell_idx];
        bool associated = birth_particle_array.associated[i];

        float x = cell_idx % grid_size + 0.5f;
        float y = (cell_idx % (grid_size * grid_size)) / grid_size + 0.5f;
        float z = cell_idx / (grid_size * grid_size) + 0.5f;

        // may employ different model along with association
        float vel_x = curand_uniform(&local_state, -max_velocity, max_velocity);
        float vel_y = curand_uniform(&local_state, -max_velocity, max_velocity);
        float vel_z = curand_uniform(&local_state, -max_velocity, max_velocity);

        // minimum velocity requirement to prevent static particles
        float vel_sq = vel_x * vel_x + vel_y * vel_y + vel_z * vel_z;
        if (vel_sq < min_vel * min_vel && vel_sq != 0.0f) {
            float rate = min_vel / sqrtf(vel_sq);
            vel_x *= rate;
            vel_y *= rate;
            vel_z *= rate;
        }

        if (associated)
        {
            birth_particle_array.weight[i] = grid_cell.w_A;
        }
        else
        {
            birth_particle_array.weight[i] = grid_cell.w_UA;
        }

        birth_particle_array.state_pos[i] = glm::vec3(x, y, z);
        birth_particle_array.state_vel[i] = glm::vec3(vel_x, vel_y, vel_z);
    }

    global_state[thread_id] = local_state;
}

} /* namespace dom */
