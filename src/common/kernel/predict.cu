#include "common/common.h"
#include "common/cuda_utils.h"
#include "common/dom_types.h"
#include "common/kernel/predict.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dom
{

__global__ void predictKernel(ParticlesSoA particle_array, curandState* __restrict__ global_state, float velocity,
                              int grid_size, int grid_size_z, float p_S, const float dt,
                              float process_noise_position, float process_noise_velocity, 
                              float init_max_velocity, int particle_count)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState local_state = global_state[thread_id];

    for (int i = thread_id; i < particle_count; i += stride)
    {
        float noise_pos_x = curand_normal(&local_state, 0.0f, process_noise_position);
        float noise_pos_y = curand_normal(&local_state, 0.0f, process_noise_position);
        float noise_pos_z = curand_normal(&local_state, 0.0f, process_noise_position);
        float noise_vel_x = curand_normal(&local_state, 0.0f, process_noise_velocity);
        float noise_vel_y = curand_normal(&local_state, 0.0f, process_noise_velocity);
        float noise_vel_z = curand_normal(&local_state, 0.0f, process_noise_velocity);
        glm::vec3 process_noise_pos(noise_pos_x, noise_pos_y, noise_pos_z);
        glm::vec3 process_noise_vel(noise_vel_x, noise_vel_y, noise_vel_z);

        particle_array.state_pos[i] += particle_array.state_vel[i] * dt + process_noise_pos;
        particle_array.state_vel[i] += process_noise_vel;
        particle_array.weight[i] *= p_S;

        particle_array.state_vel[i].x = clamp(particle_array.state_vel[i].x, - 5 * init_max_velocity, 5 * init_max_velocity);
        particle_array.state_vel[i].y = clamp(particle_array.state_vel[i].y, - 5 * init_max_velocity, 5 * init_max_velocity);
        particle_array.state_vel[i].z = clamp(particle_array.state_vel[i].z, - 5 * init_max_velocity, 5 * init_max_velocity);

        glm::vec3 state = particle_array.state_pos[i];
        float x = state[0];
        float y = state[1];
        float z = state[2];

        // Particle out of grid [0,grid_size) so get rid of its chance of being resampled
        if ((x >= grid_size || x < 0) || (y >= grid_size || y < 0) || (z >= grid_size_z || z < 0))
        {
            particle_array.weight[i] = 0.0f;
        }

        int idx_x = clamp(static_cast<int>(x), 0, grid_size - 1);
        int idx_y = clamp(static_cast<int>(y), 0, grid_size - 1);
        int idx_z = clamp(static_cast<int>(z), 0, grid_size_z - 1);
        particle_array.grid_cell_idx[i] = idx_x + grid_size * idx_y + grid_size * grid_size * idx_z;
    }

    global_state[thread_id] = local_state;
}

} /* namespace dom */
