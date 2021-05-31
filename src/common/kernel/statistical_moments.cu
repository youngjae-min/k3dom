#include "common/common.h"
#include "common/cuda_utils.h"
#include "common/dom_types.h"
#include "common/kernel/statistical_moments.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dom
{

__device__ float calc_mean(const float* __restrict__ vel_array_accum, int start_idx, int end_idx, float rho_p)
{
    if (rho_p > 0.0f)
    {
        float vel_accum = subtract(vel_array_accum, start_idx, end_idx);
        return vel_accum / rho_p;
    }
    return 0.0f;
}

__global__ void statisticalMomentsKernel1(const ParticlesSoA particle_array,
                                          float* __restrict__ vel_x_array, float* __restrict__ vel_y_array,
                                          float* __restrict__ vel_z_array, int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        float weight = particle_array.weight[i];
        float vel_x = particle_array.state_vel[i].x;
        float vel_y = particle_array.state_vel[i].y;
        float vel_z = particle_array.state_vel[i].z;
        vel_x_array[i] = weight * vel_x;
        vel_y_array[i] = weight * vel_y;
        vel_z_array[i] = weight * vel_z;
    }
}

__global__ void statisticalMomentsKernel2(GridCell* __restrict__ grid_cell_array,
                                          const float* __restrict__ vel_x_array_accum,
                                          const float* __restrict__ vel_y_array_accum,
                                          const float* __restrict__ vel_z_array_accum,
                                          int cell_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
    {
        int start_idx = grid_cell_array[i].start_idx;
        int end_idx = grid_cell_array[i].end_idx;
        float rho_p = grid_cell_array[i].pers_mass;

        if (start_idx != -1)
        {
            grid_cell_array[i].mean_x_vel = calc_mean(vel_x_array_accum, start_idx, end_idx, rho_p);
            grid_cell_array[i].mean_y_vel = calc_mean(vel_y_array_accum, start_idx, end_idx, rho_p);
            grid_cell_array[i].mean_z_vel = calc_mean(vel_z_array_accum, start_idx, end_idx, rho_p);
        }
        else
        {
            grid_cell_array[i].mean_x_vel = 0.0f;
            grid_cell_array[i].mean_y_vel = 0.0f;
            grid_cell_array[i].mean_z_vel = 0.0f;
        }
    }
}

} /* namespace dom */
