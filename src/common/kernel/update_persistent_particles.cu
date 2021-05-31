#include "common/common.h"
#include "common/cuda_utils.h"
#include "common/dom_types.h"
#include "common/kernel/update_persistent_particles.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dom
{

__device__ float calc_norm(float weight_sum, float rho_p)
{
    return weight_sum > 0.0f ? rho_p / weight_sum : 0.0f;
}

__global__ void updatePersistentParticlesKernel1(const ParticlesSoA particle_array,
                                                 const GridCell* __restrict__ grid_cell_array,
                                                 int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        // unnormalized weight
        particle_array.post_weight[i] = grid_cell_array[particle_array.grid_cell_idx[i]].likelihood * particle_array.weight[i];
    }
}

__global__ void updatePersistentParticlesKernel2(GridCell* __restrict__ grid_cell_array,
                                                 const float* __restrict__ weight_array_accum, int cell_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
    {
        int start_idx = grid_cell_array[i].start_idx;
        int end_idx = grid_cell_array[i].end_idx;

        if (start_idx != -1)
        {
            float mass_accum = subtract(weight_array_accum, start_idx, end_idx);
            float rho_p = grid_cell_array[i].pers_mass;
            grid_cell_array[i].mu_A = calc_norm(mass_accum, rho_p);
            grid_cell_array[i].mu_UA = calc_norm(grid_cell_array[i].pred_mass, rho_p);
        }
    }
}

__global__ void updatePersistentParticlesKernel3(const ParticlesSoA particle_array,
                                                 const GridCell* __restrict__ grid_cell_array,
                                                 int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        const GridCell& cell = grid_cell_array[particle_array.grid_cell_idx[i]];
        particle_array.weight[i] = cell.p_A * cell.mu_A * particle_array.post_weight[i] + (1.0f - cell.p_A) * cell.mu_UA * particle_array.weight[i];
    }
}

} /* namespace dom */
