#include "common/cuda_utils.h"
#include "common/dom_types.h"
#include "common/kernel/particle_to_grid.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

namespace dom
{

__device__ bool is_first_particle(const ParticlesSoA& particle_array, int i)
{
    return i == 0 || particle_array.grid_cell_idx[i] != particle_array.grid_cell_idx[i - 1];
}

__device__ bool is_last_particle(const ParticlesSoA& particle_array, int particle_count, int i)
{
    return i == particle_count - 1 || particle_array.grid_cell_idx[i] != particle_array.grid_cell_idx[i + 1];
}

__global__ void particleToGridKernel(const ParticlesSoA particle_array, GridCell* __restrict__ grid_cell_array,
                                     int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        int j = particle_array.grid_cell_idx[i];

        if (is_first_particle(particle_array, i))
        {
            grid_cell_array[j].start_idx = i;
        }
        if (is_last_particle(particle_array, particle_count, i))
        {
            grid_cell_array[j].end_idx = i;
        }
    }
}

} /* namespace dom */
