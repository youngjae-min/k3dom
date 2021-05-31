#include "common/common.h"
#include "common/cuda_utils.h"
#include "common/dom_types.h"
#include "common/kernel/init.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dom
{

__global__ void setupRandomStatesKernel(curandState* __restrict__ states, unsigned long long seed, int count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
    {
        curand_init(seed, i, 0, &states[i]);
    }
}

__global__ void initGridCellsKernel(GridCell* __restrict__ grid_cell_array,
                                    int grid_size, int cell_count, float prior_f, float prior_s, float prior_d, float prior_o)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
    {
        grid_cell_array[i].free_mass = prior_f;
        grid_cell_array[i].static_mass = prior_s;
        grid_cell_array[i].dynamic_mass = prior_d;
        grid_cell_array[i].occ_mass = prior_o;
        grid_cell_array[i].start_idx = -1;
        grid_cell_array[i].end_idx = -1;

        grid_cell_array[i].likelihood = 1.0f;
        grid_cell_array[i].p_A = 0.0f;
    }
}

__global__ void reinitGridParticleIndices(GridCell* __restrict__ grid_cell_array, int cell_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
    {
        grid_cell_array[i].start_idx = -1;
        grid_cell_array[i].end_idx = -1;
    }
}

} /* namespace dom */
