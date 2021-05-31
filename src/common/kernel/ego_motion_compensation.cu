#include "common/common.h"
#include "common/cuda_utils.h"
#include "common/dom_types.h"
#include "common/kernel/ego_motion_compensation.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dom
{

__global__ void moveParticlesKernel(ParticlesSoA particle_array, int x_move, int y_move, int z_move, int particle_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < particle_count; i += blockDim.x * gridDim.x)
    {
        particle_array.state_pos[i][0] -= x_move;
        particle_array.state_pos[i][1] -= y_move;
        particle_array.state_pos[i][2] -= z_move;
    }
}

__global__ void moveMapKernel(GridCell* __restrict__ grid_cell_array, const GridCell* __restrict__ old_grid_cell_array,
                              int x_move, int y_move, int z_move, int grid_size, int grid_size_z, int cell_count)
{
    for (int cell_idx = blockIdx.x * blockDim.x + threadIdx.x; cell_idx < cell_count; cell_idx += blockDim.x * gridDim.x)
    {
        int x = cell_idx % grid_size;
        int y = (cell_idx % (grid_size * grid_size)) / grid_size;
        int z = cell_idx / (grid_size * grid_size);
        
        int new_z = z + z_move;
        int new_y = y + y_move;
        int new_x = x + x_move;

        if (new_x >= 0 && new_x < grid_size && new_y >= 0 && new_y < grid_size && new_z >= 0 && new_z < grid_size_z)
        {
            int new_index = new_x + grid_size * new_y + grid_size * grid_size * new_z;
            grid_cell_array[cell_idx] = old_grid_cell_array[new_index];
        }
    }
}

} /* namespace dom */
