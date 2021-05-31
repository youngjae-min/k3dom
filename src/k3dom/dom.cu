#include "common/common.h"
#include "common/cuda_utils.h"
#include "k3dom/dom.h"
#include "common/dom_types.h"

#include "common/kernel/init_new_particles.h"
#include "k3dom/kernel/mass_update.h"

#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace dom
{

DOM::DOM(const Params& params)
    : DOM_c(params)
{
}

DOM::~DOM()
{
}

void DOM::updateGrid(float t, std::vector<float>& measurements_x,
                    std::vector<float>& measurements_y, std::vector<float>& measurements_z)
{
    if (updated_time > 0) { // skip the first time w/ updated_time = -1.0f
        float dt = t - updated_time;
        particlePrediction(dt);
        particleAssignment();
        gridCellOccupancyUpdate(dt, measurements_x, measurements_y, measurements_z);
        updatePersistentParticles();
        initializeNewParticles();
        statisticalMoments();
        resampling();
        
        // particle_array_test.copy(particle_array, cudaMemcpyDeviceToHost);
        particle_array = particle_array_next;
    }
    else {initializeParticles(measurements_x, measurements_y, measurements_z);}

    CHECK_ERROR(cudaDeviceSynchronize());

    updated_time = t;
}

void DOM::initializeParticles(std::vector<float>& measurements_x,
                              std::vector<float>& measurements_y, std::vector<float>& measurements_z)
{
    /// reflect measurements first without any assigned particles
    gridCellOccupancyUpdate(0.0f, measurements_x, measurements_y, measurements_z);
    
    CHECK_ERROR(cudaGetLastError());

    thrust::device_vector<float> particle_orders_accum(grid_cell_count);
    accumulate(born_masses_array, particle_orders_accum);
    float* particle_orders_array_accum = thrust::raw_pointer_cast(particle_orders_accum.data());

    float new_weight = 1.0f / particle_count;

    normalize_particle_orders(particle_orders_array_accum, grid_cell_count, particle_count);

    initParticlesKernel1<<<grid_map_grid, block_dim>>>(grid_cell_array, particle_array,
                                                       particle_orders_array_accum, grid_cell_count);

    CHECK_ERROR(cudaGetLastError());

    initParticlesKernel2<<<particles_grid, block_dim>>>(
        particle_array, grid_cell_array, rng_states, params.init_max_velocity, params.particle_min_vel, grid_size, new_weight, particle_count);

    CHECK_ERROR(cudaGetLastError());
}

void DOM::gridCellOccupancyUpdate(float dt, std::vector<float>& measurements_x,
                                  std::vector<float>& measurements_y, std::vector<float>& measurements_z)
{
    meas_len = measurements_x.size();
    assert(meas_len == measurements_y.size() && meas_len == measurements_z.size());

    CHECK_ERROR(cudaMalloc(&meas_x, meas_len * sizeof(float)));
    CHECK_ERROR(cudaMalloc(&meas_y, meas_len * sizeof(float)));
    CHECK_ERROR(cudaMalloc(&meas_z, meas_len * sizeof(float)));
    CHECK_ERROR(cudaMemcpy(meas_x, measurements_x.data(), meas_len * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(meas_y, measurements_y.data(), meas_len * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(meas_z, measurements_z.data(), meas_len * sizeof(float), cudaMemcpyHostToDevice));
    
    gridCellPredictionUpdateKernel<<<grid_map_grid, block_dim>>>(grid_cell_array, particle_array,
                                                                 born_masses_array,
                                                                 params.birth_prob, grid_cell_count,
                                                                 meas_x, meas_y, meas_z, meas_len, dt,
                                                                 grid_size, grid_size_z, params.resolution,
                                                                 params.sigma, params.ls, params.gamma, params.mass_scale,
                                                                 sensor_pos_x - center_pos_x, sensor_pos_y - center_pos_y,
                                                                 sensor_pos_z - center_pos_z);
    
    CHECK_ERROR(cudaGetLastError());

    CHECK_ERROR(cudaFree(meas_x));
    CHECK_ERROR(cudaFree(meas_y));
    CHECK_ERROR(cudaFree(meas_z));
}

} /* namespace dom */
