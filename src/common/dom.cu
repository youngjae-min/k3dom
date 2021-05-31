#include "common/common.h"
#include "common/cuda_utils.h"
#include "common/dom.h"
#include "common/dom_types.h"

#include "common/kernel/ego_motion_compensation.h"
#include "common/kernel/init.h"
#include "common/kernel/init_new_particles.h"
// #include "k3dom/kernel/mass_update.h"
#include "common/kernel/particle_to_grid.h"
#include "common/kernel/predict.h"
#include "common/kernel/resampling.h"
#include "common/kernel/statistical_moments.h"
#include "common/kernel/update_persistent_particles.h"

// #include <thrust/sort.h>
// #include <thrust/transform.h>

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace dom
{

constexpr int BLOCK_SIZE = 256;

DOM_c::DOM_c(const Params& params)
    : params(params), grid_size(static_cast<int>(params.size / params.resolution)), grid_size_z(static_cast<int>(params.size_z / params.resolution)),
      particle_count(params.particle_count), grid_cell_count(grid_size * grid_size * grid_size_z),
      new_born_particle_count(params.new_born_particle_count), block_dim(BLOCK_SIZE), first_pose_received(false),
      first_measurement_received(false), updated_time(-1.0f), sensor_pos_x(0.0f), sensor_pos_y(0.0f), sensor_pos_z(0.0f),
      center_pos_x(0.0f), center_pos_y(0.0f), center_pos_z(0.0f)
{
    int device;
    CHECK_ERROR(cudaGetDevice(&device));

    cudaDeviceProp device_prop;
    CHECK_ERROR(cudaGetDeviceProperties(&device_prop, device));

    int blocks_per_sm = device_prop.maxThreadsPerMultiProcessor / block_dim.x;
    dim3 dim(device_prop.multiProcessorCount * blocks_per_sm);
    particles_grid = birth_particles_grid = grid_map_grid = dim;

    // particle_array_test.init(particle_count, false);
    particle_array.init(particle_count, true);
    particle_array_next.init(particle_count, true);
    birth_particle_array.init(new_born_particle_count, true);

    CHECK_ERROR(cudaMalloc(&grid_cell_array, grid_cell_count * sizeof(GridCell)));

    grid_cell_array_host = new GridCell[grid_cell_count];

    CHECK_ERROR(cudaMalloc(&born_masses_array, grid_cell_count * sizeof(float)));

    CHECK_ERROR(cudaMalloc(&vel_x_array, particle_count * sizeof(float)));
    CHECK_ERROR(cudaMalloc(&vel_y_array, particle_count * sizeof(float)));
    CHECK_ERROR(cudaMalloc(&vel_z_array, particle_count * sizeof(float)));

    CHECK_ERROR(cudaMalloc(&rand_array, particle_count * sizeof(float)));

    CHECK_ERROR(cudaMalloc(&rng_states, particles_grid.x * block_dim.x * sizeof(curandState)));

    initialize();
}

DOM_c::~DOM_c()
{
    // particle_array_test.free();
    particle_array.free();
    particle_array_next.free();
    birth_particle_array.free();

    ::free(grid_cell_array_host);

    CHECK_ERROR(cudaFree(grid_cell_array));

    CHECK_ERROR(cudaFree(born_masses_array));

    CHECK_ERROR(cudaFree(vel_x_array));
    CHECK_ERROR(cudaFree(vel_y_array));
    CHECK_ERROR(cudaFree(vel_z_array));

    CHECK_ERROR(cudaFree(rand_array));

    CHECK_ERROR(cudaFree(rng_states));
}

void DOM_c::initialize()
{
    cudaStream_t particles_stream, grid_stream;
    CHECK_ERROR(cudaStreamCreate(&particles_stream));
    CHECK_ERROR(cudaStreamCreate(&grid_stream));

    setupRandomStatesKernel<<<particles_grid, block_dim>>>(rng_states, 123456, particles_grid.x * block_dim.x);

    CHECK_ERROR(cudaGetLastError());
    CHECK_ERROR(cudaDeviceSynchronize());

    initGridCellsKernel<<<grid_map_grid, block_dim, 0, grid_stream>>>(grid_cell_array, grid_size,
                                                                      grid_cell_count, params.prior_free,
                                                                      params.prior_static, params.prior_dynamic, params.prior_occ);

    CHECK_ERROR(cudaGetLastError());

    CHECK_ERROR(cudaStreamDestroy(particles_stream));
    CHECK_ERROR(cudaStreamDestroy(grid_stream));
}

void DOM_c::updatePose(float new_x, float new_y, float new_z)
{
    sensor_pos_x = new_x;
    sensor_pos_y = new_y;
    sensor_pos_z = new_z;

    if (!first_pose_received)
    {
        first_pose_received = true;
        center_pos_x = sensor_pos_x - params.sensor_off_x;
        center_pos_y = sensor_pos_y - params.sensor_off_y;
        center_pos_z = sensor_pos_z - params.sensor_off_z;
    }
    else
    {
        const int x_move = static_cast<int>((sensor_pos_x - center_pos_x - params.sensor_off_x) / params.resolution);
        const int y_move = static_cast<int>((sensor_pos_y - center_pos_y - params.sensor_off_y) / params.resolution);
        const int z_move = static_cast<int>((sensor_pos_z - center_pos_z - params.sensor_off_z) / params.resolution);

        if (abs(x_move) >= params.map_shift_thresh || abs(y_move) >= params.map_shift_thresh || abs(z_move) >= params.map_shift_thresh)
        {
            GridCell* old_grid_cell_array;
            CHECK_ERROR(cudaMalloc(&old_grid_cell_array, grid_cell_count * sizeof(GridCell)));

            CHECK_ERROR(cudaMemcpy(old_grid_cell_array, grid_cell_array, grid_cell_count * sizeof(GridCell),
                                   cudaMemcpyDeviceToDevice));

            CHECK_ERROR(cudaDeviceSynchronize());
            initGridCellsKernel<<<grid_map_grid, block_dim>>>(grid_cell_array, grid_size,
                                                            grid_cell_count, params.prior_free,
                                                            params.prior_static, params.prior_dynamic, params.prior_occ);
            CHECK_ERROR(cudaGetLastError());

            moveParticlesKernel<<<particles_grid, block_dim>>>(particle_array, x_move, y_move, z_move, particle_count);
            CHECK_ERROR(cudaGetLastError());

            moveMapKernel<<<grid_map_grid, block_dim>>>(grid_cell_array, old_grid_cell_array, x_move, y_move, z_move,
                                                        grid_size, grid_size_z, grid_cell_count);
            CHECK_ERROR(cudaGetLastError());

            CHECK_ERROR(cudaFree(old_grid_cell_array));

            center_pos_x += x_move * params.resolution;
            center_pos_y += y_move * params.resolution;
            center_pos_z += z_move * params.resolution;
        }
    }
}

void DOM_c::particlePrediction(float dt)
{
    predictKernel<<<particles_grid, block_dim>>>(
        particle_array, rng_states, params.stddev_velocity, grid_size, grid_size_z, params.persistence_prob, dt,
        params.stddev_process_noise_position, params.stddev_process_noise_velocity, params.init_max_velocity, particle_count);

    CHECK_ERROR(cudaGetLastError());
}

void DOM_c::particleAssignment()
{
    reinitGridParticleIndices<<<grid_map_grid, block_dim>>>(grid_cell_array, grid_cell_count);

    CHECK_ERROR(cudaGetLastError());

    // sort particles
    thrust::device_ptr<int> grid_index_ptr(particle_array.grid_cell_idx);
    thrust::device_ptr<float> weight_ptr(particle_array.weight);
    thrust::device_ptr<bool> associated_ptr(particle_array.associated);
    thrust::device_ptr<glm::vec3> state_pos_ptr(particle_array.state_pos);
    thrust::device_ptr<glm::vec3> state_vel_ptr(particle_array.state_vel);

    auto it = thrust::make_zip_iterator(thrust::make_tuple(weight_ptr, associated_ptr, state_pos_ptr, state_vel_ptr));
    thrust::sort_by_key(grid_index_ptr, grid_index_ptr + particle_count, it);

    particleToGridKernel<<<particles_grid, block_dim>>>(particle_array, grid_cell_array, particle_count);

    CHECK_ERROR(cudaGetLastError());
}

void DOM_c::updatePersistentParticles()
{
    updatePersistentParticlesKernel1<<<particles_grid, block_dim>>>(particle_array, grid_cell_array,
                                                                    particle_count);

    CHECK_ERROR(cudaGetLastError());

    thrust::device_vector<float> weights_accum(particle_count);
    accumulate(particle_array.post_weight, weights_accum);
    float* weight_array_accum = thrust::raw_pointer_cast(weights_accum.data());

    updatePersistentParticlesKernel2<<<grid_map_grid, block_dim>>>(grid_cell_array,
                                                                    weight_array_accum, grid_cell_count);

    CHECK_ERROR(cudaGetLastError());

    updatePersistentParticlesKernel3<<<particles_grid, block_dim>>>(particle_array, grid_cell_array,
                                                                    particle_count);

    CHECK_ERROR(cudaGetLastError());

}

void DOM_c::initializeNewParticles()
{
    thrust::device_vector<float> particle_orders_accum(grid_cell_count);
    accumulate(born_masses_array, particle_orders_accum);
    float* particle_orders_array_accum = thrust::raw_pointer_cast(particle_orders_accum.data());

    normalize_particle_orders(particle_orders_array_accum, grid_cell_count, new_born_particle_count);

    initNewParticlesKernel1<<<grid_map_grid, block_dim>>>(grid_cell_array,
                                                          born_masses_array, birth_particle_array,
                                                          particle_orders_array_accum, grid_cell_count);

    CHECK_ERROR(cudaGetLastError());

    initNewParticlesKernel2<<<birth_particles_grid, block_dim>>>(birth_particle_array, grid_cell_array, rng_states,
                                                                 params.stddev_velocity, params.init_max_velocity, params.particle_min_vel,
                                                                 grid_size, new_born_particle_count);

    CHECK_ERROR(cudaGetLastError());
}

void DOM_c::statisticalMoments()
{
    statisticalMomentsKernel1<<<particles_grid, block_dim>>>(particle_array, vel_x_array, vel_y_array,
                                                             vel_z_array, particle_count);

    CHECK_ERROR(cudaGetLastError());
    
    thrust::device_vector<float> vel_x_accum(particle_count);
    accumulate(vel_x_array, vel_x_accum);
    float* vel_x_array_accum = thrust::raw_pointer_cast(vel_x_accum.data());

    thrust::device_vector<float> vel_y_accum(particle_count);
    accumulate(vel_y_array, vel_y_accum);
    float* vel_y_array_accum = thrust::raw_pointer_cast(vel_y_accum.data());

    thrust::device_vector<float> vel_z_accum(particle_count);
    accumulate(vel_z_array, vel_z_accum);
    float* vel_z_array_accum = thrust::raw_pointer_cast(vel_z_accum.data());

    statisticalMomentsKernel2<<<grid_map_grid, block_dim>>>(grid_cell_array, vel_x_array_accum, vel_y_array_accum,
                                                            vel_z_array_accum, grid_cell_count);

    CHECK_ERROR(cudaGetLastError());
}

void DOM_c::resampling()
{
    penalizeStaticParticlesKernel<<<particles_grid, block_dim>>>(particle_array, birth_particle_array,
                                                                params.particle_min_vel, particle_count, new_born_particle_count);
    CHECK_ERROR(cudaGetLastError());

    thrust::device_ptr<float> persistent_weights(particle_array.weight);
    thrust::device_ptr<float> new_born_weights(birth_particle_array.weight);

    thrust::device_vector<float> joint_weight_array;
    joint_weight_array.insert(joint_weight_array.end(), persistent_weights, persistent_weights + particle_count);
    joint_weight_array.insert(joint_weight_array.end(), new_born_weights, new_born_weights + new_born_particle_count);

    thrust::device_vector<float> joint_weight_accum(joint_weight_array.size());
    accumulate(joint_weight_array, joint_weight_accum);

    float joint_max = joint_weight_accum.back();

    resamplingGenerateRandomNumbersKernel<<<particles_grid, block_dim>>>(rand_array, rng_states, joint_max,
                                                                         particle_count);

    CHECK_ERROR(cudaGetLastError());

    thrust::device_ptr<float> rand_ptr(rand_array);
    thrust::device_vector<float> rand_vector(rand_ptr, rand_ptr + particle_count);

    thrust::sort(rand_vector.begin(), rand_vector.end());

    thrust::device_vector<int> idx_resampled(particle_count);
    calc_resampled_indices(joint_weight_accum, rand_vector, idx_resampled, joint_max);
    int* idx_array_resampled = thrust::raw_pointer_cast(idx_resampled.data());

    float new_weight = joint_max / particle_count;

    resamplingKernel<<<particles_grid, block_dim>>>(particle_array, particle_array_next, birth_particle_array,
                                                    idx_array_resampled, new_weight, particle_count);

    CHECK_ERROR(cudaGetLastError());
}

} /* namespace dom */
