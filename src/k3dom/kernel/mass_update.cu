#include "common/common.h"
#include "common/cuda_utils.h"
#include "common/dom_types.h"
#include "k3dom/kernel/mass_update.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dom
{

inline __device__ float calc_credit(float mass, float mass_scale = 3.0f) {
    return 1.0f - 2.0f / (1.0f + expf(2*mass/mass_scale));
}

inline __device__ float positive_diff(float a, float b) {
    return max(0.0f, a - b);
}

inline __device__ float calc_ratio(float numerator, float rest) {
    return (numerator <= 0.0f || rest < 0.0f)? 0.0f : (1.0f / (1.0f + rest / numerator));
}

inline __device__ float kernel_func(float d, float ls, float sigma) {
    return max(0.0f, sigma * ((2.0f + cospif(2.0f * d / ls)) * (1.0f - d / ls) / 3.0f + sinpif(2.0f * d / ls) / (2.0f * 3.141592f)));
}

// since very close free and occupancy measurements cancel each other
inline __device__ float distance_with_margin(float distance, float resolution) {
    return max(0.9f * distance, distance - 2.0f * resolution);
}
__device__ float2 BGKI(const float* __restrict__ meas_array_x,
                       const float* __restrict__ meas_array_y,
                       const float* __restrict__ meas_array_z, int meas_len,
                       float x, float y, float z, float resolution, float sigma, float ls)
{
    float del_occ = 0.0f;
    float del_free = 0.0f;

    //TODO: efficient search considering kernel function (ex: rtree)
    float d_c = sqrtf(x * x + y * y + z * z);
    float d, d_proj, d_m, resize, x_m, y_m, z_m;
    for (int i = 0; i < meas_len; i++) {
        x_m = meas_array_x[i];
        y_m = meas_array_y[i];
        z_m = meas_array_z[i];
        
        // update for occ
        d = sqrtf((x - x_m) * (x - x_m) + (y - y_m) * (y - y_m) + (z - z_m) * (z - z_m));
        if (d < ls) {
            del_occ += kernel_func(d, ls, sigma);
        }

        // update for free (use closest point in each measurement ray as free measurement)
        d_m = sqrtf(x_m * x_m + y_m * y_m + z_m * z_m);
        d_proj = (x * x_m + y * y_m + z * z_m) / d_m;  // projected length
        if (d_proj <= 0.0f) {   // closest point: origin
            d = d_c;
        } else if (d_proj < distance_with_margin(d_m, resolution)) { // closest point: projected point
            d = sqrtf(d_c * d_c - d_proj * d_proj);
        } else if (d_proj < d_m) {  // closest point: point at border with margin
            resize = distance_with_margin(d_m, resolution) / d_m;
            d = sqrtf((x - resize * x_m) * (x - resize * x_m) +
                        (y - resize * y_m) * (y - resize * y_m) +
                        (z - resize * z_m) * (z - resize * z_m));
        } else {    // measured point is closest -> no free
            d = ls + 1.0f; // to disable free_mass update
        }
        if (d < ls) {
            del_free += kernel_func(d, ls, sigma);
        }
    }

    assert(del_free >= 0.0f && del_occ >= 0.0f);  // include checking nan

    return make_float2(del_occ, del_free);
}

__device__ void predict_mass(GridCell& grid_cell, float dynamic_mass_pred, float d_max, float dt, float gamma, float mass_scale) {

    grid_cell.static_mass *= powf(gamma, dt);
    grid_cell.dynamic_mass = min(dynamic_mass_pred, max(0.0f, d_max - grid_cell.static_mass));
    grid_cell.free_mass = d_max - grid_cell.dynamic_mass;
}

__device__ void update_mass(GridCell& grid_cell, int cell_idx, 
                            const float* __restrict__ meas_array_x,
                            const float* __restrict__ meas_array_y,
                            const float* __restrict__ meas_array_z, int meas_len,
                            int grid_size, int grid_size_z, float resolution, float sigma, float ls, float mass_scale,
                            float sensor_x, float sensor_y, float sensor_z) {
    float x_idx = cell_idx % grid_size + 0.5f;
    float y_idx = (cell_idx % (grid_size * grid_size)) / grid_size + 0.5f;
    float z_idx = cell_idx / (grid_size * grid_size) + 0.5f;

    // coordinates from sensor
    float x = (x_idx - (float)grid_size / 2.0f) * resolution - sensor_x;
    float y = (y_idx - (float)grid_size / 2.0f) * resolution - sensor_y;
    float z = (z_idx - (float)grid_size_z / 2.0f) * resolution - sensor_z;
    
    // kernel accumulation
    float2 del = BGKI(meas_array_x, meas_array_y, meas_array_z, meas_len, x, y, z, resolution, sigma, ls);
    float del_occ = del.x;
    float del_free = del.y;

    if (del_free > 0.0f || del_occ > 0.0f) {      
        float m_f = grid_cell.free_mass;
        float m_s = grid_cell.static_mass;
        float m_d = grid_cell.dynamic_mass;

        assert(m_f >= 0.0f && m_s >= 0.0f && m_d >= 0.0f);

        /////////////// update step1: rebalancing ///////////////
        float del_sigma = del_free + del_occ;
        float prime_sigma = m_f + m_s + m_d;
        
        float credit = calc_credit(sqrtf(del_sigma * prime_sigma), mass_scale);

        float lamda_DtoF = credit * positive_diff(del_free, del_occ) / del_sigma * m_d / prime_sigma;
        float lamda_StoFD = credit * positive_diff(del_free, del_occ) / del_sigma * positive_diff(m_s + m_d, m_f) / prime_sigma;
        float lamda_FtoD = credit * positive_diff(del_occ, del_free) / del_sigma * positive_diff(m_f, m_s + m_d) / prime_sigma;

        float diff_f = lamda_StoFD / 2.0f * m_s + lamda_DtoF * m_d - lamda_FtoD * m_f;
        float diff_s = -lamda_StoFD * m_s;
        float diff_d = lamda_FtoD * m_f + lamda_StoFD / 2.0f * m_s - lamda_DtoF * m_d;

        m_f += diff_f;
        m_s += diff_s;
        m_d += diff_d;
        
        /////////////// update step2: kernel inference ///////////////
        credit = calc_credit(m_s + m_d, mass_scale);
        float beta = (1.0f - credit) + credit * calc_ratio(m_s, m_d);
        grid_cell.static_mass = m_s + beta * del_occ;
        grid_cell.dynamic_mass = m_d + (1.0f - beta) * del_occ;
        grid_cell.free_mass = m_f + del_free;
    }
    assert(grid_cell.static_mass >= 0.0f && grid_cell.dynamic_mass >= 0.0f && grid_cell.free_mass >= 0.0f);  // include checking nan
}

__device__ float separate_newborn_part(float m_dyn_pred, float m_total_pred, float m_dyn_up, float p_B)
{
    if (m_dyn_pred <= 0.0f) {   // (0,0) case is included here
        return m_dyn_up;
    } else if (m_total_pred <= m_dyn_pred) {
        return 0.0f;
    } else {
        return (m_dyn_up * p_B * (m_total_pred - m_dyn_pred)) / (m_dyn_pred + p_B * (m_total_pred - m_dyn_pred));
    }
}

__global__ void gridCellPredictionUpdateKernel(GridCell* __restrict__ grid_cell_array, ParticlesSoA particle_array,
                                               float* __restrict__ born_masses_array, float p_B, int cell_count,
                                               const float* __restrict__ meas_array_x,
                                               const float* __restrict__ meas_array_y,
                                               const float* __restrict__ meas_array_z, int meas_len, float dt,
                                               int grid_size, int grid_size_z, float resolution,
                                               float sigma, float ls, float gamma, float mass_scale,
                                               float sensor_x, float sensor_y, float sensor_z)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
    {
        int start_idx = grid_cell_array[i].start_idx;
        int end_idx = grid_cell_array[i].end_idx;

        float pred_total = (grid_cell_array[i].free_mass + grid_cell_array[i].dynamic_mass) * powf(gamma, dt);
        float dynamic_mass_pred = 0.0f;
        if (start_idx != -1)
        {
            for (int j = start_idx; j < end_idx + 1; j++) {
                dynamic_mass_pred += particle_array.weight[j];
            }
            assert(dynamic_mass_pred >= 0.0f);
        }
        predict_mass(grid_cell_array[i], dynamic_mass_pred, pred_total, dt, gamma, mass_scale);
        update_mass(grid_cell_array[i], i, meas_array_x, meas_array_y, meas_array_z, meas_len, grid_size, grid_size_z, 
                    resolution, sigma, ls, mass_scale, sensor_x, sensor_y, sensor_z);

        born_masses_array[i] = separate_newborn_part(dynamic_mass_pred, pred_total, grid_cell_array[i].dynamic_mass, p_B);
        grid_cell_array[i].pers_mass = grid_cell_array[i].dynamic_mass - born_masses_array[i];
        grid_cell_array[i].pred_mass = dynamic_mass_pred;
    }
}

} /* namespace dom */
