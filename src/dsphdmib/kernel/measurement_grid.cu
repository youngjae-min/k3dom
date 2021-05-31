#include "common/dom_types.h"
#include "dsphdmib/kernel/measurement_grid.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PI 3.14159265358979323846f

namespace dom
{

__device__ float2 combine_masses(float2 prior, float2 meas)
{
    // Masses: mOcc, mFree
    float occ = prior.x;
    float free = prior.y;

    float meas_occ = meas.x;
    float meas_free = meas.y;

    float unknown_pred = 1.0f - occ - free;
    float meas_cell_unknown = 1.0f - meas_occ - meas_free;
    float K = free * meas_occ + occ * meas_free;

    float2 res;
    res.x = (occ * meas_cell_unknown + unknown_pred * meas_occ + occ * meas_occ) / (1.0f - K);
    res.y = (free * meas_cell_unknown + unknown_pred * meas_free + free * meas_free) / (1.0f - K);

    return res;
}

__device__ float pOcc(
	const float x, const float y, const float z,
	const float m_x, const float m_y, const float m_z, const float delta = 0.6f
)
{
	float occ_max = 0.95f;
	float delta_sqr = delta * delta;
	float r_sqr = (x - m_x)*(x - m_x) + (y - m_y)*(y - m_y) + (z - m_z)*(z - m_z);

	return occ_max * exp(-0.5f * r_sqr / delta_sqr);
}

__device__ float pFree(
	const float x, const float y, const float z,
	const float max_range
)
{
	const float p_min = 0.15f;
	const float p_max = 1.0f;
	float r = sqrt(x*x + y * y + z * z);
	return p_min + r / max_range * (p_max - p_min);
}

__global__ void copyMassesKernel(const MeasurementCell* __restrict__ meas_cell_array, float* __restrict__ masses,
    int cell_count)
{
for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < cell_count; j += blockDim.x * gridDim.x)
{
masses[j] = meas_cell_array[j].occ_mass;
}
}

__global__ void calcMeasurementGridKernel(
	const float* __restrict__ m_x,
	const float* __restrict__ m_y,
	const float* __restrict__ m_z,
	const int num_meas, float window_radius,
	MeasurementCell* __restrict__ meas_grid,
    GridCell* __restrict__ grid_cell_array,
    int grid_size, int grid_size_z, int cell_count, float resolution, float max_range, float meas_occ_stddev,
    float sensor_rel_pos_x, float sensor_rel_pos_y, float sensor_rel_pos_z
)
{
    for (int cell_idx = blockIdx.x * blockDim.x + threadIdx.x; cell_idx < cell_count; cell_idx += blockDim.x * gridDim.x)
    {
        float x_idx = cell_idx % grid_size + 0.5f;
        float y_idx = (cell_idx % (grid_size * grid_size)) / grid_size + 0.5f;
        float z_idx = cell_idx / (grid_size * grid_size) + 0.5f;

        // assume the sensor is at the center of the grid
        float x = (x_idx - (float)grid_size / 2.0f) * resolution - sensor_rel_pos_x;
        float y = (y_idx - (float)grid_size / 2.0f) * resolution - sensor_rel_pos_y;
        float z = (z_idx - (float)grid_size_z / 2.0f) * resolution - sensor_rel_pos_z;
        
        int neighbor_size = 0;
        int neighbor_size_max = 100;
        int neighbor_idx[100];
        float abs_c_sqr = x * x + y * y + z * z;
        // save idx of ray passing nearby
        for (int i = 0; i < num_meas; i++) {
            float inner_prod = x * m_x[i] + y * m_y[i] + z * m_z[i];
            float abs_m_sqr = m_x[i] * m_x[i] + m_y[i] * m_y[i] + m_z[i] * m_z[i];
            if (inner_prod > 0 && abs_c_sqr - inner_prod * inner_prod / abs_m_sqr < window_radius * window_radius) {
                neighbor_idx[neighbor_size] = i;
                neighbor_size++;
                if (neighbor_size == neighbor_size_max) {break;}
            }
        }

        if (neighbor_size == 0) {
            // if no ray is passing nearby, then m_occ = m_free = 0
            meas_grid[cell_idx].occ_mass = 0.0f;
            meas_grid[cell_idx].free_mass = 0.0f;
            grid_cell_array[cell_idx].likelihood = 1.0f;
            grid_cell_array[cell_idx].p_A = 0.0f;
            continue;
        }
        
        bool is_all_farther = true;
        float max_occ_m = 0.0f;
        for (int i = 0; i < neighbor_size; i++) {
            int idx = neighbor_idx[i];
            if (is_all_farther && abs_c_sqr > m_x[idx] * m_x[idx] + m_y[idx] * m_y[idx] + m_z[idx] * m_z[idx]) {
                is_all_farther = false;
            }

            float occ_m = pOcc(x, y, z, m_x[idx], m_y[idx], m_z[idx], meas_occ_stddev);
            if (occ_m > max_occ_m) {
                max_occ_m = occ_m;
            }
        }

        // masses.x : mass of occupancy
        // masses.y : mass of free
        float2 masses = make_float2(0.0f, 0.0f);
        if (is_all_farther) {
            // if all points are farther than the grid center
            float uncertain_m = pFree(x, y, z, max_range);
            
            if (max_occ_m > uncertain_m) {
                // if max of pOcc > linear uncertainty
                masses.x = max_occ_m;
                masses.y = 0.0f;
            }
            else {
                masses.x = 0.0f;
                masses.y = 1.0f - uncertain_m;
            }
        }
        else {
            if (max_occ_m > 0.5f) {
                masses.x = max_occ_m;
            }
            else {
                masses.x = 0.0f;
            }
            masses.y = 0.0f;
        }

        // clipping
        const float epsilon = 0.00001f;
        masses.x = max(epsilon, min(1.0f - epsilon, masses.x));
        masses.y = max(epsilon, min(1.0f - epsilon, masses.y));

        meas_grid[cell_idx].occ_mass = masses.x;
        meas_grid[cell_idx].free_mass = masses.y;
        grid_cell_array[cell_idx].likelihood = 1.0f;
        grid_cell_array[cell_idx].p_A = 0.0f;
    }
	return;
}

} /* namespace dom */