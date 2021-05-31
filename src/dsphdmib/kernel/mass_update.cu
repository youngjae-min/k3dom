#include "common/common.h"
#include "common/cuda_utils.h"
#include "common/dom_types.h"
#include "dsphdmib/kernel/mass_update.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace dom
{

__device__ float predict_free_mass(const GridCell& grid_cell, float m_occ_pred, float alpha = 0.9)
{
    return min(alpha * grid_cell.free_mass, 1.0f - m_occ_pred);
}

__device__ float2 update_masses(float m_occ_pred, float m_free_pred, const MeasurementCell& meas_cell)
{
    float unknown_pred = 1.0 - m_occ_pred - m_free_pred;
    float meas_unknown = 1.0 - meas_cell.free_mass - meas_cell.occ_mass;
    float K = m_free_pred * meas_cell.occ_mass + m_occ_pred * meas_cell.free_mass;

    float occ_mass =
        (m_occ_pred * meas_unknown + unknown_pred * meas_cell.occ_mass + m_occ_pred * meas_cell.occ_mass) / (1.0 - K);
    float free_mass =
        (m_free_pred * meas_unknown + unknown_pred * meas_cell.free_mass + m_free_pred * meas_cell.free_mass) /
        (1.0 - K);

    return make_float2(occ_mass, free_mass);
}

__device__ float separate_newborn_part(float m_occ_pred, float m_occ_up, float p_B)
{
    return (m_occ_up * p_B * (1.0 - m_occ_pred)) / (m_occ_pred + p_B * (1.0 - m_occ_pred));
}

__device__ void normalize_weights(const ParticlesSoA& particle_array, int start_idx,
                                  int end_idx, float occ_pred)
{
    for (int i = start_idx; i < end_idx + 1; i++)
    {
        particle_array.weight[i] /= occ_pred;
    }
}

__global__ void gridCellPredictionUpdateKernel(GridCell* __restrict__ grid_cell_array, ParticlesSoA particle_array,
                                               const MeasurementCell* __restrict__ meas_cell_array,
                                               float* __restrict__ born_masses_array, float p_B, int cell_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cell_count; i += blockDim.x * gridDim.x)
    {
        int start_idx = grid_cell_array[i].start_idx;
        int end_idx = grid_cell_array[i].end_idx;

        float2 masses_up;
        float rho_b = 0.0f;
        float m_occ_pred = 0.0f;
        if (start_idx != -1)
        {
            for (int j = start_idx; j < end_idx + 1; j++) {
                m_occ_pred += particle_array.weight[j];
            }

            if (m_occ_pred > 1.0f)
            {
                normalize_weights(particle_array, start_idx, end_idx, m_occ_pred);
                m_occ_pred = 1.0f;
            }
            assert (m_occ_pred >= 0.0f);

            float m_free_pred = predict_free_mass(grid_cell_array[i], m_occ_pred);
            masses_up = update_masses(m_occ_pred, m_free_pred, meas_cell_array[i]);
            rho_b = separate_newborn_part(m_occ_pred, masses_up.x, p_B);
        }
        else
        {
            float m_occ = grid_cell_array[i].occ_mass;
            float m_free = predict_free_mass(grid_cell_array[i], m_occ);
            masses_up = update_masses(m_occ, m_free, meas_cell_array[i]);
        }
        born_masses_array[i] = rho_b;
        grid_cell_array[i].pers_mass = masses_up.x - rho_b;
        grid_cell_array[i].free_mass = masses_up.y;
        grid_cell_array[i].occ_mass = masses_up.x;
        grid_cell_array[i].pred_mass = m_occ_pred;
    }
}

} /* namespace dom */
