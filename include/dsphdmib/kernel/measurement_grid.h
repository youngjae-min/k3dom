#pragma once

#include <device_launch_parameters.h>

namespace dom
{
struct MeasurementCell;
struct GridCell;

__global__ void copyMassesKernel(const MeasurementCell* __restrict__ meas_cell_array, float* __restrict__ masses,
                                 int cell_count);

__global__ void calcMeasurementGridKernel(
	const float* __restrict__ m_x,
	const float* __restrict__ m_y,
	const float* __restrict__ m_z,
	const int num_meas, float window_radius,
	MeasurementCell* __restrict__ meas_grid,
	GridCell* __restrict__ grid_cell_array,
	int grid_size, int grid_size_z, int cell_count, float resolution, float max_range, float meas_occ_stddev,
	float sensor_rel_pos_x, float sensor_rel_pos_y, float sensor_rel_pos_z
);

}