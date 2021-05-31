#pragma once

#include "common/dom_types.h"
#include "common/dom.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <memory>

#include <vector>

namespace dom
{

class DOM : public DOM_c
{
public:
    DOM(const Params& params);
    ~DOM();

    void updateGrid(float t, std::vector<float>& measurements_x,
                    std::vector<float>& measurements_y, std::vector<float>& measurements_z);

    void initializeParticles(std::vector<float>& measurements_x,
                             std::vector<float>& measurements_y, std::vector<float>& measurements_z);
    void gridCellOccupancyUpdate(std::vector<float>& measurements_x,
                                 std::vector<float>& measurements_y, std::vector<float>& measurements_z);

    void generateMeasGrid(std::vector<float>& measurements_x, std::vector<float>& measurements_y,
                          std::vector<float>& measurements_z);

public:
    MeasurementCell* meas_cell_array;
    float window_radius; // range for looking up rays to generate meas_grid
};

} /* namespace dom */
