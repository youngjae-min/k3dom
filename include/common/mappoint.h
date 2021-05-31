#pragma once

#include "pcl/point_types.h"

struct MapPoint
{
PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
float v_x, v_y, v_z;              // velocities
float occ_val;                    // occupancy value to be classified
float dyn_val;                    // dynamic value to be classified
float eval_aug;                   // additional value to filter out unobserved grids
float dyn_aug;                    // additional value to classify dynamic
EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned

} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (MapPoint,           // here we assume a XYZ + ~ (as fields)
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, v_x, v_x)
                                (float, v_y, v_y)
                                (float, v_z, v_z)
                                (float, occ_val, occ_val)
                                (float, dyn_val, dyn_val)
                                (float, eval_aug, eval_aug)
                                (float, dyn_aug, dyn_aug)
)