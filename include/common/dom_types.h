#pragma once

#include "common/cuda_utils.h"
#include <glm/vec3.hpp>

#include <iostream>
#include <math.h>
#include <vector>

namespace dom
{
struct GridCell
{
    int start_idx;
    int end_idx;

    float mu_A;
    float mu_UA;

    float w_A;
    float w_UA;

    float mean_x_vel;
    float mean_y_vel;
    float mean_z_vel;

    // below two should be updated for every measurement if available
    float likelihood;
    float p_A;

    // For k3dom,
    // free/static/dynamic mass: concentration parameter (alpha) for dirichlet distribution
    // pers_mass: total (dynamic) mass of persistent particles (dynamic_mass = pers_mass + mass for new born particles)
    // pred_mass: total (dynamic) mass of newly incident particles before mass update
    // For ds-phd/mib,
    // free/occ mass: evidence in Dempster-Shafer theory
    // pers_mass: total (occupancy) mass of persistent particles (occ_mass = pers_mass + mass for new born particles)
    // pred_mass: total (occupancy) mass of newly incident particles before mass update
    float free_mass;
    float static_mass;
    float dynamic_mass;
    float occ_mass;
    float pers_mass;
    float pred_mass;

    // for k3dom
    inline float total_mass() const {return free_mass + static_mass + dynamic_mass;}
    inline float occ_mean() const {return (static_mass + dynamic_mass) / total_mass();}
    inline float dyn_mean() const {return dynamic_mass / total_mass();}
    float calcVar() const {
        float mass_sum = total_mass();
        if (mass_sum == 0.0f) {
            return 1e6; // large number
        }
        float max_mass = std::max(std::max(free_mass, static_mass), dynamic_mass);
        float mean = max_mass / mass_sum;
        return  mean * (1.0f - mean) / (mass_sum + 1.0f);
    }

    // for ds-phd/mib
    inline float occ_prob() const {return occ_mass + (1.0f - occ_mass - free_mass) / 2.0f;}
    inline float mean_vel_sq() const {
        return mean_x_vel * mean_x_vel + mean_y_vel * mean_y_vel + mean_z_vel * mean_z_vel;
    }
};

// only for ds-phd/mib to generate measurement evidence map
struct MeasurementCell
{
    float free_mass;
    float occ_mass;
    // float likelihood;
    // float p_A;
};

struct ParticlesSoA
{
    glm::vec3* state_pos;
    glm::vec3* state_vel;
    int* grid_cell_idx;
    float* weight;
    float* post_weight; // likelihood * weight (currently only for k3dom)
    bool* associated;

    int size;
    bool device;

    ParticlesSoA() : size(0), device(true) {}

    ParticlesSoA(int new_size, bool is_device) { init(new_size, is_device); }

    void init(int new_size, bool is_device)
    {
        size = new_size;
        device = is_device;
        if (device)
        {
            CHECK_ERROR(cudaMalloc((void**)&state_pos, size * sizeof(glm::vec3)));
            CHECK_ERROR(cudaMalloc((void**)&state_vel, size * sizeof(glm::vec3)));
            CHECK_ERROR(cudaMalloc((void**)&grid_cell_idx, size * sizeof(int)));
            CHECK_ERROR(cudaMalloc((void**)&weight, size * sizeof(float)));
            CHECK_ERROR(cudaMalloc((void**)&post_weight, size * sizeof(float)));
            CHECK_ERROR(cudaMalloc((void**)&associated, size * sizeof(bool)));
        }
        else
        {
            state_pos = (glm::vec3*)malloc(size * sizeof(glm::vec3));
            state_vel = (glm::vec3*)malloc(size * sizeof(glm::vec3));
            grid_cell_idx = (int*)malloc(size * sizeof(int));
            weight = (float*)malloc(size * sizeof(float));
            post_weight = (float*)malloc(size * sizeof(float));
            associated = (bool*)malloc(size * sizeof(bool));
        }
    }

    void free()
    {
        if (device)
        {
            CHECK_ERROR(cudaFree(state_pos));
            CHECK_ERROR(cudaFree(state_vel));
            CHECK_ERROR(cudaFree(grid_cell_idx));
            CHECK_ERROR(cudaFree(weight));
            CHECK_ERROR(cudaFree(post_weight));
            CHECK_ERROR(cudaFree(associated));
        }
        else
        {
            ::free(state_pos);
            ::free(state_vel);
            ::free(grid_cell_idx);
            ::free(weight);
            ::free(post_weight);
            ::free(associated);
        }
    }

    void copy(const ParticlesSoA& other, cudaMemcpyKind kind)
    {
        CHECK_ERROR(cudaMemcpy(grid_cell_idx, other.grid_cell_idx, size * sizeof(int), kind));
        CHECK_ERROR(cudaMemcpy(weight, other.weight, size * sizeof(float), kind));
        CHECK_ERROR(cudaMemcpy(post_weight, other.post_weight, size * sizeof(float), kind));
        CHECK_ERROR(cudaMemcpy(associated, other.associated, size * sizeof(bool), kind));
        CHECK_ERROR(cudaMemcpy(state_pos, other.state_pos, size * sizeof(glm::vec3), kind));
        CHECK_ERROR(cudaMemcpy(state_vel, other.state_vel, size * sizeof(glm::vec3), kind));
    }

    ParticlesSoA& operator=(const ParticlesSoA& other)
    {
        if (this != &other)
        {
            copy(other, cudaMemcpyDeviceToDevice);
        }

        return *this;
    }

    __device__ void copy(const ParticlesSoA& other, int index, int other_index)
    {
        grid_cell_idx[index] = other.grid_cell_idx[other_index];
        weight[index] = other.weight[other_index];
        post_weight[index] = other.post_weight[other_index];
        associated[index] = other.associated[other_index];
        state_pos[index] = other.state_pos[other_index];
        state_vel[index] = other.state_vel[other_index];
    }
};

} /* namespace dom */
