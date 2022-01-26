#pragma once

#include <vector>
#include <numeric>

// Control validation of the circles model?
#define CIRCLES_VALIDATION 0

// Constant applied across all implementations?
#define ENV_REPULSE 0.05f

const unsigned long long int DEFAULT_SEED = 0u;
const unsigned int DEFAULT_STEPS = 200u;
const unsigned int DEFAULT_REPETITIONS = 3u;
const int DEFAULT_DEVICE = 0;


struct custom_cli {
    unsigned long long int seed = DEFAULT_SEED;
    unsigned int steps = DEFAULT_STEPS;
    unsigned int repetitions = DEFAULT_REPETITIONS;
    int device = DEFAULT_DEVICE;
    bool dry = false;
};

struct RunSimulationInputs {
    const int32_t CUDA_DEVICE;
    const uint32_t STEPS;
    const uint64_t SEED;
    const uint32_t AGENT_COUNT;
    const float ENV_WIDTH;
    const float COMM_RADIUS;
};


struct RunSimulationOutputs { 
    std::shared_ptr<std::vector<double>> s_per_step = nullptr;
    double s_rtc = 0.f;
    double s_simulation = 0.f;
    double s_init = 0.f;
    double s_exit = 0.f;
    double s_stepMean = 0.f;
    float mean_messageCount = 0.f;
    float agentDensity = 0.f;
};
