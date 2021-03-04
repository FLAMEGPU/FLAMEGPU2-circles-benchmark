#pragma once

#include <vector>

// Control validation of the circles model?
#define CIRCLES_VALIDATION 0

// Constant applied across all implementations?
#define COMM_RADIUS 2.0f
#define ENV_REPULSE 0.05f

const unsigned long long int DEFAULT_SEED = 123u;
const unsigned int DEFAULT_STEPS = 200u;
const unsigned int DEFAULT_REPETITIONS = 3u;
const int DEFAULT_DEVICE = 0;


struct custom_cli {
    unsigned long long int seed = DEFAULT_SEED;
    unsigned int steps = DEFAULT_STEPS;
    unsigned int repetitions = DEFAULT_REPETITIONS;
    int device = DEFAULT_DEVICE;
};

struct RunSimulationInputs {
    std::string modelName;
    const uint64_t HOST_SEED;
    const uint32_t AGENT_COUNT;
    const uint32_t STEPS;
    const int32_t CUDA_DEVICE;
};


struct RunSimulationOutputs { 
    std::shared_ptr<std::vector<float>> ms_per_step = nullptr;
    float ms_rtc = 0.f;
    float ms_simulation = 0.f;
    float ms_init = 0.f;
    float ms_exit = 0.f;
    float ms_stepMean = 0.f;
};
