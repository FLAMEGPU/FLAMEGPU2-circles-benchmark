#include <algorithm>
#include <cstdio>
#include <chrono>
#include <ctime>

#include "flamegpu/flame_api.h"

// Include the bruteforce implementation

#include "common.cuh"
#include "util.cuh"

// Prototypes for methods from other .cu files
void run_circles_bruteforce(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);
void run_circles_bruteforce_rtc(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);
void run_circles_spatial3D(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);
void run_circles_spatial3D_rtc(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);


bool run_experiment(
    const std::string LABEL,
    const int DEVICE,
    const uint64_t SEED,
    const uint32_t REPETITIONS,
    std::vector<RunSimulationInputs> INPUTS_STRUCTS,
    std::map<std::string, std::function<void(const RunSimulationInputs, RunSimulationOutputs&)>> MODELS
) { 
    printf("Running experiment %s - %zu configs, %zu simulators, %u repetitions\n", LABEL.c_str(), INPUTS_STRUCTS.size(), MODELS.size(), REPETITIONS);

    // Open CSV files
    std::string filename_perSimulationCSV = LABEL + std::string("_perSimulationCSV.csv");
    std::FILE * fp_perSimulationCSV = std::fopen(filename_perSimulationCSV.c_str(), "w");
    if(fp_perSimulationCSV == nullptr) {
        printf("Error: could not open csv file %s\n", filename_perSimulationCSV.c_str());
        std::fclose(fp_perSimulationCSV);
        return false;
    }
    std::string filename_perStepPerSimulationCSV = LABEL + std::string("_perStepPerSimulationCSV.csv");
    std::FILE * fp_perStepPerSimulationCSV = std::fopen(filename_perStepPerSimulationCSV.c_str(), "w");
    if(fp_perStepPerSimulationCSV == nullptr) {
        printf("Error: could not open csv file %s\n", filename_perStepPerSimulationCSV.c_str());
        std::fclose(fp_perSimulationCSV);
        std::fclose(fp_perStepPerSimulationCSV);
        return false;
    }

    // Output the CSV header for each output CSV file.
    if (fp_perSimulationCSV) {
        fprintf(fp_perSimulationCSV, "GPU,release_mode,seatbelts_on,model,steps,agent_count,comm_radius,repeat,mean_messageCount,ms_rtc,ms_simulation,ms_init,ms_exit,ms_step_mean\n");
    }
        
    if (fp_perStepPerSimulationCSV) {
        fprintf(fp_perStepPerSimulationCSV, "GPU,release_mode,seatbelts_on,model,steps,agent_count,comm_radius,repeat,step,ms_step\n");
    }


    const std::string deviceName = getGPUName(DEVICE);
    
    
    const uint32_t totalSimulationCount = INPUTS_STRUCTS.size() * MODELS.size() * REPETITIONS;
    uint32_t simulationIdx = 0; 
    
    // For each input structure
    for (const auto& inputStruct : INPUTS_STRUCTS) {
        for (const auto& modelNameFunctionPair : MODELS) {
            auto const& modelName = modelNameFunctionPair.first;
            auto const& modelFunction = modelNameFunctionPair.second; 
            for (uint32_t repeatIdx = 0u; repeatIdx < REPETITIONS; repeatIdx++){
                // Output progress
                printProgress(modelName, simulationIdx, totalSimulationCount, inputStruct.AGENT_COUNT, inputStruct.COMM_RADIUS, repeatIdx);

                // Run the simulation, capturing values for output.
                const RunSimulationInputs runInputs = {
                    modelName, 
                    inputStruct.HOST_SEED + repeatIdx, // Mutate the seed.
                    inputStruct.AGENT_COUNT, 
                    inputStruct.STEPS, 
                    DEVICE,
                    inputStruct.COMM_RADIUS
                };
                RunSimulationOutputs runOutputs = {};
                modelFunction(runInputs, runOutputs);

                // Add a row to the row per simulation csv file
                if (fp_perSimulationCSV) {
                    fprintf(
                        fp_perSimulationCSV, 
                        "%s,%d,%d,%s,%u,%u,%.3f,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                        deviceName.c_str(),
                        isReleaseMode(),
                        isSeatbeltsON(),
                        modelName.c_str(),
                        inputStruct.STEPS,
                        inputStruct.AGENT_COUNT,
                        inputStruct.COMM_RADIUS,
                        repeatIdx,
                        runOutputs.mean_messageCount,
                        runOutputs.ms_rtc,
                        runOutputs.ms_simulation,
                        runOutputs.ms_init,
                        runOutputs.ms_exit,
                        runOutputs.ms_stepMean); 
                }
                // Add a row to the per step per simulation CSV
                if (fp_perStepPerSimulationCSV) {
                    for(uint32_t step = 0; step < runOutputs.ms_per_step->size(); step++){
                        auto& ms_step = runOutputs.ms_per_step->at(step);
                        fprintf(fp_perStepPerSimulationCSV,
                            "%s,%d,%d,%s,%u,%u,%.3f,%u,%u,%.3f\n",
                            deviceName.c_str(),
                            isReleaseMode(),
                            isSeatbeltsON(),
                            modelName.c_str(),
                            inputStruct.STEPS,
                            inputStruct.AGENT_COUNT,
                            inputStruct.COMM_RADIUS,
                            repeatIdx,
                            step,
                            ms_step);
                    }
                }
                simulationIdx++;
            }
        }
    }
    
    // Close csv file handles.
    if(fp_perSimulationCSV){
        std::fclose(fp_perSimulationCSV);
        fp_perSimulationCSV = nullptr; 
    }
    if(fp_perStepPerSimulationCSV) {
        std::fclose(fp_perStepPerSimulationCSV);
        fp_perStepPerSimulationCSV = nullptr; 
    }

    return true;
}


bool experiment_total_scale_all(custom_cli cli){
    // Name the experiment - this will end up in filenames/paths.
    const std::string EXPERIMENT_LABEL="fixed-comm-radius";

    // Select comm radius value(s)
    const float COMM_RADIUS = 2.0f;

    // Select population sizes.
    std::vector<uint32_t> POPULATION_SIZES = {};
    const uint32_t imin = 14u; 
    const uint32_t imax = 15u;
    for(uint32_t i = imin; i < imax; i++){
        POPULATION_SIZES.push_back((1 << i));
        if(i < imax -1){
            POPULATION_SIZES.push_back((1 << i) + (1 << (i-1)));
        }
    }

    // Select the models to execute.
    std::map<std::string, std::function<void(const RunSimulationInputs, RunSimulationOutputs&)>> MODELS = {
        // {std::string("circles_spatial3D"), run_circles_spatial3D},
        {std::string("circles_spatial3D_rtc"), run_circles_spatial3D_rtc},
        // {std::string("circles_bruteforce"), run_circles_bruteforce},
        // {std::string("circles_bruteforce_rtc"), run_circles_bruteforce_rtc},
    };

    // Construct the vector of RunSimulationInputs to pass to the run_experiment method.
    auto INPUTS_STRUCTS = std::vector<RunSimulationInputs>();
    for(const auto& popSize : POPULATION_SIZES ){
        INPUTS_STRUCTS.push_back({
            "@todo-modelName", 
            cli.seed,
            popSize, 
            cli.steps, 
            cli.device,
            COMM_RADIUS
        });
    }

    // Run the experriment
    bool success = run_experiment(
        EXPERIMENT_LABEL,
        cli.device,
        cli.seed,
        cli.repetitions,
        INPUTS_STRUCTS,
        MODELS
    );

    return success;
}

bool experiment_density_spatial(const custom_cli cli) {
    // Name the experiment - this will end up in filenames/paths.
    const std::string EXPERIMENT_LABEL="variable-comm-radius";

    // Select comm radius value(s). 2.0f is default. Com radius is related to cuberoot of population....
    std::vector<float> COMM_RADII = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};

    std::vector<uint32_t> POPULATION_SIZES = {1<<14, 1<<16, 1<<18};
    // std::vector<uint32_t> POPULATION_SIZES = {1<<16};

    // Select population sizes.
    // std::vector<uint32_t> POPULATION_SIZES = {};
    // const uint32_t imin = 14u; 
    // const uint32_t imax = 15u;
    // for(uint32_t i = imin; i < imax; i++){
    //     POPULATION_SIZES.push_back((1 << i));
    //     if(i < imax -1){
    //         POPULATION_SIZES.push_back((1 << i) + (1 << (i-1)));
    //     }
    // }

    // Select the models to execute.
    std::map<std::string, std::function<void(const RunSimulationInputs, RunSimulationOutputs&)>> MODELS = {
        // {std::string("circles_spatial3D"), run_circles_spatial3D},
        {std::string("circles_spatial3D_rtc"), run_circles_spatial3D_rtc},
        // {std::string("circles_bruteforce"), run_circles_bruteforce},
        // {std::string("circles_bruteforce_rtc"), run_circles_bruteforce_rtc},
    };

    // Construct the vector of RunSimulationInputs to pass to the run_experiment method.
    auto INPUTS_STRUCTS = std::vector<RunSimulationInputs>();
    for(const auto& popSize : POPULATION_SIZES ){
        for(const auto& commRadius : COMM_RADII) {
            INPUTS_STRUCTS.push_back({
                "@todo-modelName", 
                cli.seed,
                popSize, 
                cli.steps, 
                cli.device,
                commRadius
            });
        }
    }

    // Run the experriment
    bool success = run_experiment(
        EXPERIMENT_LABEL,
        cli.device,
        cli.seed,
        cli.repetitions,
        INPUTS_STRUCTS,
        MODELS
    );

    return success;
}


// @todo - actual device poower state warmup? Maybe run the 0th sim twice and only use the second one?
// @todo deal with what happens if a simulation throws an exception?
int main(int argc, const char ** argv) {
    // Custom arg parsing, to prevent the current F2 arg parsing from occuring. 
    // @todo - improve arg parsing within F2. 
    custom_cli cli = parse_custom_cli(argc, argv);

    // Deal with the CSV output directory and abort if neccesary.
    printf("@todo - output dir via cli (and use it).\n");
    if(false){
        fprintf(stderr, "bad output directory? @todo\n");
        return EXIT_FAILURE;
    }

    // Launch each experiment.
    bool success_1 = experiment_total_scale_all(cli);
    bool success_2 = experiment_density_spatial(cli);

    // exit code
    return success_1 && success_2 ? EXIT_SUCCESS : EXIT_FAILURE;
}


// Todo:

/* 
+ [x] Change the order of loops so pops are first, toa llow early exit.
+ [x] RTC bruteforce
+ [ ] Move pop gen to init fn? so it gets timed.
+ [x] RTC Spatial
+ [x] Better disk io? 
+ [ ] Better error checking. 
+ [x] Plotting (.py)
    + [ ] Headless plotting.
+ [x] density experiment
+ [ ] Individual visualistion
+ [ ] Comments
+ [ ] Seeding?
+ [ ] readme
+ [ ] Check initialisation 
+ [ ] Decide on parameters to use, number of reps
+ [ ] V100 (bessemer) script(s) / trial run. Don't commit these to the public rpo.
+ [ ] limit the scale of some simulators - i.e. bruteforce cpp is horribly slow, so don't push the pops as far. 
+ [x] Have each agent store the message count it read. Exit fn that reduces theses and adds min/max/mean to the output data and CSVs. This might be useful
*/
