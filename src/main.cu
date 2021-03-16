#include <algorithm>
#include <cstdio>
#include <chrono>
#include <ctime>

#include "flamegpu/flame_api.h"
#include "common.cuh"
#include "util.cuh"

#define SEED_PRIME 97
#define DRY_RUN 0

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
    std::map<std::string, std::function<void(const RunSimulationInputs, RunSimulationOutputs&)>> MODELS,
    const bool dry 
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
        fprintf(fp_perSimulationCSV, "GPU,release_mode,seatbelts_on,model,steps,agent_count,env_width,comm_radius,repeat,agent_density,mean_message_count,ms_rtc,ms_simulation,ms_init,ms_exit,ms_step_mean,pre_flame_used_bytes,pre_flame_free_bytes,flame_used_bytes,flame_free_bytes\n");
    }
        
    if (fp_perStepPerSimulationCSV) {
        fprintf(fp_perStepPerSimulationCSV, "GPU,release_mode,seatbelts_on,model,steps,agent_count,env_width,comm_radius,repeat,agent_density,step,ms_step\n");
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
                printProgress(
                    modelName, 
                    simulationIdx, 
                    totalSimulationCount, 
                    inputStruct.AGENT_COUNT, 
                    inputStruct.ENV_WIDTH, 
                    inputStruct.COMM_RADIUS, 
                    repeatIdx);

                // Only print the progress if a dry run.
                if (dry) { 
                    continue;
                }

                // Run the simulation, capturing values for output.
                const RunSimulationInputs runInputs = {
                    DEVICE,
                    inputStruct.STEPS, 
                    inputStruct.HOST_SEED + (repeatIdx * SEED_PRIME), // Mutate the seed.
                    inputStruct.AGENT_COUNT, 
                    inputStruct.ENV_WIDTH,
                    inputStruct.COMM_RADIUS
                };
                RunSimulationOutputs runOutputs = {};
                modelFunction(runInputs, runOutputs);

                // If the run was successful, output csv, otherwise skip.
                // @todo - ideally this would also not attempt to run any larger sims, but oh well.
                if(runOutputs.completed) {
                    // Add a row to the row per simulation csv file
                    if (fp_perSimulationCSV) {
                        fprintf(
                            fp_perSimulationCSV, 
                            "%s,%d,%d,%s,%u,%u,%.3f,%.3f,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%zu,%zu,%zu,%zu\n",
                            deviceName.c_str(),
                            isReleaseMode(),
                            isSeatbeltsON(),
                            modelName.c_str(),
                            inputStruct.STEPS,
                            inputStruct.AGENT_COUNT,
                            inputStruct.ENV_WIDTH,
                            inputStruct.COMM_RADIUS,
                            repeatIdx,
                            runOutputs.agentDensity,
                            runOutputs.mean_messageCount,
                            runOutputs.ms_rtc,
                            runOutputs.ms_simulation,
                            runOutputs.ms_init,
                            runOutputs.ms_exit,
                            runOutputs.ms_stepMean,
                            runOutputs.preFlameUsedBytes,
                            runOutputs.preFlameFreeBytes,
                            runOutputs.flameUsedBytes,
                            runOutputs.flameFreeBytes); 
                        fflush(fp_perSimulationCSV);
                    }
                    // Add a row to the per step per simulation CSV
                    if (fp_perStepPerSimulationCSV) {
                        for(uint32_t step = 0; step < runOutputs.ms_per_step->size(); step++){
                            auto& ms_step = runOutputs.ms_per_step->at(step);
                            fprintf(fp_perStepPerSimulationCSV,
                                "%s,%d,%d,%s,%u,%u,%.3f,%.3f,%u,%.3f,%u,%.3f\n",
                                deviceName.c_str(),
                                isReleaseMode(),
                                isSeatbeltsON(),
                                modelName.c_str(),
                                inputStruct.STEPS,
                                inputStruct.AGENT_COUNT,
                                inputStruct.ENV_WIDTH,
                                inputStruct.COMM_RADIUS,
                                repeatIdx,
                                runOutputs.agentDensity,
                                step,
                                ms_step);
                            fflush(fp_perStepPerSimulationCSV);
                        }
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
    const std::string EXPERIMENT_LABEL="fixed-density";

    // Fixed comm radius 
    const float COMM_RADIUS = 2.f;
    // Fixed density
    const float DENSITY = 1.0f; 

    // Sweep over environment widths, which lead to scaled 
    // Env width needs to be atleast 5 * comm_radius to not read all messages? (so that there are bins in atleast each dim?)
    // @density 1, 8 width = 512 pop. 16 = 4k, 20 = 8k, 40 width = 64k pop, 100 = 1million.
    // const std::vector<float> ENV_WIDTHS = {8.f, 12.f, 16.f, 20.f};
    // const std::vector<float> ENV_WIDTHS = {8.f, 12.f, 16.f, 20.f, 30.f, 40.f, 50.f, 60.f, 70.f, 80.f, 90.f, 100.f};
    std::vector<float> ENV_WIDTHS = {};

    // @todo - need to try catch calls to simulate() if I want to sample multiple repetitions, and output some kind of error value (and then abort any larger sizes.)

    // const std::vector<float> TARGET_ENV_VOLUMES = {10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000};

    // Start with some small target volumes to get some sampling at the low end.
    std::vector<float> TARGET_ENV_VOLUMES = {1, 32, 128, 256, 512, 1024, 4096, 16000, 32000, 64000};
    // Then sample at every 100k upto 1 million.
    for(uint32_t i = 1; i < 10; i++){
        const float scaleFactor = 100000.f;
        TARGET_ENV_VOLUMES.push_back(i * scaleFactor);
    }
    // Then sample at every 1million upto 10 milliion 
    for(uint32_t i = 1; i < 10; i++){
        const float scaleFactor = 1000000.f;
        TARGET_ENV_VOLUMES.push_back(i * scaleFactor);
    }
    // Then sample at every 10 million upto 200 million.
    for(uint32_t i = 1; i < 21; i++){
        const float scaleFactor = 10000000.f;
        TARGET_ENV_VOLUMES.push_back(i * scaleFactor);
    }

    for(const float& targetVolume : TARGET_ENV_VOLUMES){
        const float envWidth = round(cbrt(targetVolume));
        const float actualVolume = envWidth * envWidth * envWidth;
        const float badness = (actualVolume - targetVolume) / targetVolume;
        ENV_WIDTHS.push_back(envWidth);
        // printf("targetVolume %f actualVolume %f width %f, volumeBadness %f\n", targetVolume, actualVolume, envWidth, badness);
    }
    // exit(1);


    // Select the models to execute.
    std::map<std::string, std::function<void(const RunSimulationInputs, RunSimulationOutputs&)>> MODELS = {
        // {std::string("circles_spatial3D"), run_circles_spatial3D},
        {std::string("circles_spatial3D_rtc"), run_circles_spatial3D_rtc},
        // {std::string("circles_bruteforce"), run_circles_bruteforce},
        {std::string("circles_bruteforce_rtc"), run_circles_bruteforce_rtc},
    };

    // Construct the vector of RunSimulationInputs to pass to the run_experiment method.
    auto INPUTS_STRUCTS = std::vector<RunSimulationInputs>();
    // for(const auto& popSize : POPULATION_SIZES ){
        // const float envWidth = static_cast<float>(ceil(cbrt(popSize)));
    for(const auto& envWidth : ENV_WIDTHS ){
        const uint32_t popSize = static_cast<float>(ceil((envWidth * envWidth * envWidth) * DENSITY)); 
        // Envwidth is scaled with population size.
        INPUTS_STRUCTS.push_back({
            cli.device,
            cli.steps,
            cli.seed,
            popSize,
            envWidth,
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
        MODELS,
        cli.dry
    );

    return success;
}

bool experiment_density_spatial(const custom_cli cli) {
    // Name the experiment - this will end up in filenames/paths.
    const std::string EXPERIMENT_LABEL="variable-density";

    // Vary the density / environment width for several agent populations.
    std::vector<float> COMM_VOLUME_FRACTIONS = {};

    // Fixed comm radius 
    const float COMM_RADIUS = 2.f;

    // Sweep over densities.
    // std::vector<float> DENSITIES = {1.f, 2.f, 4.f, 8.f}; 
    // std::vector<float> DENSITIES = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 8.f, 9.f, 10.f}; 
    
    // Sweep over environment widths, which lead to scaled 
    // std::vector<float> ENV_WIDTHS = {8.f, 20.f, 40.f};
    // const std::vector<float> ENV_WIDTHS = {8.f, 12.f, 16.f, 20.f, 30.f, 40.f, 50.f, 60.f, 70.f, 80.f, 90.f, 100.f};
    // const std::vector<float> ENV_WIDTHS = {40, 50, 60, 70};

    std::vector<float> DENSITIES = {1.f, 2.f, 3.f, 4.f}; 
    const std::vector<float> TARGET_ENV_VOLUMES = {10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 150000, 200000, 250000, 300000,  350000, 400000, 450000, 500000};
    std::vector<float> ENV_WIDTHS = {};
    for(const float& targetVolume : TARGET_ENV_VOLUMES){
        const float envWidth = round(cbrt(targetVolume));
        const float actualVolume = envWidth * envWidth * envWidth;
        const float badness = (actualVolume - targetVolume) / targetVolume;
        ENV_WIDTHS.push_back(envWidth);
        // printf("targetVolume %f actualVolume %f width %f, volumeBadness %f\n", targetVolume, actualVolume, envWidth, badness);
    }



    // Select the models to execute.
    std::map<std::string, std::function<void(const RunSimulationInputs, RunSimulationOutputs&)>> MODELS = {
        // {std::string("circles_spatial3D"), run_circles_spatial3D},
        // {std::string("circles_spatial3D_rtc"), run_circles_spatial3D_rtc},
        // {std::string("circles_bruteforce"), run_circles_bruteforce},
        // {std::string("circles_bruteforce_rtc"), run_circles_bruteforce_rtc},
    };

    // Construct the vector of RunSimulationInputs to pass to the run_experiment method.
    auto INPUTS_STRUCTS = std::vector<RunSimulationInputs>();
    for(const auto& envWidth : ENV_WIDTHS ){
        for(const auto& density : DENSITIES ){
            const uint32_t popSize = static_cast<float>(ceil((envWidth * envWidth * envWidth) * density)); 
            // Envwidth is scaled with population size.
            INPUTS_STRUCTS.push_back({
                cli.device,
                cli.steps,
                cli.seed,
                popSize,
                envWidth,
                COMM_RADIUS
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
        MODELS,
        cli.dry
    );

    return success;
}


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

/* 
// Todo:
+ [x] Change the order of loops so pops are first, toa llow early exit.
+ [x] RTC bruteforce
+ [ ] Move pop gen to init fn? so it gets timed? Agent vec in init would be good.
+ [x] RTC Spatial
+ [ ] Output to a specified directory (which may or may not exist?)
+ [ ] Better error checking. 
    + [ ] if a simulation throws an exception?
    + [ ] If could not create the csv file
+ [x] Plotting (.py)
+ [x] Headless plotting.
+ [x] density experiment
+ [ ] Individual visualistion
+ [ ] Comments
+ [x] Seeding?
+ [ ] readme
+ [ ] Check initialisation 
+ [ ] Decide on parameters to use, number of reps
+ [x] V100 (bessemer) script(s) / trial run. Don't commit these to the public rpo.
+ [ ] limit the scale of some simulators - i.e. bruteforce cpp is horribly slow, so don't push the pops as far. 
+ [x] Have each agent store the message count it read. Exit fn that reduces theses and adds min/max/mean to the output data and CSVs. This might be useful
+ [ ]actual device poower state warmup? Maybe run the 0th sim twice and only use the second one?
*/