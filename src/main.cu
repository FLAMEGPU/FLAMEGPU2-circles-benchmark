#include <algorithm>
#include <cstdio>
#include <chrono>
#include <ctime>

#include "flamegpu/flamegpu.h"
#include "common.cuh"
#include "util.cuh"

#define DRY_RUN 0

// Prototypes for methods from other .cu files
void run_circles_bruteforce(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);
void run_circles_bruteforce_rtc(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);
void run_circles_spatial3D(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);
void run_circles_spatial3D_rtc(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);
void run_circles_bruteforce_sorted(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);
void run_circles_bruteforce_rtc_sorted(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);


bool run_experiment(
    const std::string LABEL,
    const int DEVICE,
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
        fprintf(fp_perSimulationCSV, "GPU,release_mode,seatbelts_on,model,steps,agent_count,env_width,comm_radius,sort_period,repeat,agent_density,mean_message_count,s_rtc,s_simulation,s_init,s_exit,s_step_mean\n");
    }
        
    if (fp_perStepPerSimulationCSV) {
        fprintf(fp_perStepPerSimulationCSV, "GPU,release_mode,seatbelts_on,model,steps,agent_count,env_width,comm_radius,sort_period,repeat,agent_density,step,s_step\n");
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
                    inputStruct.SEED + repeatIdx,
                    inputStruct.AGENT_COUNT, 
                    inputStruct.ENV_WIDTH,
                    inputStruct.COMM_RADIUS,
                    inputStruct.SORT_PERIOD
                };
                RunSimulationOutputs runOutputs = {};
                modelFunction(runInputs, runOutputs);

                // Add a row to the row per simulation csv file
                if (fp_perSimulationCSV) {
                    fprintf(
                        fp_perSimulationCSV, 
                        "%s,%d,%d,%s,%u,%u,%.6f,%.6f,%u,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                        deviceName.c_str(),
                        isReleaseMode(),
                        isSeatbeltsON(),
                        modelName.c_str(),
                        inputStruct.STEPS,
                        inputStruct.AGENT_COUNT,
                        inputStruct.ENV_WIDTH,
                        inputStruct.COMM_RADIUS,
                        inputStruct.SORT_PERIOD,
                        repeatIdx,
                        runOutputs.agentDensity,
                        runOutputs.mean_messageCount,
                        runOutputs.s_rtc,
                        runOutputs.s_simulation,
                        runOutputs.s_init,
                        runOutputs.s_exit,
                        runOutputs.s_stepMean); 
                }
                // Add a row to the per step per simulation CSV
                if (fp_perStepPerSimulationCSV) {
                    for(uint32_t step = 0; step < runOutputs.s_per_step->size(); step++){
                        auto& s_step = runOutputs.s_per_step->at(step);
                        fprintf(fp_perStepPerSimulationCSV,
                            "%s,%d,%d,%s,%u,%u,%.6f,%.6f,%u,%u,%.6f,%u,%.6f\n",
                            deviceName.c_str(),
                            isReleaseMode(),
                            isSeatbeltsON(),
                            modelName.c_str(),
                            inputStruct.STEPS,
                            inputStruct.AGENT_COUNT,
                            inputStruct.ENV_WIDTH,
                            inputStruct.COMM_RADIUS,
                            inputStruct.SORT_PERIOD,
                            repeatIdx,
                            runOutputs.agentDensity,
                            step,
                            s_step);
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
    // Fixed sort period
    const uint32_t SORT_PERIOD = 1u; 

    // Sweep over environment widths, which lead to scaled 
    // Env width needs to be atleast 5 * comm_radius to not read all messages? (so that there are bins in atleast each dim?)
    // @density 1, 8 width = 512 pop. 16 = 4k, 20 = 8k, 40 width = 64k pop, 100 = 1million.
    // const std::vector<float> ENV_WIDTHS = {8.f, 12.f, 16.f, 20.f};
    // const std::vector<float> ENV_WIDTHS = {8.f, 12.f, 16.f, 20.f, 30.f, 40.f, 50.f, 60.f, 70.f, 80.f, 90.f, 100.f};
    std::vector<float> ENV_WIDTHS = {};


    const std::vector<float> TARGET_ENV_VOLUMES = {10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000};
    for(const float& targetVolume : TARGET_ENV_VOLUMES){
        const float envWidth = round(cbrt(targetVolume));
        const float actualVolume = envWidth * envWidth * envWidth;
        const float badness = (actualVolume - targetVolume) / targetVolume;
        ENV_WIDTHS.push_back(envWidth);
        // printf("targetVolume %f actualVolume %f width %f, volumeBadness %f\n", targetVolume, actualVolume, envWidth, badness);
    }


    // Select the models to execute.
    std::map<std::string, std::function<void(const RunSimulationInputs, RunSimulationOutputs&)>> MODELS = {
        {std::string("circles_spatial3D"), run_circles_spatial3D},
        {std::string("circles_spatial3D_rtc"), run_circles_spatial3D_rtc},
        {std::string("circles_bruteforce"), run_circles_bruteforce},
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
            COMM_RADIUS,
            SORT_PERIOD
        });
    }

    // Run the experriment
    bool success = run_experiment(
        EXPERIMENT_LABEL,
        cli.device,
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

    const uint32_t SORT_PERIOD = 1u; 
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
        {std::string("circles_spatial3D"), run_circles_spatial3D},
        {std::string("circles_spatial3D_rtc"), run_circles_spatial3D_rtc},
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
                COMM_RADIUS,
                SORT_PERIOD
            });
        }
    }

    // Run the experriment
    bool success = run_experiment(
        EXPERIMENT_LABEL,
        cli.device,
        cli.repetitions,
        INPUTS_STRUCTS,
        MODELS,
        cli.dry
    );

    return success;
}

bool experiment_sort_period(custom_cli cli){
    // Name the experiment - this will end up in filenames/paths.
    const std::string EXPERIMENT_LABEL="sort-period";
    
    const uint32_t popSize = 64000;
    const float ENV_WIDTH = 40.0f;  

    const std::vector<float> comm_radii = {2.0f, 4.0f, 6.0f, 8.0f};
    const std::vector<uint32_t> sortPeriods = {0u, 1u, 2u, 5u, 10u, 20u, 50u, 100u, 200u}; 

    // Select the models to execute.
    std::map<std::string, std::function<void(const RunSimulationInputs, RunSimulationOutputs&)>> MODELS = {
        {std::string("circles_spatial3D"), run_circles_spatial3D},
        {std::string("circles_spatial3D_rtc"), run_circles_spatial3D_rtc},
        //{std::string("circles_bruteforce"), run_circles_bruteforce},
        //{std::string("circles_bruteforce_rtc"), run_circles_bruteforce_rtc},
        //{std::string("circles_bruteforce_sorted"), run_circles_bruteforce_sorted},
        //{std::string("circles_bruteforce_rtc_sorted"), run_circles_bruteforce_rtc_sorted},
    };

    // Construct the vector of RunSimulationInputs to pass to the run_experiment method.
    auto INPUTS_STRUCTS = std::vector<RunSimulationInputs>();
    for(const auto& sortPeriod : sortPeriods){
	for(const auto& comm_radius : comm_radii) {
            // Envwidth is scaled with population size.
            INPUTS_STRUCTS.push_back({
                cli.device,
                cli.steps,
                cli.seed,
                popSize,
                ENV_WIDTH,
                comm_radius,
                sortPeriod
            });
	}
    }

    // Run the experriment
    bool success = run_experiment(
        EXPERIMENT_LABEL,
        cli.device,
        cli.repetitions,
        INPUTS_STRUCTS,
        MODELS,
        cli.dry
    );

    return success;
}

bool experiment_comm_radius(custom_cli cli){
    // Name the experiment - this will end up in filenames/paths.
    const std::string EXPERIMENT_LABEL="comm-radius";
    
    const uint32_t popSize = 64000;
    const float ENV_WIDTH = 40.0f;  
    const uint32_t SORT_PERIOD = 1u; 
    const std::vector<float> comm_radii = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f, 32.0f, 34.0f, 36.0f, 38.0f, 40.0f};

    // Select the models to execute.
    std::map<std::string, std::function<void(const RunSimulationInputs, RunSimulationOutputs&)>> MODELS = {
        {std::string("circles_spatial3D"), run_circles_spatial3D},
        {std::string("circles_spatial3D_rtc"), run_circles_spatial3D_rtc},
        {std::string("circles_bruteforce"), run_circles_bruteforce},
        {std::string("circles_bruteforce_rtc"), run_circles_bruteforce_rtc},
        {std::string("circles_bruteforce_sorted"), run_circles_bruteforce_sorted},
        {std::string("circles_bruteforce_rtc_sorted"), run_circles_bruteforce_rtc_sorted},
    };

    // Construct the vector of RunSimulationInputs to pass to the run_experiment method.
    auto INPUTS_STRUCTS = std::vector<RunSimulationInputs>();
    for(const auto& comm_radius : comm_radii ){
        // Envwidth is scaled with population size.
        INPUTS_STRUCTS.push_back({
            cli.device,
            cli.steps,
            cli.seed,
            popSize,
            ENV_WIDTH,
            comm_radius,
            SORT_PERIOD
        });
    }

    // Run the experriment
    bool success = run_experiment(
        EXPERIMENT_LABEL,
        cli.device,
        cli.repetitions,
        INPUTS_STRUCTS,
        MODELS,
        cli.dry
    );

    return success;
}

int main(int argc, const char ** argv) {
    // Custom arg parsing, to prevent the current F2 arg parsing from occuring. 
    custom_cli cli = parse_custom_cli(argc, argv);

    // Launch each experiment.
    //bool success_1 = experiment_total_scale_all(cli);
    //bool success_2 = experiment_density_spatial(cli);
    //bool success_3 = experiment_comm_radius(cli);
    bool success_4 = experiment_sort_period(cli);

    // exit code
    //return success_1 && success_2 && success_3 ? EXIT_SUCCESS : EXIT_FAILURE;
    return success_4 ? EXIT_SUCCESS : EXIT_FAILURE;
}
