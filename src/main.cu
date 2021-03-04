#include <algorithm>
#include <cstdio>
#include <chrono>
#include <ctime>

#include "flamegpu/flame_api.h"

// Include the bruteforce implementation

#include "common.cuh"

void run_circles_bruteforce(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);
void run_circles_bruteforce_rtc(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);
void run_circles_spatial3D(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);
void run_circles_spatial3D_rtc(const RunSimulationInputs runInputs, RunSimulationOutputs &runOutputs);


// Convert some compiler flag values into global constants (if defined) to be output to file
#if defined(NDEBUG) || defined(_NDEBUG)
    const bool RELEASE_MODE = true;
#else 
    const bool RELEASE_MODE = false;
#endif

#if defined(SEATBELTS) && !SEATBELTS
    const bool SEATBELTS_ON = false;
#else 
    const bool SEATBELTS_ON = true;
#endif


void print_cli_help(const int argc, const char ** argv );
custom_cli parse_custom_cli(const int argc, const char ** argv);

void printProgress(const std::string modelName, const uint32_t count, const uint32_t total, const uint32_t agentCount, const uint32_t repeat){
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char buf[sizeof "2011-10-08T07:07:09Z"];
    std::strftime(buf, sizeof buf, "%FT%TZ", std::gmtime(&now));
    fprintf(stdout, "%s: %u/%u: %s %u %u\n", buf, count + 1, total, modelName.c_str(), agentCount, repeat);
}

// @todo - actual device poower state warmup? Maybe run the 0th sim twice and only use the second one?
// @todo deal with what happens if a simulation throws an exception?
int main(int argc, const char ** argv) {
    // Custom arg parsing, to prevent the current F2 arg parsing from occuring. 
    // @todo - improve arg parsing within F2. 
    custom_cli cli = parse_custom_cli(argc, argv);
    
    // Define the benchmark parameters. I.e. initial environment width, max width, method of interpolation etc. 
    /* std::vector<uint32_t> POPULATION_SIZES = {
        1u << 4,
        1u << 5,
        1u << 6,
        // 1u << 7,
        // 1u << 8,
        // 1u << 9,
        // 1u << 10,
        // 1u << 11,
        // 1u << 12,
        // 1u << 13,
        // 1u << 14,
        // 1u << 15,
        // 1u << 16,
        // 1u << 17,
        // 1u << 18,
        // 1u << 19,
        // 1u << 20,
    }; */


    std::vector<uint32_t> POPULATION_SIZES = {};
    const uint32_t imin = 5; 
    const uint32_t imax = 22;
    for(uint32_t i = imin; i < imax; i++){
        POPULATION_SIZES.push_back((1 << i));
        if(i < imax -1){
            POPULATION_SIZES.push_back((1 << i) + (1 << (i-1)));
        }
    }

    // Define the models to execute, with a function pointer that builds and runs the model.
    std::map<std::string, std::function<void(const RunSimulationInputs, RunSimulationOutputs&)>> MODELS = {
        {std::string("circles_spatial3D"), run_circles_spatial3D},
        {std::string("circles_spatial3D_rtc"), run_circles_spatial3D_rtc},
        {std::string("circles_bruteforce"), run_circles_bruteforce},
        {std::string("circles_bruteforce_rtc"), run_circles_bruteforce_rtc},
    };

    // Make the output directory if required.
    printf("@todo - output dir via cli (and use it).\n");

    

    // @todo - error checking, overwrite checking, filename, etc. 
    std::FILE * fp_rowPerSimulation = std::fopen("row-per-simulation.csv", "w");
    if(fp_rowPerSimulation == nullptr){
        printf("could not open file? @todo\n");
        exit(EXIT_FAILURE);
    }
    
    // Output the header for the per run timing.
    fprintf(fp_rowPerSimulation, "GPU,release_mode,seatbelts,model,steps,agentCount,repeat,mean_messageCount,ms_rtc,ms_simulation,ms_init,ms_exit,ms_stepMean\n");
    
    // Write a row per step out to  a differnt file.
    std::FILE * fp_rowPerStepPerSimulation = std::fopen("row-per-step-per-simulation.csv", "w");
    if(fp_rowPerSimulation == nullptr){
        printf("could not open file? @todo\n");
        exit(EXIT_FAILURE);
    }

    // Output a header row. 
    fprintf(fp_rowPerStepPerSimulation, "GPU,release_mode,seatbelts,model,steps,agentCount,repeat,step,ms_step\n");
    
    // Get the name of the gpu. 
    std::string deviceName("unknown");
    cudaError_t status = cudaSuccess;
    int cudaDeviceCount = 0;
    status = cudaGetDeviceCount(&cudaDeviceCount);
    if ( cudaSuccess == status) {
        if ( cli.device < cudaDeviceCount ) { 
            cudaDeviceProp props;
            status = cudaGetDeviceProperties(&props, cli.device);
            if (cudaSuccess == status) {
                deviceName = std::string(props.name);
                cudaFree(0); // Make a context / slightly warm the decice?
            } else {
                printf("@todo handle error \n");
            }
        } else {
            printf("@todo - handle bad cuda device id\n");
        }
    } else {
        printf("@todo handle error \n");
    }
    
    // find the total number of sims to run.
    uint32_t totalSimulations = MODELS.size() * POPULATION_SIZES.size() * cli.repetitions;
    uint32_t counter = 0;

    // Iterate over the population sizes for that model
    for(auto const& agentCount : POPULATION_SIZES){     
        // Iterate the models/simulations to run.
        for(auto const& modelFunctionPair : MODELS){
            auto const& modelName = modelFunctionPair.first;
            auto const& modelFunction = modelFunctionPair.second;    
            // Repeat a number of times to get an average.
            for(uint32_t repeat = 0u; repeat < cli.repetitions; repeat++) { 
                // Progress. 
                printProgress(modelName, counter, totalSimulations, agentCount, repeat);
                // @todo - different seeds? same init differnt device? or both diff?
                const uint64_t seed = cli.seed;
                
                // Run the simulation, capturing values for output.
                const RunSimulationInputs runInputs = {modelName, seed, agentCount, cli.steps, cli.device};
                RunSimulationOutputs runOutputs = {};
                modelFunction(runInputs, runOutputs);

                // Add a row to the row per simulation csv file
                fprintf(fp_rowPerSimulation, "%s,%d,%d,%s,%u,%u,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", deviceName.c_str(), RELEASE_MODE, SEATBELTS_ON, modelName.c_str(), cli.steps, agentCount, repeat, runOutputs.mean_messageCount, runOutputs.ms_rtc, runOutputs.ms_simulation, runOutputs.ms_init, runOutputs.ms_exit, runOutputs.ms_stepMean); 
                
                // add a row to the row per step per simulation file for each step. This is wasting a lot of disk space... 
                for(uint32_t step = 0; step < runOutputs.ms_per_step->size(); step++){
                    auto& ms_step = runOutputs.ms_per_step->at(step);
                    fprintf(fp_rowPerStepPerSimulation, "%s,%d,%d,%s,%u,%u,%u,%u,%.3f\n", deviceName.c_str(), RELEASE_MODE, SEATBELTS_ON, modelName.c_str(), cli.steps, agentCount, repeat, step, ms_step); 
                }
                counter++;
            }
        }
    }
    std::fclose(fp_rowPerStepPerSimulation);
    fp_rowPerStepPerSimulation = nullptr;
    std::fclose(fp_rowPerSimulation);
    fp_rowPerSimulation = nullptr;
}

void print_cli_help(const int argc, const char ** argv ) {
    printf("usage: %s", argv[0]);
    printf(" [-r random]");
    printf(" [-s steps]");
    printf(" [-d device]");
    printf(" [--repetitions repetitions]");
    printf("\n");
    printf("optional args:\n");
    printf("  -r, --random <seed>             Seed for RNG\n");
    printf("  -s, --steps <steps>             Number of simulation iterations\n");
    printf("  -d, --device <device>           CUDA device to use\n");
    printf("      --repetitions <repetitions> The number of benchmark repetitions to perform\n");
}

custom_cli parse_custom_cli(const int argc, const char ** argv) {
    custom_cli values = {};
    // @todo - long term replace this with CLI library which will be included within F2
    for(int i = 0; i < argc; i++){
        std::string arg(argv[i]);
        if(arg.compare("-h") == 0 || arg.compare("--help") == 0){
            print_cli_help(argc, argv);
            exit(EXIT_FAILURE);
        } else if(arg.compare("-r") == 0 || arg.compare("--random-seed") == 0) {
            if(i + 1 < argc){
                try {
                    std::string v(argv[i+1]);
                    values.seed = std::stoull(v);
                } catch (const std::exception& e){
                    printf("Error: Invalid value for -r/--random.");
                    exit(EXIT_FAILURE);
                }
            } else {
                printf("Error: Missing value for -r/--random\n");
                print_cli_help(argc, argv);
                exit(EXIT_FAILURE);
            }
        } else if (arg.compare("-s") == 0 || arg.compare("--steps") == 0 ) {
            if(i + 1 < argc){
                try {
                    std::string v(argv[i+1]);
                    values.steps = std::stoul(v);
                } catch (const std::exception& e){
                    printf("Error: Invalid value for -s/--steps argument.");
                    exit(EXIT_FAILURE);
                }
            } else {
                printf("Error: Missing value for -s/--steps argument\n");
                print_cli_help(argc, argv);
                exit(EXIT_FAILURE);
            }
        } else if (arg.compare("-d") == 0 || arg.compare("--d") == 0 ) {
            if(i + 1 < argc){
                try {
                    std::string v(argv[i+1]);
                    values.device = std::stod(v);
                } catch (const std::exception& e){
                    printf("Error: Invalid value for -d/--device argument.");
                    exit(EXIT_FAILURE);
                }
            } else {
                printf("Error: Missing value for -d/--device argument\n");
                print_cli_help(argc, argv);
                exit(EXIT_FAILURE);
            }
        } else if(arg.compare("--repetitions") == 0) {
            if(i + 1 < argc){
                try {
                    std::string v(argv[i+1]);
                    values.repetitions = std::stoul(v);
                } catch (const std::exception& e){
                    printf("Error: Invalid value for --repetitions.");
                    exit(EXIT_FAILURE);
                }
            } else {
                printf("Error: Missing value for --repetitions\n");
                print_cli_help(argc, argv);
                exit(EXIT_FAILURE);
            }
        }
    }
    return values;
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
+ [ ] density experiment
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
