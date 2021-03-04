#include <algorithm>
#include <cstdio>
#include <chrono>
#include <ctime>

#include "flamegpu/flame_api.h"

// Include the bruteforce implementation

#include "common.cuh"

void run_circles_bruteforce(const simMethodParametrs params, simulationTiming &times);
void run_circles_bruteforce_rtc(const simMethodParametrs params, simulationTiming &times);
void run_circles_spatial3D(const simMethodParametrs params, simulationTiming &times);
void run_circles_spatial3D_rtc(const simMethodParametrs params, simulationTiming &times);



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
    

__global__ void warmup(float * data, uint32_t elements, uint32_t reps) {

    for(uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < elements; idx += blockDim.x * gridDim.x){
        data[idx] = idx;
        for(uint32_t rep = 0; rep < reps; rep++){
            data[idx] = data[idx] + ((threadIdx.x * rep) % blockDim.x);
        }
    }

}

// Do some arbitrary work on the device to get it  itno a different power state.
void cudaWarmup() {
    // @todo - cuda check.
    const uint32_t elements = 2 << 20; 
    const size_t bytes = elements * sizeof(float);
    const uint32_t reps = 2 << 8; 

    float * d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemset(d_data, 0, bytes);

    int blockSize = 256;
    int gridSize = (elements + blockSize - 1) / blockSize;

    warmup<<<gridSize, blockSize, 0, 0>>>(d_data, elements, reps);
    
    cudaFree(d_data);
    d_data = nullptr;
    



}

void printProgress(const std::string modelName, const uint32_t count, const uint32_t total, const uint32_t agentCount, const uint32_t repeat){
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char buf[sizeof "2011-10-08T07:07:09Z"];
    std::strftime(buf, sizeof buf, "%FT%TZ", std::gmtime(&now));
    fprintf(stdout, "%s: %u/%u: %s %u %u\n", buf, count + 1, total, modelName.c_str(), agentCount, repeat);
}


// @todo deal with what happens if a simulation throws an exception?
int main(int argc, const char ** argv) {
    // Custom arg parsing, to prevent the current F2 arg parsing from occuring. 
    // @todo - improve arg parsing within F2. 
    custom_cli cli = parse_custom_cli(argc, argv);
    
    // Define the benchmark parameters. I.e. initial environment width, max width, method of interpolation etc. 
    // @todo
    std::vector<uint32_t> POPULATION_SIZES = {
        2u << 8,
        2u << 9,
        2u << 10,
        2u << 11,
        2u << 12,
        2u << 13,
        2u << 14,
        2u << 15,
        2u << 16,
        2u << 17,
        2u << 18,
        2u << 19,
        2u << 20,
    };

    // Define the models to execute, with a function pointer that builds and runs the model.
    std::map<std::string, std::function<void(const simMethodParametrs, simulationTiming&)>> MODELS = {
        // {std::string("circles_spatial3D"), run_circles_spatial3D},
        {std::string("circles_spatial3D_rtc"), run_circles_spatial3D_rtc},
        // {std::string("circles_bruteforce"), run_circles_bruteforce},
        {std::string("circles_bruteforce_rtc"), run_circles_bruteforce_rtc},
    };


    // Make the output directory if required.
    printf("@todo - output dir via cli (and use it).\n");

    
    // Write out the parameters used to generate the model? This might not be required as it can be figured out from the actual data...

    // std::ofstream paramsFile("params.csv");
    // @todo - output directory.
    // @todo - add cuda device name. 
    // if (paramsFile.is_open()) {
    //     paramsFile << initialPopSize << "," << finalPopSize << "," << popSizeIncrement << std::endl;
    //     paramsFile << initialNumSpecies << "," << finalNumSpecies << "," << numSpeciesIncrement << std::endl;
    // }

    // @todo - error checking, overwrite checking, filename, etc. 
    std::FILE * fp_rowPerSimulation = std::fopen("row-per-simulation.csv", "w");
    if(fp_rowPerSimulation == nullptr){
        printf("could not open file? @todo\n");
        exit(EXIT_FAILURE);
    }
    
    
    // Output the header for the per run timing.
    fprintf(fp_rowPerSimulation, "GPU, release_mode, seatbelts, model, steps, agentCount, repeat, ms_rtc, ms_simulation, ms_init, ms_exit, ms_stepMean\n");
    

    
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
                cudaWarmup();
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

    // Iterate over population size first. This then allows for early exit when sims become too slow? Alternatively do the fastest simulations first, but this would require changing the map to be ordered.
    // Iterate the models/simulations to run.
    for(auto const& modelFunctionPair : MODELS){
        auto const& modelName = modelFunctionPair.first;
        auto const& modelFunction = modelFunctionPair.second;    
        
        // @todo - do (some) aggregation of timers? e.g. output 1 csv for a given sim-popsize combo, with step times for each sim and mean step times? for 
        
        // Iterate over the population sizes for that model
        for(auto const& agentCount : POPULATION_SIZES){            
            // Repeat a number of times to get an average.
            for(uint32_t repeat = 0u; repeat < cli.repetitions; repeat++) { 

                // Progress. 
                printProgress(modelName, counter, totalSimulations, agentCount, repeat);

                // @todo - better filenaming
                std::string simulationStepsRawFilename = std::string("simulation-steps-raw-") + std::to_string(counter) + std::string(".csv");
                // @todo better file handling, directory etc.
                std::FILE * fp_simulationStepsRaw = std::fopen(simulationStepsRawFilename.c_str(), "w");
                if(fp_simulationStepsRaw == nullptr){
                    printf("could not open file? @todo\n");
                    exit(EXIT_FAILURE);
                }
                // @todo - Do i need to use a different seed for each repetition? Probably should do both...
                const uint64_t seed = cli.seed;
                
                // Call the fn to run this simulation witht his pop for this rep. 
                // @todo get timing info to save for alter.
                simulationTiming t = {};
                modelFunction({modelName, seed, agentCount, cli.steps, cli.device}, t);


                // Output the individual runs times
                fprintf(fp_rowPerSimulation, "\"%s\", %d, %d, \"%s\", %u, %u, %u, %.3f, %.3f, %.3f, %.3f, %.3f\n", deviceName.c_str(), RELEASE_MODE, SEATBELTS_ON, modelName.c_str(), cli.steps, agentCount, repeat, t.ms_rtc, t.ms_simulation, t.ms_init, t.ms_exit, t.ms_stepMean); 
                
                // Output a csv containing the per step time 
                fprintf(fp_simulationStepsRaw, "GPU, release_mode, seatbelts, model, steps, agentCount, repeat, step, ms_step\n");
                for(uint32_t step = 0; step < t.ms_per_step->size(); step++){
                    auto& ms_step = t.ms_per_step->at(step);
                    fprintf(fp_simulationStepsRaw, "\"%s\", %d, %d, \"%s\", %u, %u, %u, %u, %.3f\n", deviceName.c_str(), RELEASE_MODE, SEATBELTS_ON, modelName.c_str(), cli.steps, agentCount, repeat, step, ms_step); 
                }

                std::fclose(fp_simulationStepsRaw);
                fp_simulationStepsRaw = nullptr;

                counter++;
            }
        }
    }
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
+ [ ] Change the order of loops so pops are first, toa llow early exit.
+ [ ] RTC bruteforce
+ [ ] Move pop gen to init fn? so it gets timed.
+ [ ] RTC Spatial
+ [ ] Better disk io? 
    + [ ] Combine the per-step time files somehow? Maybe even just cat them into a very tall, repettitive csv?
+ [ ] Better error checking. 
+ [ ] Plotting (.py)
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
+ [ ] Have each agent store the message count it read. Exit fn that reduces theses and adds min/max/mean to the output data and CSVs. This might be useful
*/
