#pragma once

#include "common.cuh"

void printProgress(
    const std::string modelName, 
    const uint32_t count, 
    const uint32_t total, 
    const uint32_t agentCount, 
    const float envWidth, 
    const float commRadius,
    const uint32_t repeat
){
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char buf[sizeof "2011-10-08T07:07:09Z"];
    std::strftime(buf, sizeof buf, "%FT%TZ", std::gmtime(&now));
    float envVolume = envWidth * envWidth * envWidth;
    float agentDensity = agentCount / envVolume;
    fprintf(stdout, "%s: %u/%u: %s %u %f %f %f %f %u\n", buf, count + 1, total, modelName.c_str(), agentCount, envWidth, envVolume, agentDensity, commRadius, repeat);
}

std::string getGPUName(int device){
    // Get the name of the gpu. 
    std::string deviceName("unknown");
    cudaError_t status = cudaSuccess;
    int cudaDeviceCount = 0;
    status = cudaGetDeviceCount(&cudaDeviceCount);
    if ( cudaSuccess == status) {
        if ( device < cudaDeviceCount ) { 
            cudaDeviceProp props;
            status = cudaGetDeviceProperties(&props, device);
            if (cudaSuccess == status) {
                deviceName = std::string(props.name);
                cudaFree(0); // Make a context / slightly warm the decice?
            } else {
                fprintf(stderr, "Fatal Error: could not get device name.\n");
                exit(EXIT_FAILURE);
            }
        } else {
            fprintf(stderr, "Fatal Error: device %d does not exist\n", device);
            exit(EXIT_FAILURE);
        }
    } else {
        fprintf(stderr, "Fatal Error: Could not detect the number of CUDA devices\n");
        exit(EXIT_FAILURE);
    }
    return deviceName;
}

constexpr bool isReleaseMode() {
    #if defined(NDEBUG) || defined(_NDEBUG)
        return true;
    #else 
        return false;
    #endif

}
constexpr bool isSeatbeltsON() {
    #if defined(SEATBELTS) && !SEATBELTS
        return false;
    #else 
        return true;
    #endif
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
    printf("      --dry                       Dry run, don't actually run the sims.\n");
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
        } else if(arg.compare("--dry") == 0) {
            values.dry = true;
        }
    }
    return values;
}