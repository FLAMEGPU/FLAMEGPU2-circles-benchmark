#!/bin/bash
# Will need to increase the time for full benchmark runs.... 
#SBATCH --time=8:00:00
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
# 1 GPU
#SBATCH --gres=gpu:1
# 1 CPU core - might want to increase this + memory to 1/4 of the node per GPU?
#SBATCH --ntasks=1

# Load modules (with matching gcc versions)
module use /usr/local/modulefiles/staging/eb/all/
module load CUDA/11.0.2-GCC-9.3.0

# CD into the script's directory, so the relative path exists when not exeucted as a batch job. 
cd "$(dirname "$0")"

# Set the location of the project root relative to this script
PROJECT_ROOT=../..

# navigate into the `build` directory.
cd $PROJECT_ROOT
cd build

# Set FLAMEGPU2_INC_DIR poiinting at the included dependency, relative to the build dir where execution is occuring.
# Long term this should not be requied
export FLAMEGPU2_INC_DIR=_deps/flamegpu2-src/include

# Output some GPU information into the Log
nvidia-smi

# Run the executable.
./bin/Release/circles-benchmarking
