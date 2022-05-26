#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=gpu-a100-tmp
#SBATCH --qos=gpu
# 1 GPU
#SBATCH --gres=gpu:1
# 1 CPU core
#SBATCH --cpus-per-task=1
# 1 GPU's worth of host memory
#SBATCH --mem=80G

# Use A100 specific module environment
module unuse /usr/local/modulefiles/live/eb/all
module unuse /usr/local/modulefiles/live/noeb
module use /usr/local/modulefiles/staging/eb-znver3/all/

# Load modules from the A100 specific environment
module load GCC/11.2.0
module load CUDA/11.4.1

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

# Output the node this was executed on
echo "HOSTNAME=${HOSTNAME}"

# Output some GPU information into the Log
nvidia-smi

# Run the executable.
./bin/Release/circles-benchmarking
