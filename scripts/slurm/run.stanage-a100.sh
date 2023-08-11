#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1

# 12 CPU cores (1/4th of the node) and 1 GPUs worth of memory < 1/4th of the node)
#SBATCH --cpus-per-task=12
#SBATCH --mem=82G

# A100 module environment is active on the A100 nodes automatically now, load appropriate modules
module load GCC/11.3.0
module load CUDA/12.0.0

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
./bin/Release/circles-benchmark
