#!/bin/bash
#SBATCH --time=00:30:00

# H100 NVL nodes have intel CPUs, so compilation can occur on a regular CPU node - no need for a gpu for this.
# (but not a login node, as the libcuda linkage will fail from the login node, IIRC)

# 64 cores and 251GB of memory per general stanage CPU node. Use ~1/8th for a compromise between compile time and node usage
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G

# intel / icelake modules will be available
module load GCC/12.3.0
module load CUDA/12.4.0
module load CMake/3.26.3-GCCcore-12.3.0

# Set the location of the project root relative to this script
PROJECT_ROOT=../..

# navigate into the root directory.
cd $PROJECT_ROOT

# Configure cmake for H100 NVL GPUs (SM_90) in Release without seatbelts
cmake -S . -B build-h100-nvl -DCMAKE_CUDA_ARCHITECTURES="90" -DCMAKE_BUILD_TYPE=Release -DFLAMEGPU_SEATBELTS=OFF -DFLAMEGPU_SHARE_USAGE_STATISTICS=OFF

# Compile the code using all available processors.
cmake --build build-h100-nvl -j `nproc`

