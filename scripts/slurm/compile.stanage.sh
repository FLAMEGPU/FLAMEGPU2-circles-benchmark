#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=compile.stanage.sh

# Must compile for the GPU nodes on the GPU nodes for the correct CPU arch. The specific GPU doesn't currently matter.
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

# 12 CPU cores (1/4th of an a100 node) and 1 GPUs worth of memory < 1/4th of the node)
#SBATCH --cpus-per-task=12
#SBATCH --mem=82G

# A100 module environment is active on the A100 nodes automatically now, load appropriate modules
module load GCC/11.3.0
module load CUDA/12.0.0
module load CMake/3.24.3-GCCcore-11.3.0

# Set the location of the project root relative to this script
PROJECT_ROOT=../..

# navigate into the root directory.
cd $PROJECT_ROOT

# Make the build directory.
mkdir -p build && cd build

# Configure cmake for A100 and H100 GPUs (SM_80;SM_90) in Release without seatbelts
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;90" -DCMAKE_BUILD_TYPE=Release -DFLAMEGPU_SEATBELTS=OFF -DFLAMEGPU_SHARE_USAGE_STATISTICS=OFF

# Compile the code using all available processors.
cmake --build . -j `nproc`

