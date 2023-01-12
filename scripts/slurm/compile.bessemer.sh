#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=compile.bessemer.sh
# 8 CPU cores + enough memory (< 8/40ths of the node memory)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load modules for compilation
module use /usr/local/modulefiles/staging/eb/all/
module load CUDA/11.0.2-GCC-9.3.0

# Load cmake via pip into a local venv
module load Anaconda3/5.3.0

conda create -yn fgpu2-circles-benchmark
source activate fgpu2-circles-benchmark
conda install -y cmake=3.18


# Set the location of the project root relative to this script
PROJECT_ROOT=../..

# navigate into the root directory.
cd $PROJECT_ROOT

# Make the build directory.
mkdir -p build && cd build

# Configure cmake.
cmake .. -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release -DFLAMEGPU_SEATBELTS=OFF -DFLAMEGPU_SHARE_USAGE_STATISTICS=OFF

# Compile the code with make for GPUs in Bessener (SM_70)
make -j `nproc`

