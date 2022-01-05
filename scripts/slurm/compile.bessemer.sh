#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --ntasks=4
#SBATCH --job-name=compile.bessemer.sh

# Load modules for compilation
module use /usr/local/modulefiles/staging/eb/all/
module load CUDA/11.0.2-GCC-9.3.0

# Load cmake via pip into a local venv
module load Anaconda3/5.3.0
conda create -n fgpu2-circles-benchmark
source activate fgpu2-circles-benchmark
pip install --user cmake==3.18.0
export PATH=$PATH:~/.local/bin

# Make sure temporary directory exists, used for RTC cache
mkdir -p $TMPDIR

# Set the location of the project root relative to this script
PROJECT_ROOT=../..

# navigate into the root directory.
cd $PROJECT_ROOT

# Make the build directory.
mkdir -p build && cd build

# Configure cmake.
cmake .. -DCUDA_ARCH=70 -DCMAKE_BUILD_TYPE=Release -DSEATBELTS=OFF 

# Compile the code with make for GPUs in Bessener (SM_70)
make -j `nproc`

