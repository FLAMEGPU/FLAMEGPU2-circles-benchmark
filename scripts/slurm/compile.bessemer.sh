#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --ntasks=4
#SBATCH --job-name=compile.bessemer.sh

# Load modules for compilation
module use /usr/local/modulefiles/staging/eb/all/
module load CUDA/11.0.2-GCC-9.3.0

# Load cmake with matching glibc version. Alternatively use a cmake installed via pip into a local venv.
module load CMake/3.16.4-GCCcore-9.3.0

# Load matching python too 
# module load Python/3.7.4-GCCcore-8.3.0

# Make tmpdir incase it doesn't exist? This should not be neccesary...
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

