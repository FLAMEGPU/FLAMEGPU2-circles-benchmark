#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
# 1 GPU
#SBATCH --gres=gpu:1
# 1 CPU core
#SBATCH --cpus-per-task=1
# 1 GPU's worth of host memory
#SBATCH --mem=32G

# Set the location of the project root relative to this script
# This version will only work within a slurm job, submit from this directory.
PROJECT_ROOT="${SLURM_SUBMIT_DIR}/../.."
# Specify the location of the apptainer image path, which has been manually copied to bessemer
APPTAINER_IMAGE_PATH=${PROJECT_ROOT}/flamegpu2-circles-benchmark-11.8.sif

# navigate into a working directory directory.
cd $PROJECT_ROOT
mkdir -p apptainer-workdir && cd apptainer-workdir

# Output the node this was executed on
echo "HOSTNAME=${HOSTNAME}"

# Output some GPU information into the Log
nvidia-smi

# Run the executable.
apptainer exec --nv --cleanenv ${APPTAINER_IMAGE_PATH} /opt/FLAMEGPU2-circles-benchmark/build/bin/Release/circles-benchmark
