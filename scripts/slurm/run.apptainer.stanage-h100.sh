#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h100:1

# 24 CPU cores (1/2 of the node) and 1 GPUs worth of memory < 1/2th of the node)
#SBATCH --cpus-per-task=24
#SBATCH --mem=82G

# Set the location of the project root relative to this script
# This version will only work within a slurm job, submit from this directory.
PROJECT_ROOT="${SLURM_SUBMIT_DIR}/../.."
# Specify the location of the apptainer image path, which has been manually copied to bessemer
APPTAINER_IMAGE_PATH=${PROJECT_ROOT}/flamegpu2-circles-benchmark-11.8.sif

# navigate into a working directory directory.
cd $PROJECT_ROOT
mkdir -p apptainer-workdir-h100 && cd apptainer-workdir-h100

# Output the node this was executed on
echo "HOSTNAME=${HOSTNAME}"

# Output some GPU information into the Log
nvidia-smi

# Run the executable.
apptainer exec --nv --cleanenv ${APPTAINER_IMAGE_PATH} /opt/FLAMEGPU2-circles-benchmark/build/bin/Release/circles-benchmark
