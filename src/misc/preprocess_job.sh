#!/bin/bash
#$ -N preprocess_job2              # Job name
#$ -cwd                   # Run in current working directory
#$ -l h_rt=24:00:0         # Walltime
#$ -pe smp 8
#$ -l mem=4G
#$ -t 1-10
#$ -o preprocess_job.sh.o$JOB_ID
#$ -e preprocess_job.sh.e$JOB_ID

# --- Initialize modules ---
source /etc/profile.d/modules.sh
module load python/miniconda3/24.3.0-0
module load openslide/3.4.1

# Activate Python environment
source $UCL_CONDA_PATH/etc/profile.d/conda.sh

conda activate wsi_env_pyvips

# --- Go to directory where your script is ---
cd ~/Scratch

# Check Python Version
python3 --version

# Compute batch index (0-based for Python script)
BATCH_INDEX=$(($SGE_TASK_ID - 1))
BATCH_SIZE=52 # Change depending on number of files required per batch

echo "Running batch $BATCH_INDEX with batch size $BATCH_SIZE"

python3 image_preprocessing.py \
    --batch-index $BATCH_INDEX \
    --batch-size $BATCH_SIZE
