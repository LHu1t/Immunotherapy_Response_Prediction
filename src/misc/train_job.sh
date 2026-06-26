#!/bin/bash
#$ -N myjob2               # Job name
#$ -cwd                   # Run in current working directory
#$ -l h_rt=8:00:0         # Walltime
#$ -pe smp 12
#$ -l gpu=1
#$ -l mem=4G
#$ -l shm=80G
#$ -o train_job2.sh.o$JOB_ID
#$ -e train_job2.sh.e$JOB_ID

# --- Initialize modules ---
source /etc/profile.d/modules.sh
module load python/3.9.10
module load cuda/11.8.0   # if needed
module load openslide/3.4.1

# Activate Python environment
source ~/envs/immuno/bin/activate

# --- Go to directory where your script is ---
cd ~/Scratch

# Check Python Version
python3 --version

# --- Run Python script ---
export TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun --nproc_per_node=1 tiled_image_classifier.py
