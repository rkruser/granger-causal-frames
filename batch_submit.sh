#!/bin/bash
# Name: BeamNG train

###SBATCH --array=0-11
#SBATCH --job-name=trainbm
#SBATCH --qos=high
#SBATCH --mem=32gb
###SBATCH -c 10  ### Use this to ask for a certain number of cores? (cpuspertask)
#SBATCH -n 6  ### number of multiprocessing tasks? (ntasks) 
###SBATCH --account scavenger
###SBATCH --partition scavenger
#SBATCH --gres=gpu:4 ### gpu:p6000:1
#SBATCH --time=36:00:00
#SBATCH --output /cfarhomes/krusinga/storage/research/causality/granger-causal-frames/logs/out.txt 
#out_%a.out
###SBATCH --error err.txt

#echo "Hello:"
#echo ${SLURM_ARRAY_TASK_ID}
echo "Hello from: "
hostname


# 4 data workers is optimal
cd /cfarhomes/krusinga/storage/research/causality/granger-causal-frames
time python -u decord_traintest.py --batch_size 128 --num_data_workers 4 --n_epochs 2
#module load cuda
#python -u traintest.py --line ${SLURM_ARRAY_TASK_ID}
#python run.py --stage 2 --id 72 --masterconfig /fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml  --pid ${SLURM_ARRAY_TASK_ID} --nprocs ${SLURM_ARRAY_TASK_COUNT}
