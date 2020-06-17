#!/bin/bash
# Name: paramsearch
# ID: 0

#SBATCH --array=0-11
#SBATCH --job-name=psearch
###SBATCH --qos=default
#SBATCH --mem=32gb
#SBATCH --account scavenger
#SBATCH --partition scavenger
#SBATCH --gres=gpu:1 ### gpu:p6000:1
#SBATCH --time=12:00:00
#SBATCH --output /cfarhomes/krusinga/causality/granger-causal-frames/logs/out_%a.out
###SBATCH --error err.txt

echo "Hello:"
echo ${SLURM_ARRAY_TASK_ID}

cd /cfarhomes/krusinga/causality/granger-causal-frames
#module load cuda
python traintest.py --line ${SLURM_ARRAY_TASK_ID}
#python run.py --stage 2 --id 72 --masterconfig /fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml  --pid ${SLURM_ARRAY_TASK_ID} --nprocs ${SLURM_ARRAY_TASK_COUNT}
