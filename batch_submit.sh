#!/bin/bash
# Name: BeamNG train

###SBATCH --array=0-11
#SBATCH --job-name=trainmd
#SBATCH --qos=high
#SBATCH --mem=32gb
#SBATCH -c 3  ### Use this to ask for a certain number of cores? (cpuspertask)
#SBATCH -n 5  ### number of multiprocessing tasks? (ntasks) 
###SBATCH --account scavenger
###SBATCH --partition scavenger
#SBATCH --gres=gpu:2 ### gpu:p6000:1
#SBATCH --time=36:00:00
#SBATCH --output /cfarhomes/krusinga/storage/research/causality/granger-causal-frames/logs/out_simple_dataset.txt 
###out_%a.out
###SBATCH --error err.txt

#echo "Hello:"
#echo ${SLURM_ARRAY_TASK_ID}
echo "Hello from: "
hostname

if [ -d "/scratch0/krusinga" ]
then
    echo "/scratch0/krusinga exists"
else
    echo "Making directory"
    mkdir /scratch0/krusinga
fi

if [ -d "/scratch0/krusinga/simple_dataset" ]
then
    echo "Dataset exists on scratch0"
else
    echo "Copying dataset"
    time cp -r /vulcanscratch/ywen/car_crash/simple_dataset/ /scratch0/krusinga/
fi

# 4 data workers is optimal
cd /cfarhomes/krusinga/storage/research/causality/granger-causal-frames
time python -u decord_traintest.py --train --dataset beamng_simple --batch_size 128 --num_data_workers 4 --n_epochs 50 --model_name train_on_simple_dataset
#module load cuda
#python -u traintest.py --line ${SLURM_ARRAY_TASK_ID}
#python run.py --stage 2 --id 72 --masterconfig /fs/vulcan-scratch/krusinga/projects/ganProbability/master.yaml  --pid ${SLURM_ARRAY_TASK_ID} --nprocs ${SLURM_ARRAY_TASK_COUNT}
