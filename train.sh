#!/bin/bash
#SBATCH  --output=/scratch_net/bravo/aungriah/job_subs/log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G
#SBATCH  --exclude=biwirender03,biwirender04,biwirender05,biwirender06,biwirender07,biwirender08,biwirender09,biwirender10,biwirender11,biwirender12

LOCALSCRATCH=/scratch_net/bravo/aungriah
source $LOCALSCRATCH/conda/etc/profile.d/conda.sh
conda activate LaneDet

cd /scratch_net/bravo/aungriah/AegisMultiTask
# Now, you should use -u to get unbuffered output and "$@" for any arguments
python train.py