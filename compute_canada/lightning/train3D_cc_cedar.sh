#!/bin/bash

#SBATCH --account=def-laporte1
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --time=1-05:00     # DD-HH:MM:SS
#SBATCH --output=%N-%j.out
#SBATCH --mail-user=muni-venkata-naga-karthik.enamundram.1@ens.etsmtl.ca
#SBATCH --mail-type=ALL

module load python/3.6 cuda cudnn
source ~/projects/def-laporte1/karthik7/venvDL/bin/activate

python train3D_cc.py