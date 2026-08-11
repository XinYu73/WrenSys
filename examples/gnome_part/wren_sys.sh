#!/bin/bash
#SBATCH    --job-name=caly
#SBATCH    --output=log.out.%j
#SBATCH    --error=log.err.%j
#SBATCH    --partition=cpu
#SBATCH    --nodes=1
#SBATCH    --ntasks=8
#SBATCH    --ntasks-per-node=8
# #SBATCH --gres=gpu:1
export OMP_NUM_THREADS=1

python ../../wren_sys/predict.py -m ../../model_dir -p input_wren_sys.csv -n  output_wren_sys.csv

