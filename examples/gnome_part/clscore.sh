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


python ../../Synthesizability-PU-CGCNN/generate_crystal_graph.py --cifs ./cif_files --n 12 --r 8 --f ./saved_crystal_graph
python ../../Synthesizability-PU-CGCNN/predict_PU_learning.py --bag 100 --graph ./saved_crystal_graph --cifs ./cif_files --modeldir ../../Synthesizability-PU-CGCNN/train_models_lr_5/

