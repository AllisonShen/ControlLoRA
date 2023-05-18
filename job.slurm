#!/bin/bash
#SBATCH --job-name=cl         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=xs6153@princeton.edu

module purge
module load anaconda3/2023.3
conda activate controllora

#cd /scratch/network/xs6153/ControlLoRA/
#python main.py --config configs/control_lora_waymo.yaml 
# Submitted batch job 1773419 (full training)

# cd /scratch/network/xs6153/ControlLoRA/
# python main.py --config configs/control_lora_multi_mix_encoder_only.yaml 
# # Submitted batch job 1773420 rank 128


#cd /scratch/network/xs6153/ControlLoRA/
#python main.py --config configs/control_lora_multi_mix_encoder_only_rank_1.yaml 
# Submitted batch job 1773421 rank 1 
# /scratch/network/xs6153/ControlLoRA/configs/control_lora_multi_mix_encoder_only_rank_1.yaml

# cd /scratch/network/xs6153/ControlLoRA/
# python main.py --config configs/control_lora_multi_mix_encoder_only_rank_2.yaml 
# job 1773422 rank 64

cd /scratch/network/xs6153/ControlLoRA/
python main.py --config configs/control_lora_multi_mix_encoder_only_rank_96.yaml 
# # job 1773423 rank 96


# cd /scratch/network/xs6153/ControlLoRA/examples
# python make_waymo_preprocessed.py