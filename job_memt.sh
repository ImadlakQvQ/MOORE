#! /bin/bash
#SBATCH --account=def-bboulet
#SBATCH --cpus-per-task=6
#SBATCH --mem=80gb
#SBATCH --output=log_cc/%j.out
#SBATCH --time=0-12:00

module load StdEnv/2020 cmake gcc opencv rust python/3.8.10
source ~/.venv/moore_minigrid/bin/activate
cd project/def-bboulet/imadlak/program/MOORE

export WANDB_MODE='offline'

ENV_NAME=$1
N_EXPERTS=$2
# python run_minigrid_ppo_memt.py  --n_exp 15 --name MEMT\
#                             --env_name ${ENV_NAME} --exp_name ppo_mt_moore_multihead_${N_EXPERTS}e \
#                             --n_epochs 100 --n_steps 2000  --n_episodes_test 16 --train_frequency 2000 --lr_actor 1e-3 --lr_critic 1e-3 \
#                             --critic_network MiniGridPPOMEMTNetwork --critic_n_features 128 --orthogonal --n_experts ${N_EXPERTS} \
#                             --actor_network MiniGridPPOMEMTNetwork --actor_n_features 128 \
#                             --batch_size 256 --gamma 0.99 --wandb

# python run_minigrid_ppo_mt.py  --n_exp 15 --name MEMT\
#                             --env_name ${ENV_NAME} --exp_name ppo_mt_moore_multihead_${N_EXPERTS}e \
#                             --n_epochs 100 --n_steps 2000  --n_episodes_test 16 --train_frequency 2000 --lr_actor 1e-3 --lr_critic 1e-3 \
#                             --critic_network MiniGridPPOMixtureMHNetwork --critic_n_features 128 --orthogonal --n_experts ${N_EXPERTS} \
#                             --actor_network MiniGridPPOMixtureMHNetwork --actor_n_features 128 \
#                             --batch_size 256 --gamma 0.99 --wandb


# python run_minigrid_ppo_mt.py  --n_exp 15 --name MEMT\
#                             --env_name ${ENV_NAME} --exp_name ppo_mt_moore_multihead_${N_EXPERTS}e \
#                             --n_epochs 100 --n_steps 2000  --n_episodes_test 16 --train_frequency 2000 --lr_actor 1e-3 --lr_critic 1e-3 \
#                             --critic_network MiniGridPPOMixtureMHNetwork --critic_n_features 128 --orthogonal --n_experts ${N_EXPERTS} \
#                             --actor_network MiniGridPPOMixtureMHNetwork --actor_n_features 128 \
#                             --batch_size 256 --gamma 0.99 --wandb 
