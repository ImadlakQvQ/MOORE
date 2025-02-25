import torch.optim as optim
from mushroom_rl.core import Logger
import torch.nn.functional as F
from tqdm import trange
import wandb
# Utils
import os
import pickle
from joblib import delayed, Parallel
import numpy as np
from moore.environments import MEMTMiniGrid as MiniGrid

task_space = np.load("task_embeddings_minigrid.npy", allow_pickle=True)

MT_EXP = {
    "MT7": {
        "MiniGrid-DoorKey-6x6-v0": task_space[0],
        "MiniGrid-DistShift1-v0": task_space[1],
        "MiniGrid-RedBlueDoors-6x6-v0": task_space[2],
        "MiniGrid-LavaGapS7-v0": task_space[3],
        "MiniGrid-MemoryS11-v0": task_space[4],
        "MiniGrid-SimpleCrossingS9N2-v0": task_space[5],
        "MiniGrid-MultiRoom-N2-S4-v0": task_space[6]
    },
    "MT3": {
        "MiniGrid-RedBlueDoors-6x6-v0": task_space[2],
        "MiniGrid-LavaGapS7-v0": task_space[3],
        "MiniGrid-MemoryS11-v0": task_space[4]
    },
    "MT5": {
        "MiniGrid-DoorKey-6x6-v0": task_space[0],
        "MiniGrid-DistShift1-v0": task_space[1],
        "MiniGrid-RedBlueDoors-6x6-v0": task_space[2],
        "MiniGrid-LavaGapS7-v0": task_space[3],
        "MiniGrid-MemoryS11-v0": task_space[4]
    }
}

def run_experiment(args, save_dir, exp_id = 0, seed = None):
    """
        Run the experiment,
        return the metrics
    """
    import matplotlib
    matplotlib.use('Agg') 

    np.random.seed()
    single_logger = Logger(f"exp_{exp_id if seed is None else seed}", results_dir=save_dir, log_console=True)
    save_dir = single_logger.path

    n_epochs = args.n_epochs
    n_steps = args.n_steps
    n_episodes_test = args.n_episodes_test

    # MDP
    env_names = MT_EXP[args.env_name]

    horizon = args.horizon
    gamma = args.gamma
    gamma_eval = args.gamma_eval
    # MT#中有几个 list中就有几个
    env_list = []
    descriptions = []
    for env_name_i, description_i in env_names.items():
        env_list.append(MiniGrid(env_name_i, horizon = horizon, gamma=gamma, render_mode=args.render_mode, description=description_i))
        descriptions.append(description_i)
    
    n_contexts = len(env_list)
    descriptions = np.array(descriptions)
    batch_size = args.batch_size
    train_frequency = args.train_frequency