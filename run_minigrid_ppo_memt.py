# mushroomrl
from mushroom_rl.core import Logger
from mushroom_rl.environments import *
from mushroom_rl.approximators.parametric.torch_approximator import *
from mushroom_rl.utils.parameters import Parameter
# deeplearning
import torch.optim as optim
import torch.nn.functional as F
from moore.core import MEMTCore
from moore.algorithms.actor_critic import MEMTPPO
from moore.policy import MEMTBoltzmannTorchPolicy
from moore.environments import MEMTMiniGrid as MiniGrid
from moore.utils.argparser_memt import argparser
from moore.utils.dataset import get_stats
import moore.utils.networks_ppo as Network
# visulization
from tqdm import trange
import wandb
# Utils
import os
import pickle
from joblib import delayed, Parallel
import numpy as np

# TODO 将这些描述转换到一个统一的任务空间
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

    if args.debug:
        args.batch_size = 8
        args.n_epochs = 2
        args.n_steps = 150
        args.n_steps_test = 100
        args.n_episodes_test = 1
        args.wandb = False

    single_logger = Logger(f"exp_{exp_id if seed is None else seed}", results_dir=save_dir, log_console=True)
    save_dir = single_logger.path

    # Environment
    env_names = MT_EXP[args.env_name]
    env_list = []
    descriptions = []
    for env_name_i, description_i in env_names.items():
        env_list.append(MiniGrid(env_name_i, horizon = args.horizon, gamma=args.gamma, render_mode=args.render_mode, description=description_i))
        descriptions.append(description_i)
    num_envs = len(env_list)
    descriptions = np.array(descriptions)

    # Actor
    actor_params = dict(beta=args.beta,
                        n_features=args.n_features,
                        n_contexts=num_envs,
                        orthogonal=args.orthogonal,
                        learning_rate = args.lr_actor,
                        n_experts=args.n_experts,
                        n_action_experts=args.n_action_experts,
                        use_cuda=args.use_cuda,
                        descriptions=descriptions,
                        coeff_experts=args.coeff_experts
                        )

    policy = MEMTBoltzmannTorchPolicy(
                                    getattr(Network, "MiniGridPPOMEMTNetwork"),
                                    env_list[0].info.observation_space.shape,
                                    (env_list[0].info.action_space.n,),
                                    **actor_params
                                    )
    
    actor_optimizer = {'class': optim.Adam, 'params': {'lr': args.lr_actor, 'betas': (0.9, 0.999)}}#
    
    # TODO 
    critic_params = dict(
                        network=getattr(Network, "MiniGridPPOMEMTNetwork"),
                        optimizer={
                            'class': optim.Adam, 
                            'params': {'lr': args.lr_critic, 'betas': (0.9, 0.999)}},
                        loss=F.mse_loss,
                        n_features=args.n_features,
                        n_contexts=num_envs,
                        orthogonal=args.orthogonal,
                        learning_rate = args.lr_critic,
                        n_experts=args.n_experts,
                        input_shape = env_list[0].info.observation_space.shape,
                        output_shape=(1,),
                        descriptions=descriptions
                        )
    
    # alg
    eps = 0.2
    ent_coeff = 0.01
    lam=.95
    alg_params = dict(
            n_epochs_policy=8,
            batch_size=args.batch_size*args.n_experts,
            eps_ppo=eps,
            ent_coeff=ent_coeff,
            lam=lam,
            actor_optimizer = actor_optimizer,
            critic_params = critic_params,
            critic_fit_params=None,
            descriptions=descriptions)

    if args.wandb:
        wandb.init(name = f"{args.exp_name}_"+str(exp_id if seed is None else seed), project = args.name, group = f"{args.exp_name}", mode="offline", job_type=args.exp_name, config=vars(args))

    # Agent
    agent = MEMTPPO(env_list[0].info, policy, n_contexts=num_envs, **alg_params)

    single_logger.info(agent._V.model.network)
    single_logger.info(agent.policy._logits.model.network)

    os.makedirs(save_dir, exist_ok=True)

    # 定义agent和环境
    core = MEMTCore(agent, env_list)

    # # RUN
    # metrics
    metrics = {mdp_i.env_name: {} for mdp_i in env_list}
    for key, value in metrics.items():
        value.update({"MinReturn": []})
        value.update({"MaxReturn": []})
        value.update({"AverageReturn": []})
        value.update({"AverageDiscountedReturn": []})
    metrics.update({"all_minigrid": {"AverageReturn": [], "AverageDiscountedReturn": []}})

    current_all_average_return = 0.0
    current_all_average_discounted_return = 0.0
    # ---------------------------------------- 评估初始策略 -----------------------------------
    # 对环境列表中的每一个环境都进行评估，并保存评估结果
    for c, mdp_c in enumerate(env_list):
        # random policy evaluation
        core.eval = True
        agent.policy.set_beta(args.beta)
        core.current_idx = c
        description = mdp_c.description
        
        dataset = core.evaluate(n_episodes=args.n_episodes_test, render=args.render_eval, quiet=True)
        min_J, max_J, mean_J, mean_discounted_J, _ = get_stats(dataset, args.gamma, args.gamma_eval)
        metrics[mdp_c.env_name]["MinReturn"].append(min_J)
        metrics[mdp_c.env_name]["MaxReturn"].append(max_J)
        metrics[mdp_c.env_name]["AverageReturn"].append(mean_J)
        metrics[mdp_c.env_name]["AverageDiscountedReturn"].append(mean_discounted_J)

        current_all_average_return+=mean_J
        current_all_average_discounted_return+=mean_discounted_J

        # single_logger.epoch_info(0,
        #                     EnvName = mdp_c.env_name,
        #                     MinReturn=min_J,
        #                     MaxReturn = max_J,
        #                     AverageReturn = mean_J,
        #                     AverageDiscountedReturn = mean_discounted_J,)
        if args.wandb:
            wandb.log({f'{mdp_c.env_name}/MinReturn': min_J,
                        f'{mdp_c.env_name}/MaxReturn': max_J,
                        f'{mdp_c.env_name}/AverageReturn':mean_J,
                        f'{mdp_c.env_name}/AverageDiscountedReturn':mean_discounted_J,
                        }, step = 0, commit=False)
    
    metrics["all_minigrid"]["AverageReturn"].append(current_all_average_return/ num_envs)
    metrics["all_minigrid"]["AverageDiscountedReturn"].append(current_all_average_discounted_return/ num_envs)

    if args.wandb:
        wandb.log({"all_minigrid/AverageReturn": current_all_average_return/ num_envs, 
                   "all_minigrid/AverageDiscountedReturn": current_all_average_discounted_return/ num_envs}, step = 0, commit=True)
    
    # ---------------------------------------- 主程序 -----------------------------------
    # 默认100个epoch
    for n in trange(args.n_epochs, desc=f"{args.exp_name}"):
        core.eval = False
        # beta是一个scale factor，乘在actor的输出上，还不知道是用来干啥的
        agent.policy.set_beta(args.beta)
        # quiet 不会出现这么多bar；n_steps epoch内的步数；
        core.learn(n_steps=args.n_steps, n_steps_per_fit=args.train_frequency, render=args.render_train, quiet=True)

        current_all_average_return = 0.0
        current_all_average_discounted_return = 0.0
        # --------------------------------- 训练完了之后，再次评估 ----------------------------------------
        for c, mdp_c in enumerate(env_list):
            core.eval = True
            agent.policy.set_beta(args.beta)
            core.current_idx = c
            dataset = core.evaluate(n_episodes=args.n_episodes_test, render=(args.render_eval if n%args.render_interval == 0 and exp_id == 0 else False), quiet=True)
            min_J, max_J, mean_J, mean_discounted_J, _ = get_stats(dataset, args.gamma, args.gamma_eval)
            metrics[mdp_c.env_name]["MinReturn"].append(min_J)
            metrics[mdp_c.env_name]["MaxReturn"].append(max_J)
            metrics[mdp_c.env_name]["AverageReturn"].append(mean_J)
            metrics[mdp_c.env_name]["AverageDiscountedReturn"].append(mean_discounted_J)

            current_all_average_return+=mean_J
            current_all_average_discounted_return+=mean_discounted_J

            if args.wandb:
                wandb.log({ f'{mdp_c.env_name}/MinReturn': min_J,
                            f'{mdp_c.env_name}/MaxReturn': max_J,
                            f'{mdp_c.env_name}/AverageReturn':mean_J,
                            f'{mdp_c.env_name}/AverageDiscountedReturn':mean_discounted_J,
                            }, step = n+1, commit=False)  

        metrics["all_minigrid"]["AverageReturn"].append(current_all_average_return/ num_envs)
        metrics["all_minigrid"]["AverageDiscountedReturn"].append(current_all_average_discounted_return/ num_envs)

        if args.wandb:
            wandb.log({"all_minigrid/AverageReturn": current_all_average_return/ num_envs, 
                       "all_minigrid/AverageDiscountedReturn": current_all_average_discounted_return/ num_envs}, step = n+1, commit=True)

    # ---------------------------------------- 保存参数  ----------------------------------------
    if args.wandb:
        wandb.finish()

    if args.save:
        single_logger.info("Parameters saved")
        agent.save_params(save_dir)

    return metrics

if __name__ == '__main__':
    # arguments
    args = argparser()

    if args.seed is not None:
        assert len(args.seed) == args.n_exp

    # logging               args.env_name=MT3
    results_dir = os.path.join(args.results_dir, "minigrid", "MT", args.env_name)
    # TODO 确定实验名字 
    args.exp_name = f"{args.exp_name}_{args.lr_actor}_{args.coeff_experts}"
    logger = Logger(args.exp_name, results_dir=results_dir, log_console=True, use_timestamp=args.use_timestamp)
    logger.strong_line()
    logger.info('Experiment Algorithm: ' + MEMTPPO.__name__)
    logger.info('Experiment Environment: ' + args.env_name)
    logger.info('Experiment Type: ' + "MEMT_test")
    logger.info("Experiment Name: " + args.exp_name)
    logger.info("Experiment parameters: " + str(vars(args)))
    save_dir = logger.path

    with open(os.path.join(save_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    if args.debug:
        args.exp = 1
        args.seed = None

    if args.seed is not None:
        out = Parallel(n_jobs=-1)(delayed(run_experiment)(args, save_dir, i, s)
                              for i, s in zip(range(args.n_exp), args.seed))
    elif args.n_exp > 1:
        out = Parallel(n_jobs=-1)(delayed(run_experiment)(args, save_dir, i)
                              for i in range(args.n_exp))
    else:
        out = run_experiment(args, save_dir)
    # save the results 
    if args.n_exp > 1:
        for key, value in out[0].items():
            for metric_key in list(value.keys()):
                metric_value = [o[key][metric_key] for o in out]
                np.save(os.path.join(save_dir, f'{key}_{metric_key}.npy'), metric_value)              
    else:
        for key, value in out.items():
            for metric_key, metric_value in value.items(): 
                np.save(os.path.join(save_dir, f'{key}_{metric_key}.npy'), metric_value)
