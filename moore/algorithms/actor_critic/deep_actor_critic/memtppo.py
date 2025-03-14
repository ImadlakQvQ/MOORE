import numpy as np
import os
import torch
import torch.nn.functional as F

from mushroom_rl.core import Agent
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor, update_optimizer_parameters
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import to_parameter

# framework
from moore.utils.dataset import parse_dataset

def compute_gae(V, c, s, ss, r, absorbing, last, gamma, lam):
    """
    Function to compute Generalized Advantage Estimation (GAE)
    and new value function target over a dataset.

    "High-Dimensional Continuous Control Using Generalized
    Advantage Estimation".
    Schulman J. et al.. 2016.

    Args:
        V (Regressor): the current value function regressor;
        s (numpy.ndarray): the set of states in which we want
            to evaluate the advantage;
        ss (numpy.ndarray): the set of next states in which we want
            to evaluate the advantage;
        r (numpy.ndarray): the reward obtained in each transition
            from state s to state ss;
        absorbing (numpy.ndarray): an array of boolean flags indicating
            if the reached state is absorbing;
        last (numpy.ndarray): an array of boolean flags indicating
            if the reached state is the last of the trajectory;
        gamma (float): the discount factor of the considered problem;
        lam (float): the value for the lamba coefficient used by GEA
            algorithm.
    Returns:
        The new estimate for the value function of the next state
        and the estimated generalized advantage.
    """
    v = V(s, c=c)
    v_next = V(ss, c=c)
    gen_adv = np.ones_like(v) * np.nan #np.empty_like(v)
    unique_c = np.unique(c)
    for ci in unique_c:
        ci = np.argwhere(c == ci).ravel()
        gen_adv_i = gen_adv[ci]
        for rev_k in range(len(v[ci])):
            k = len(v[ci]) - rev_k - 1
            if last[ci][k] or rev_k == 0:
                gen_adv_i[k] = r[ci][k] - v[ci][k]
                if not absorbing[ci][k]:
                    gen_adv_i[k] += gamma * v_next[ci][k]
            else:
                gen_adv_i[k] = r[ci][k] + gamma * v_next[ci][k] - v[ci][k] + gamma * lam * gen_adv_i[k + 1]
        gen_adv[ci] = gen_adv_i
    return gen_adv + v, gen_adv

class MEMTPPO(Agent):
    """
    Multi-Task Variant of Proximal Policy Optimization algorithm.
    "Proximal Policy Optimization Algorithms".
    Schulman J. et al.. 2017.

    """
    def __init__(self, mdp_info, policy, actor_optimizer, critic_params,
                 n_epochs_policy, batch_size, eps_ppo, lam, ent_coeff=0.0,
                 critic_fit_params=None, n_contexts = 1, descriptions=None):
        """
        Constructor.

        Args:
            policy (TorchPolicy): torch policy to be learned by the algorithm
            actor_optimizer (dict): parameters to specify the actor optimizer
                algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            n_epochs_policy ([int, Parameter]): number of policy updates for every dataset;
            batch_size ([int, Parameter]): size of minibatches for every optimization step
            eps_ppo ([float, Parameter]): value for probability ratio clipping;
            lam ([float, Parameter], 1.): lambda coefficient used by generalized
                advantage estimation;
            ent_coeff ([float, Parameter], 1.): coefficient for the entropy regularization term;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.
            n_contexts ([int, Parameter]): number of contexts (tasks) learned by the algorithm

        """
        self._critic_fit_params = dict(n_epochs=1) if critic_fit_params is None else critic_fit_params #modified from 10 to 1

        self._n_contexts = n_contexts

        self._n_epochs_policy = to_parameter(n_epochs_policy)
        self._batch_size = to_parameter(batch_size)
        self._eps_ppo = to_parameter(eps_ppo)

        self.policy_optimizer = actor_optimizer['class'](policy.parameters(), **actor_optimizer['params'])
        # self.router_optimizer = torch.optim.Adam(params=policy.router_parameters(), lr=1e-3, betas=(0.9, 0.999))
        self._lambda = to_parameter(lam)
        self._ent_coeff = to_parameter(ent_coeff)

        self._V = Regressor(TorchApproximator, **critic_params)

        self._iter = 1

        self._add_save_attr(
            _critic_fit_params='pickle', 
            _n_epochs_policy='mushroom',
            _batch_size='mushroom',
            _eps_ppo='mushroom',
            _ent_coeff='mushroom',
            _optimizer='torch',
            _lambda='mushroom',
            _V='mushroom',
            _iter='primitive',
            _n_contexts='primitive',
        )
        self.descriptions=descriptions
        super().__init__(mdp_info, policy, None)

        self._device = torch.device("cuda:0" if self.policy.use_cuda else "cpu")

    
    def fit(self, dataset, **info):
        c, x, u, r, xn, absorbing, last = parse_dataset(dataset, n_contexts=self._n_contexts)
        x = x.astype(np.float32)
        u = u.astype(np.float32)
        r = r.astype(np.float32)
        xn = xn.astype(np.float32)
        obs = to_float_tensor(x, self.policy.use_cuda)
        act = to_float_tensor(u, self.policy.use_cuda)

        v_target, np_adv = compute_gae(self._V, c, x, xn, r, absorbing, last, self.mdp_info.gamma, self._lambda())

        adv = torch.ones(np_adv.shape).to(self._device) * torch.nan # to handle the gpu case
        for ci in np.unique(c):
            ci = np.argwhere(c == ci).ravel()
            np_adv_i = (np_adv[ci] - np.mean(np_adv[ci])) / (np.std(np_adv[ci]) + 1e-8)
            adv[ci] = to_float_tensor(np_adv_i, self.policy.use_cuda)
                
        old_pol_dist, _ = self.policy.distribution_t([c, obs])
        old_log_p = old_pol_dist.log_prob(act)[:, None].detach()
        self._V.fit(x, v_target, c = c, **self._critic_fit_params)
        self._update_policy(c, obs, act, adv, old_log_p)


        # Print fit information
        self._log_info(dataset, c, x, v_target, old_pol_dist)
        self._iter += 1

    def _update_policy(self, c, obs, act, adv, old_log_p):
        # TODO loss function 然后训练router
        for epoch in range(self._n_epochs_policy()):
            for c_i, obs_i, act_i, adv_i, old_log_p_i in minibatch_generator(
                    self._batch_size(), c, obs, act, adv, old_log_p):

                self.policy_optimizer.zero_grad()
                prob_ratio = torch.exp(
                    self.policy.log_prob_t([c_i, obs_i], act_i) - old_log_p_i
                )

                clipped_ratio = torch.clamp(prob_ratio, 1 - self._eps_ppo(),
                                            1 + self._eps_ppo.get_value())


                loss = -torch.mean(torch.min(prob_ratio * adv_i,
                                             clipped_ratio * adv_i))
                entropy, experts_loss = self.policy.entropy_t([c_i, obs_i])
                loss -= self._ent_coeff()*entropy
                loss += experts_loss


                loss.backward()
                self.policy_optimizer.step()

    def _log_info(self, dataset, c, x, v_target, old_pol_dist):
        if self._logger:
            logging_verr = []
            torch_v_targets = torch.tensor(v_target, dtype=torch.float)
            for idx in range(len(self._V)):
                v_pred = torch.tensor(self._V(x, c = c, idx=idx), dtype=torch.float)
                v_err = F.mse_loss(v_pred, torch_v_targets)
                logging_verr.append(v_err.item())

            logging_ent = self.policy.entropy([c, x])
            new_pol_dist = self.policy.distribution([c, x])
            logging_kl = torch.mean(torch.distributions.kl.kl_divergence(
                new_pol_dist, old_pol_dist))
            avg_rwd = np.mean(compute_J(dataset))
            msg = "Iteration {}:\n\t\t\t\trewards {} vf_loss {}\n\t\t\t\tentropy {}  kl {}".format(
                self._iter, avg_rwd, logging_verr, logging_ent, logging_kl)

            self._logger.info(msg)
            self._logger.weak_line()

    def _post_load(self):
        if self.policy_optimizer is not None:
            update_optimizer_parameters(self.policy_optimizer, list(self.policy.parameters()))
    
    def save_params(self, save_dir):
        self._V.model.network.save_params(os.path.join(save_dir, "critic.pth"))
        self.policy._logits.model.network.save_params(os.path.join(save_dir, "actor.pth"))
        return 
    
    
