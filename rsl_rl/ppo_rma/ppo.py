import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb
from params_proto import PrefixProto
import pickle

# from go1_gym_learn.ppo_cse import ActorCritic
# from go1_gym_learn.ppo_cse import RolloutStorage
# from go1_gym_learn.ppo_cse import caches
from rsl_rl.ppo_rma.actor_critic import ActorCritic
from rsl_rl.ppo_rma.rollout_storage import RolloutStorage
from rsl_rl.ppo_rma.history_encoder import min_max_scaler


class PPORMA_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3  # 5.e-4
    adaptation_module_learning_rate = 1.e-3
    num_adaptation_module_substeps = 1
    schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.

    selective_adaptation_module_loss = False


class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic, device='cpu'):

        wandb.init(project= "Context-conditioned Estimator")
        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=PPORMA_Args.learning_rate)
        # self.optimizer = optim.Adam(list(self.actor_critic.actor_body.parameters()) + list(self.actor_critic.critic_body.parameters()), lr=PPORMA_Args.learning_rate)
        # self.optimizer = optim.Adam(list(self.actor_critic.actor_body.parameters()) + \
        #                             list(self.actor_critic.critic_body.parameters()) + \
        #                             list(self.actor_critic.history_encoder.parameters()) , lr=PPORMA_Args.learning_rate)


        self.adaptation_module_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                      lr=PPORMA_Args.adaptation_module_learning_rate)
        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(self.actor_critic.parameters(),
                                                          lr=PPORMA_Args.adaptation_module_learning_rate)
        self.transition = RolloutStorage.Transition()

        self.learning_rate = PPORMA_Args.learning_rate

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, obs_history, privileged_obs).detach()  # obs_history
        self.transition.values = self.actor_critic.evaluate(obs, privileged_obs).detach() # obs_history + privileged_obs
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # self.transition.env_bins = infos["env_bins"]  # by why
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += PPORMA_Args.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)

        # transition_data = {
        #     "observations": self.transition.observations.cpu(),
        #     "privileged_observations": self.transition.privileged_observations.cpu(),
        #     "observation_histories": self.transition.observation_histories.cpu(),
        #     "critic_observations": self.transition.critic_observations.cpu(),
        #     "actions": self.transition.actions.cpu(),
        #     "rewards": self.transition.rewards.cpu(),
        #     "dones": self.transition.dones.cpu(),
        #     "values": self.transition.values.cpu(),
        #     "actions_log_prob": self.transition.actions_log_prob.cpu(),
        #     "action_mean": self.transition.action_mean.cpu(),
        #     "action_sigma": self.transition.action_sigma.cpu()
        # }
        # with open('transition_data.pkl', 'ab') as f:
        #     pickle.dump(transition_data, f)
        
        self.transition.clear()
        self.actor_critic.reset(dones)
        

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).detach()
        self.storage.compute_returns(last_values, PPORMA_Args.gamma, PPORMA_Args.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        # mean_decoder_loss = 0
        # mean_decoder_loss_student = 0
        mean_adaptation_module_test_loss = 0
        # mean_decoder_test_loss = 0
        # mean_decoder_test_loss_student = 0
        generator = self.storage.mini_batch_generator(PPORMA_Args.num_mini_batches, PPORMA_Args.num_learning_epochs)
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch  in generator: # env_bins_batch by why

            self.actor_critic.act(obs_batch, obs_history_batch, privileged_obs_batch, masks=masks_batch)   # actor input: obs_history + estimated latents
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_batch, privileged_obs_batch, masks=masks_batch) # critic input: obs_history + privileged_obs
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if PPORMA_Args.desired_kl != None and PPORMA_Args.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (
                                torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (
                                2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > PPORMA_Args.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < PPORMA_Args.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - PPORMA_Args.clip_param,
                                                                               1.0 + PPORMA_Args.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if PPORMA_Args.use_clipped_value_loss:
                value_clipped = target_values_batch + \
                                (value_batch - target_values_batch).clamp(-PPORMA_Args.clip_param,
                                                                          PPORMA_Args.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + PPORMA_Args.value_loss_coef * value_loss - PPORMA_Args.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()  
            # for name, param in self.actor_critic.named_parameters():
            #     if param.grad is not None:
            #         print(f"Parameter: {name} has been updated, Gradient Norm: {param.grad.norm()}")
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), PPORMA_Args.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()


            # Adaptation module gradient step

            data_size = privileged_obs_batch.shape[0]  # 2048 * 24 / 4
            num_train = int(data_size // 10 * 9) # 2048 * 24 // 5 * 4  = 9828
            for epoch in range(PPORMA_Args.num_adaptation_module_substeps):

                adaptation_pred = self.actor_critic.history_encoder(obs_history_batch) # obs_history_batch [2048 * 24 / 4 , 48 * 30] adaptation_pred [2048 * 24 / 4 , 1]
                with torch.no_grad():
                    adaptation_target = self.actor_critic.env_factor_encoder(privileged_obs_batch).detach()
                    # residual = (adaptation_target - adaptation_pred).norm(dim=1)
                    # caches.slot_cache.log(env_bins_batch[:, 0].cpu().numpy().astype(np.uint8),
                    #                       sysid_residual=residual.cpu().numpy())

                selection_indices = torch.linspace(0, adaptation_pred.shape[1]-1, steps=adaptation_pred.shape[1], dtype=torch.long) # =0  shape [1]
                if PPORMA_Args.selective_adaptation_module_loss:
                    # mask out indices corresponding to swing feet
                    selection_indices = 0

                
                # print(f"target {adaptation_target[:5, :]}")
                # print(f"estimated {adaptation_pred[:5, :]}")
                scaled_adaptation_pred = min_max_scaler(adaptation_pred).detach()
                scaled_adaptation_target = min_max_scaler(adaptation_target).detach()
                
                adaptation_loss = F.mse_loss(adaptation_pred[:num_train, selection_indices], adaptation_target[:num_train, selection_indices])
                adaptation_test_loss = F.mse_loss(adaptation_pred[num_train:, selection_indices], adaptation_target[num_train:, selection_indices])
                
                scaled_adaptation_loss = F.mse_loss(scaled_adaptation_pred, scaled_adaptation_target)
                
                
                self.adaptation_module_optimizer.zero_grad()
                adaptation_loss.backward() 
                # for name, param in self.actor_critic.named_parameters():
                #     if param.grad is not None:
                #         print(f"Parameter: {name} has been updated, Gradient Norm: {param.grad.norm()}")
                self.adaptation_module_optimizer.step()

                mean_adaptation_module_loss += adaptation_loss.item()
                mean_adaptation_module_test_loss += adaptation_test_loss.item()


        num_updates = PPORMA_Args.num_learning_epochs * PPORMA_Args.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= (num_updates * PPORMA_Args.num_adaptation_module_substeps)
        # mean_decoder_loss /= (num_updates * PPORMA_Args.num_adaptation_module_substeps)
        # mean_decoder_loss_student /= (num_updates * PPORMA_Args.num_adaptation_module_substeps)
        mean_adaptation_module_test_loss /= (num_updates * PPORMA_Args.num_adaptation_module_substeps)
        # mean_decoder_test_loss /= (num_updates * PPORMA_Args.num_adaptation_module_substeps)
        # mean_decoder_test_loss_student /= (num_updates * PPORMA_Args.num_adaptation_module_substeps)


        wandb.log({
            "mean_value_loss": mean_value_loss,
            "mean_surrogate_loss": mean_surrogate_loss,
            "mean_adaptation_module_loss": mean_adaptation_module_loss,
            "mean_adaptation_module_test_loss": mean_adaptation_module_test_loss,
            "scaled_adaptation_loss": scaled_adaptation_loss,
            "learning_rate": self.learning_rate,
            "total_loss": loss.item()
        })
        self.storage.clear()

        # return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student
        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_adaptation_module_test_loss
