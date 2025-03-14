import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    observation_encoder_hidden_dims = [256, 128] # by why
    num_history_latents = 16
    adaptation_module_branch_hidden_dims = [256, 128]

    use_history_encoder = False  # assume use_history_encoder = True => use_adaptation_module = True
    use_adaptation_module = True  #by why
    use_decoder = False


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = AC_Args.use_decoder
        self.use_adaptation_module = AC_Args.use_adaptation_module
        self.use_history_encoder = AC_Args.use_history_encoder
        self.num_history_latents = AC_Args.num_history_latents
        super().__init__()

        self.num_obs = num_obs
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = get_activation(AC_Args.activation)

        if AC_Args.use_history_encoder:
            history_encoder_layers = []
            history_encoder_layers.append(nn.Linear(self.num_obs_history, AC_Args.observation_encoder_hidden_dims[0]))
            history_encoder_layers.append(activation)
            for l in range(len(AC_Args.observation_encoder_hidden_dims)):
                if l == len(AC_Args.observation_encoder_hidden_dims) - 1:
                    history_encoder_layers.append(
                        nn.Linear(AC_Args.observation_encoder_hidden_dims[l], self.num_history_latents))
                else:
                    history_encoder_layers.append(
                        nn.Linear(AC_Args.observation_encoder_hidden_dims[l],
                                AC_Args.observation_encoder_hidden_dims[l + 1]))
                    history_encoder_layers.append(activation)
            self.history_encoder = nn.Sequential(*history_encoder_layers)

            # Adaptation module
            adaptation_module_layers = []
            adaptation_module_layers.append(nn.Linear(self.num_obs_history, AC_Args.adaptation_module_branch_hidden_dims[0]))
            adaptation_module_layers.append(activation)
            for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
                if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
                    adaptation_module_layers.append(
                        nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
                else:
                    adaptation_module_layers.append(
                        nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
                                AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                    adaptation_module_layers.append(activation)
            self.adaptation_module = nn.Sequential(*adaptation_module_layers)

            # Policy
            actor_layers = []
            actor_layers.append(nn.Linear(self.num_privileged_obs + self.num_history_latents, AC_Args.actor_hidden_dims[0]))
            actor_layers.append(activation)
            for l in range(len(AC_Args.actor_hidden_dims)):
                if l == len(AC_Args.actor_hidden_dims) - 1:
                    actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
                else:
                    actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                    actor_layers.append(activation)
            self.actor_body = nn.Sequential(*actor_layers)

            # Value function
            critic_layers = []
            critic_layers.append(nn.Linear(self.num_privileged_obs + self.num_history_latents, AC_Args.critic_hidden_dims[0]))
            critic_layers.append(activation)
            for l in range(len(AC_Args.critic_hidden_dims)):
                if l == len(AC_Args.critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
                else:
                    critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                    critic_layers.append(activation)
            self.critic_body = nn.Sequential(*critic_layers)


        else:
            # Adaptation module
            if AC_Args.use_adaptation_module:
                adaptation_module_layers = []
                adaptation_module_layers.append(nn.Linear(self.num_obs_history, AC_Args.adaptation_module_branch_hidden_dims[0]))
                adaptation_module_layers.append(nn.Tanh())
                # adaptation_module_layers.append(nn.BatchNorm1d(AC_Args.adaptation_module_branch_hidden_dims[0]))
                for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
                    if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
                        adaptation_module_layers.append(
                            nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
                    else:
                        adaptation_module_layers.append(
                            nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
                                    AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                        adaptation_module_layers.append(nn.Tanh())
                        # adaptation_module_layers.append(nn.BatchNorm1d(AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(nn.Tanh())
                self.adaptation_module = nn.Sequential(*adaptation_module_layers)

            # Policy
            actor_layers = []
            if AC_Args.use_adaptation_module:  # by why
                actor_layers.append(nn.Linear(self.num_obs + self.num_privileged_obs, AC_Args.actor_hidden_dims[0]))
            else:
                actor_layers.append(nn.Linear(self.num_obs_history, AC_Args.actor_hidden_dims[0]))
            # actor_layers.append(nn.Linear(self.num_obs_history, AC_Args.actor_hidden_dims[0]))
            actor_layers.append(activation)
            for l in range(len(AC_Args.actor_hidden_dims)):
                if l == len(AC_Args.actor_hidden_dims) - 1:
                    actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
                else:
                    actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                    actor_layers.append(activation)
            self.actor_body = nn.Sequential(*actor_layers)

            # Value function
            critic_layers = []
            if AC_Args.use_adaptation_module:  # by why
                critic_layers.append(nn.Linear(self.num_obs + self.num_privileged_obs, AC_Args.critic_hidden_dims[0]))
            else:
                critic_layers.append(nn.Linear( self.num_obs_history, AC_Args.critic_hidden_dims[0]))
            critic_layers.append(activation)
            for l in range(len(AC_Args.critic_hidden_dims)):
                if l == len(AC_Args.critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
                else:
                    critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                    critic_layers.append(activation)
            self.critic_body = nn.Sequential(*critic_layers)

        if self.use_history_encoder:
            print(f"History Encoder: {self.history_encoder}")
        if self.use_adaptation_module:
            print(f"Adaptation Module: {self.adaptation_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs, observation_history):
        if AC_Args.use_history_encoder:
            history_latents = self.history_encoder(observation_history)
            latent = self.adaptation_module(observation_history)
            mean = self.actor_body(torch.cat((history_latents, latent), dim=-1))
        else:
            # mean = self.actor_body(observation_history)
            if AC_Args.use_adaptation_module:
                latent = self.adaptation_module(observation_history)
                mean = self.actor_body(torch.cat((obs, latent), dim=-1))
            else:
                mean = self.actor_body(obs)
            # mean = self.actor_body(obs)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, obs, observation_history, **kwargs):
        self.update_distribution(obs, observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs"], ob["obs_history"], policy_info=policy_info)

    def act_student(self, obs, observation_history, policy_info={}):
        if AC_Args.use_history_encoder:
            history_latents = self.history_encoder(observation_history)
            latent = self.adaptation_module(observation_history)
            actions_mean = self.actor_body(torch.cat((history_latents, latent), dim=-1))
            policy_info["latents"] = latent.detach().cpu().numpy()
        else:
            # actions_mean = self.actor_body(observation_history)
            # latent = self.adaptation_module(observation_history)
            # print(f"estimated {latent[:100, 0]}" )
            if AC_Args.use_adaptation_module:
                latent = self.adaptation_module(observation_history)
                # print(f"estimated {latent[:3, 0]}" )
                actions_mean = self.actor_body(torch.cat((obs, latent), dim=-1))
                policy_info["latents"] = latent.detach().cpu().numpy()
            else:
                actions_mean = self.actor_body(obs)
            # actions_mean = self.actor_body(obs)
        return actions_mean

    def act_teacher(self, observation_history, privileged_info, policy_info={}):
        actions_mean = self.actor_body(torch.cat((observation_history, privileged_info), dim=-1))
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, obs, privileged_observations, **kwargs):
        if AC_Args.use_history_encoder:
            history_latents = self.history_encoder(observation_history)
            value = self.critic_body(torch.cat((history_latents, privileged_observations), dim=-1))
        else:
            if AC_Args.use_adaptation_module:
                value = self.critic_body(torch.cat((obs, privileged_observations), dim=-1))
            else:
                value = self.critic_body(obs)
        return value

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
