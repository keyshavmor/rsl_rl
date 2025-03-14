import warnings
warnings.filterwarnings('ignore')
import os
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MUJOCO_GL'] = 'egl'

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
# from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer, ReplayBufferV2
import logger
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'





def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episodes):
		obs, privileged_obs, obs_history = env.reset()
		done, ep_reward, t = False, 0, 0
		if video: video.init(env, enabled=(i==0))
		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			obs_dict, reward, done, _ = env.step(action) # cpu().numpy()
			obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
			ep_reward += reward
			if video: video.record(env)
			t += 1
		episode_rewards.append(ep_reward.cpu().numpy())
		if video: video.save(env_step)
	return np.nanmean(episode_rewards)


def train(cfg):
	"""Training script for TD-MPC. Requires a CUDA-enabled device."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	work_dir = Path('/home/themandalorian/Workspace/hanyu-repo/rsl_rl/rsl_rl/tdmpc') / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	# env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)
	args = get_args()
	env, env_cfg = task_registry.make_env(name=args.task, args=args, runner_type="ppo")
	env = OneBatchWrapper(env)
    
	# ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, runner_type="rma")
	# ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
	
	agent, buffer = TDMPC(cfg), ReplayBufferV2(cfg)
	
	# Run training
	L = logger.Logger(work_dir, cfg)
	episode_idx, start_time = 0, time.time()
	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):

		# Collect trajectory
		obs, privileged_obs, obs_history = env.reset() # obs [48]
		episode = Episode(cfg, obs)
		while not episode.done:
			action = agent.plan(obs, step=step, t0=episode.first)
			obs_dict, reward, done, _ = env.step(action) #.cpu().numpy()   # action [12]
			obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
			episode += (obs, action, reward, done)
		# assert len(episode) == cfg.episode_length
		if len(episode) >=7:
			buffer += episode
		else:
			pass

		# Update model
		train_metrics = {}
		if step >= cfg.seed_steps:  # 5000
			num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i))

		# motor strength
		# motor_strength_onehot = obs[-2:]
		#motor_strength_onehot = env.motor_strength_onehot[0,:]
		#if motor_strength_onehot[0]==1:
		#	motor_strength = 0.9
		#	assert motor_strength == env.motor_strengths[0, 0]
		#elif motor_strength_onehot[1] == 1:
		#	motor_strength = 1.0
		#	assert motor_strength == env.motor_strengths[0, 0]
		#elif motor_strength_onehot[2] == 1:
		#	motor_strength = 1.1
		#	assert motor_strength == env.motor_strengths[0, 0]
		# motor_strength = obs[-1]
		
		motor_strength = 1.0

		# Log training episode
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			# 'vel_command': obs[9].detach()/2,
			'motor_strength': torch.tensor(motor_strength),
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward}
		train_metrics.update(common_metrics)
		L.log(train_metrics, category='train')

		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:  # cfg.eval_freq 20000
			common_metrics['episode_reward'] = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video) # cfg.eval_episodes 10
			L.log(common_metrics, category='eval')

	L.finish(agent)
	print('Training completed successfully')


if __name__ == '__main__':
	train(parse_cfg(Path('/home/themandalorian/Workspace/hanyu-repo/rsl_rl/rsl_rl/tdmpc')/ __CONFIG__))
