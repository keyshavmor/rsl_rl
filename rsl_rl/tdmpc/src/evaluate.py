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


def evaluate(cfg, num_episodes):
	"""Evaluate a trained agent and optionally save a video."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	work_dir = Path('/home/themandalorian/Workspace/hanyu-repo/rsl_rl/rsl_rl/tdmpc') / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
	args = get_args()
	env, env_cfg = task_registry.make_env(name=args.task, args=args, runner_type="ppo")
	env = OneBatchWrapper(env)

	agent = TDMPC(cfg)
	agent.load(work_dir / 'models' /'model.pt')
	step = cfg.train_steps

	L = logger.Logger(work_dir, cfg)
	episode_idx, start_time = 0, time.time()

	episode_rewards = []
	for i in range(num_episodes):
		obs, privileged_obs, obs_history = env.reset()
		done, ep_reward, t = False, 0, 0

		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			obs_dict, reward, done, _ = env.step(action) # cpu().numpy()
			obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
			ep_reward += reward
			t += 1
		episode_rewards.append(ep_reward.cpu().numpy())
		episode_idx += 1


		#motor_strength_onehot = env.motor_strength_onehot[0,:]
		#if motor_strength_onehot[0]==1:
		#	motor_strength = 0.9
	#		assert motor_strength == env.motor_strengths[0, 0]
		#elif motor_strength_onehot[1] == 1:
		#	motor_strength = 1.0
		#	assert motor_strength == env.motor_strengths[0, 0]
		#elif motor_strength_onehot[2] == 1:
		#	motor_strength = 1.1
		#	assert motor_strength == env.motor_strengths[0, 0]

		motor_strength = 1.0

		common_metrics = {
			'episode': episode_idx,
			'env_step': episode_idx,
			'motor_strength': torch.tensor(motor_strength),
			'total_time': time.time() - start_time,
			'episode_reward': ep_reward.cpu().numpy()}
		L.log(common_metrics, category='eval')

	L.finish(agent)
	print('Evaluation completed successfully')
	return np.nanmean(episode_rewards)


if __name__ == '__main__':
	evaluate(parse_cfg(Path('/home/themandalorian/Workspace/hanyu-repo/rsl_rl/rsl_rl/tdmpc')/ __CONFIG__), num_episodes=20)
