import numpy as np
import torch
import gym
import argparse
import os

import utils
import MVRL
import clustering


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Runs policy for X episodes and returns average reward
def eval_policy(policy, env_name, seed, n_step, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		#state = state[:-1]
		previous_state = np.zeros((n_step, state_dim + action_dim))

		while not done:
			state1 = state
			previous_state1 = np.expand_dims(previous_state, axis=0) 
			action = policy.select_action(np.array(previous_state1), np.array(state))

			state, reward, done, _ = eval_env.step(action)
			#state = state[:-1]
			state_action = np.append(state1, action)
			previous_state = previous_state[1 : n_step]
			previous_state = np.row_stack((previous_state, state_action))

			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy_name", default="MVRL")					# Policy name
	parser.add_argument("--env_name", default="HalfCheetah-v2")			# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e4, type=int)		# How many time steps purely random policy is run for
	parser.add_argument("--eval_freq", default=5e3, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)		# Max time steps to run environment for
	parser.add_argument("--save_models", action="store_true")			# Whether or not models are saved
	parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)			# Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)			# Discount factor
	parser.add_argument("--tau", default=0.02, type=float)				# Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)		# Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)			# Frequency of delayed policy updates
	parser.add_argument("--n_step", default=15, type=int)			# number of steps
	parser.add_argument("--cluster_num", default=1e5, type=int)
	parser.add_argument("--km_num", default=3, type=int)
	parser.add_argument("--sample_rate", default=0.5, type=float)

	args = parser.parse_args()

	file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
	print("---------------------------------------")
	print("Settings: %s" % (file_name))
	print("---------------------------------------")
	if not os.path.exists("./results"):
		os.makedirs("./results")

	env = gym.make(args.env_name)

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	#state_dim = env.observation_space.shape[0] - 1
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	reward_dim = 1
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim, 
		"action_dim": action_dim, 
		"reward_dim": reward_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"n_step": args.n_step,
	}

	# Initialize policy
	if args.policy_name == "MVRL": 
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = MVRL.MVRL(**kwargs)
		Encoder = MVRL.AE1(state_dim, action_dim, args.n_step).to(device)



	buffer = utils.ReplayBuffer(state_dim, action_dim, args.n_step, args.km_num)
	buffer1 = utils.ReplayBuffer1(state_dim, action_dim, args.n_step)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env_name, args.seed, args.n_step)] 

	state, done = env.reset(), False
	state2 = state
	#state = state[:-1]
	previous_state = np.zeros((args.n_step, state_dim + action_dim))

	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	kinds_number = []

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			previous_state1 = np.expand_dims(previous_state, axis=0) 
			action = (
				policy.select_action(np.array(previous_state1), np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		#next_state = next_state[:-1]
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		previous_next_state = previous_state[1 : args.n_step]
		state_action = np.append(state, action)
		previous_next_state = np.row_stack((previous_next_state, state_action))

		# Store data in replay buffer
		buffer.add(previous_state, state, action, previous_next_state, next_state, reward, done_bool)

		k = int(t // args.cluster_num)
		args.batch_size =int(256 * (k + 8) / 8)
		args.batch_size = min(args.batch_size,1024)

		previous_state = previous_next_state
		state = next_state
		episode_reward += reward

		ind = np.zeros((0),dtype=int)

		if t > 0 and (t+1) % args.cluster_num == 0:
			kinds_number, ind = clustering.clustering(buffer, policy, kinds_number, args.km_num, ind, args.sample_rate)

		temp_sample = buffer.sample_ind(ind)

		for i in range(len(temp_sample)):
			buffer1.add(temp_sample[i][0],temp_sample[i][1],temp_sample[i][2],temp_sample[i][3], temp_sample[i][4], temp_sample[i][5], 1-temp_sample[i][6])


		# Train agent after collecting sufficient data
		if t > args.batch_size and t < args.eval_freq:
			policy.pre_train(buffer, args.batch_size, args.n_step)

		if t >= args.eval_freq and t < args.cluster_num:
			policy.train(buffer, args.batch_size, args.n_step)
		if t >= args.cluster_num:
			i = int(t // args.cluster_num)
			i = min(i, 5)
			j = int(t % (2 * i + 16))	
			if j < i:
				policy.train(buffer1, args.batch_size, args.n_step)
			else:
				policy.train(buffer, args.batch_size, args.n_step)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(("Total T: %d Episode Num: %d Episode T: %d Reward: %.3f") % (t+1, episode_num+1, episode_timesteps, episode_reward))
			# print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")			
			# Reset environment
			state, done = env.reset(), False
			#state = state[:-1]
			previous_state = np.zeros((args.n_step, state_dim + action_dim))
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env_name, args.seed, args.n_step))
			#policy.save("%s" % (file_name),directory="./pytorch_models")
			np.save("./results/%s" % (file_name), evaluations)
