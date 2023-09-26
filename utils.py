import numpy as np
import torch
import random
import math


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim,n, km_num, max_size=int(1e5)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.previous_state = np.zeros((max_size, n, state_dim + action_dim))
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.km_num = km_num
		self.previous_next_state = np.zeros((max_size, n, state_dim + action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, previous_state, state, action, previous_next_state, next_state, reward, done):
		self.previous_state[self.ptr] = previous_state
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.previous_next_state[self.ptr] = previous_next_state
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.previous_state[ind]).to(self.device),
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.previous_next_state[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def sample1(self, batch_size, temp_number):
		ind = np.zeros((0),dtype=int)
		for i in range(len(temp_number)-1):
			ind1 = random.sample(range(temp_number[i], temp_number[i+1]), int(math.ceil(batch_size / self.km_num)))
			ind = np.hstack((ind1, ind))
		return (
			torch.FloatTensor(self.previous_state[ind]).to(self.device),
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.previous_next_state[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def sampleall(self):
		return (
			torch.FloatTensor(self.previous_state).to(self.device),
			torch.FloatTensor(self.state).to(self.device),
			torch.FloatTensor(self.action).to(self.device),
			torch.FloatTensor(self.previous_next_state).to(self.device),
			torch.FloatTensor(self.next_state).to(self.device),
			torch.FloatTensor(self.reward).to(self.device),
			torch.FloatTensor(self.not_done).to(self.device)
		)


	def Choose_sample(self, result):
		index = []
		for i in range(self.km_num):
			index0 = np.where(result == i)
			index0 = np.array(index0)
			index0 = index0.tolist()
			index0 = index0[0]
			index0 = np.array(index0)
			index.append(index0)

		return index


	def sample_ind(self,ind):
		sample = []
		for i in range(len(ind)):
			temp_sample = [self.previous_state[int(ind[i])],self.state[int(ind[i])], self.action[int(ind[i])],self.previous_next_state[int(ind[i])], self.next_state[int(ind[i])], self.reward[int(ind[i])], self.not_done[int(ind[i])]]
			sample.append(temp_sample)
		return sample



class ReplayBuffer1(object):
	def __init__(self, state_dim, action_dim,n, max_size=int(5e5)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.previous_state = np.zeros((max_size, n, state_dim + action_dim))
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.previous_next_state = np.zeros((max_size, n, state_dim + action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, previous_state, state, action, previous_next_state, next_state, reward, done):
		self.previous_state[self.ptr] = previous_state
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.previous_next_state[self.ptr] = previous_next_state
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.previous_state[ind]).to(self.device),
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.previous_next_state[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
