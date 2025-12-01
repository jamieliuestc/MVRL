import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AE1(nn.Module):
	def __init__(self, state_dim, action_dim, n_step):
		super(AE1, self).__init__()

		self.encoder = nn.LSTM(state_dim + action_dim, math.ceil(state_dim / n_step), 5)
		self.decoder = nn.LSTM(math.ceil(state_dim / n_step), state_dim + action_dim, 5)

	def forward(self, x):

		encoder_state, h_t = self.encoder(x)
		decoder_state, _ = self.decoder(encoder_state)

		return encoder_state, decoder_state


class AE2(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(AE2, self).__init__()

		#encoder
		self.e1 = nn.Linear(state_dim + action_dim, 512)
		self.e2 = nn.Linear(512, 512)
		self.z_layer = nn.Linear(512, state_dim)

		#decoder
		self.d1 = nn.Linear(state_dim, 512)
		self.d2 = nn.Linear(512, 512)
		self.x_bar_layer = nn.Linear(512, state_dim + action_dim)

	def forward(self, state, action):
    	
		sa = torch.cat([state, action], 1)

		# encoder
		a = F.relu(self.e1(sa))
		a = F.relu(self.e2(a))

		encoder_state = self.z_layer(a)

		# decoder
		b = F.relu(self.d1(encoder_state))
		b = F.relu(self.d2(b))
		decoder_state = self.x_bar_layer(b)

		return encoder_state, decoder_state


class AE3(nn.Module):
	def __init__(self, state_dim, reward_dim):
		super(AE3, self).__init__()

		#encoder
		self.e1 = nn.Linear(state_dim + reward_dim , 512)
		self.e2 = nn.Linear(512, 512)
		self.z_layer = nn.Linear(512, state_dim)

		#decoder
		self.d1 = nn.Linear(state_dim, 512)
		self.d2 = nn.Linear(512, 512)
		self.x_bar_layer = nn.Linear(512, state_dim+reward_dim)

	def forward(self, state, reward):
		sr = torch.cat([state, reward], 1)

		# encoder
		a = F.relu(self.e1(sr))
		a = F.relu(self.e2(a))

		encoder_state = self.z_layer(a)

		# decoder
		b = F.relu(self.d1(encoder_state))
		b = F.relu(self.d2(b))
		decoder_state = self.x_bar_layer(b)

		return encoder_state, decoder_state


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, n_step):
		super(Actor, self).__init__()

		#self.l1 = nn.Linear(n_step * state_dim, 128)
		self.l1 = nn.Linear(n_step * math.ceil(state_dim / n_step), 128)
		self.l2 = nn.Linear(state_dim, 384)
		self.l3 = nn.Linear(512, 512)
		self.l4 = nn.Linear(512, action_dim)
		self.dropout = nn.Dropout(p=0.2)
		self.max_action = max_action
		self.state_dim = state_dim
		self.n_step = n_step

	def forward(self, previous_state, state):

		a = F.relu(self.l1(previous_state))
		b = F.relu(self.l2(state))

		c = torch.cat([a, b], 1)
		c = F.relu(self.l3(c))
		#c = self.dropout(c)
		return self.max_action * torch.tanh(self.l4(c))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, n_step):
		super(Critic, self).__init__()

		# Q1 architecture
		#self.l1 = nn.Linear((n_step + 1) * state_dim + action_dim, 600)
		self.l1 = nn.Linear(n_step * math.ceil(state_dim / n_step), 128)
		self.l2 = nn.Linear(state_dim + action_dim, 384)
		self.l3 = nn.Linear(512, 512)
		self.l4 = nn.Linear(512, 1)

		# Q2 architecture
		#self.l4 = nn.Linear((n_step + 1) * state_dim + action_dim, 600)
		self.l5 = nn.Linear(n_step * math.ceil(state_dim / n_step), 128)
		self.l6 = nn.Linear(state_dim + action_dim, 384)
		self.l7 = nn.Linear(512, 512)
		self.l8 = nn.Linear(512, 1)

		self.dropout = nn.Dropout(p=0.1)

	def forward(self, previous_state, state, action):

		sa = torch.cat([state, action], 1)

		a1 = F.relu(self.l1(previous_state))
		b1 = F.relu(self.l2(sa))
		psa1 = torch.cat([a1, b1], 1)

		q1 = F.relu(self.l3(psa1))
		#q1 = self.dropout(q1)
		q1 = self.l4(q1)
		

		a2 = F.relu(self.l5(previous_state))
		b2 = F.relu(self.l6(sa))
		psa2 = torch.cat([a2, b2], 1)

		q2 = F.relu(self.l7(psa2))
		#q2 = self.dropout(q2)
		q2 = self.l8(q2)
		return q1, q2


	def Q1(self, previous_state, state, action):

		sa = torch.cat([state, action], 1)

		a1 = F.relu(self.l1(previous_state))
		b1 = F.relu(self.l2(sa))
		psa1 = torch.cat([a1, b1], 1)

		q1 = F.relu(self.l3(psa1))
		#q1 = self.dropout(q1)
		q1 = self.l4(q1)
		return q1


class MVRL(object):
	def __init__(
		self, 
		state_dim, 
		action_dim, 
		reward_dim,
		max_action,
		discount, 
		tau, 
		policy_noise, 
		noise_clip, 
		policy_freq,
		n_step
	):
	
		self.ae1 = AE1(state_dim, action_dim, n_step).to(device)
		self.ae1_optimizer = torch.optim.Adam(self.ae1.parameters(), lr=3e-4)

		self.ae2 = AE2(state_dim, action_dim).to(device)
		self.ae2_optimizer = torch.optim.Adam(self.ae2.parameters(), lr=3e-4)

		self.ae3 = AE3(state_dim, reward_dim).to(device)
		self.ae3_optimizer = torch.optim.Adam(self.ae3.parameters(), lr=3e-4)

		self.actor = Actor(state_dim, action_dim, max_action, n_step).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim, n_step).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, previous_state, state):
		previous_state = torch.FloatTensor(previous_state).to(device)
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		previous_state = previous_state.transpose(0,1) 
		encoder_state, decoder_state = self.ae1(previous_state)
		encoder_state = encoder_state.transpose(0,1) 
		encoder_state = encoder_state.view(encoder_state.size(0), -1)
		return self.actor(encoder_state, state).cpu().data.numpy().flatten()

	def extract_state(self, previous_state):
		previous_state = previous_state.transpose(0,1) 
		encoder_state, decoder_state = self.ae1(previous_state)
		encoder_state = encoder_state.transpose(0,1) 
		encoder_state = encoder_state.reshape(encoder_state.size(0), -1)
		#encoder_state = encoder_state.contiguous().view(encoder_state.size(0), -1)
		return encoder_state

	def extract_cur_state(self, state, action):
		encoder_cur_state, decoder_cur_state = self.ae2(state, action)
		#encoder_state = encoder_state.contiguous().view(encoder_state.size(0), -1)
		return encoder_cur_state

	def extract_nr_state(self, state, reward):
		encoder_nr_state, decoder_nr_state = self.ae3(state, reward)
		return encoder_nr_state

	def pre_train(self, replay_buffer, batch_size, n_step):

		previous_state, state, action, previous_next_state, next_state, reward, not_done = replay_buffer.sample(batch_size)

		previous_state = previous_state.transpose(0,1)
		encoder_state, decoder_state = self.ae1(previous_state)

		cosine_sim = F.cosine_similarity(previous_state, decoder_state, dim=2)
		similarity_loss = (1.0 - cosine_sim.mean()) ** 2
		MSE_loss= F.mse_loss(previous_state, decoder_state)
		ae1_loss = 0.99 * similarity_loss + 0.01 * MSE_loss

		self.ae1_optimizer.zero_grad()
		ae1_loss.backward()
		self.ae1_optimizer.step()
		

	def train(self, replay_buffer, batch_size, n_step):
		self.total_it += 1

		# Sample replay buffer 
		previous_state, state, action, previous_next_state, next_state, reward, not_done = replay_buffer.sample(batch_size)

		encoder_cur_state, decoder_cur_state = self.ae2(state, action)
		ae2_loss = F.mse_loss(torch.cat([state, action], 1), decoder_cur_state)
		
		self.ae2_optimizer.zero_grad()
		ae2_loss.backward()
		self.ae2_optimizer.step()

		encoder_nr_state, decoder_nr_state = self.ae3(next_state, reward)
		ae3_loss = F.mse_loss(torch.cat([next_state, reward], 1), decoder_nr_state)
		
		self.ae3_optimizer.zero_grad()
		ae3_loss.backward()
		self.ae3_optimizer.step()

		previous_state = previous_state.transpose(0,1)
		previous_next_state = previous_next_state.transpose(0,1)
		encoder_state, decoder_state = self.ae1(previous_state)
		encoder_next_state, decoder_next_state = self.ae1(previous_next_state)

		cosine_sim = F.cosine_similarity(previous_state, decoder_state, dim=2)
		similarity_loss = (1.0 - cosine_sim.mean()) ** 2
		MSE_loss= F.mse_loss(previous_state, decoder_state)
		ae1_loss = 0.99 * similarity_loss + 0.01 * MSE_loss
		
		self.ae1_optimizer.zero_grad()
		ae1_loss.backward()
		self.ae1_optimizer.step()

		encoder_state = encoder_state.transpose(0,1) 
		encoder_state = encoder_state.reshape(encoder_state.size(0), -1) 

		encoder_next_state = encoder_next_state.transpose(0,1) 
		encoder_next_state = encoder_next_state.reshape(encoder_next_state.size(0), -1) 

		encoder_state = encoder_state.detach()
		encoder_next_state = encoder_next_state.detach()

		with torch.no_grad():
			# Select action according to policy and add clipped noise 
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(encoder_next_state, next_state) + noise
			).clamp(-self.max_action, self.max_action)
			
			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(encoder_next_state, next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(encoder_state, state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		#critic_loss.backward(retain_graph=True)
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(encoder_state, state, self.actor(encoder_state, state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self,filename,directory):
		torch.save(self.actor.state_dict(),'%s/%s_actor.pth'% (director,filename))
		torch.save(self.critic.state_dict(),'%s/%s_critic.pth'% (director,filename))
