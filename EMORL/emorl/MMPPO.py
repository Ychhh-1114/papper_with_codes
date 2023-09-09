import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
from constant_params import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from copy import deepcopy
from roles import *
from evo_algo import Offspring

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Sigmoid()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state).flatten()
        cov_mat = torch.diag(self.action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample().flatten()


        with torch.no_grad():
            action_logprob = dist.log_prob(action).data.numpy()

        action[0] = torch.clip(action[0] * MAX_ANGLE, 0, MAX_ANGLE)
        action[1] = torch.clip(action[1] * MAX_DIS, 0, MAX_DIS)
        action[2] = torch.clip(action[2], 0, 1)

        memory.states.append(state)

        memory.actions.append(action)
        memory.logprobs.append(torch.Tensor([float(action_logprob)]))

        return action.detach().to(torch.float32)

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)

        dist_entropy = dist.entropy()

        state_value = self.critic(state)

        return action_logprobs.to(torch.float32), torch.squeeze(state_value).to(torch.float32), dist_entropy.to(
            torch.float32)


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory,w):

        rewards = []
        discounted_reward = np.array([0,0,0])
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)


        rewards = torch.tensor(rewards).to(device).to(torch.float32)

        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach().to(torch.float32)
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach().to(torch.float32)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach().to(torch.float32)

        for _ in range(self.K_epochs):

            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)


            ratios = torch.exp(logprobs - old_logprobs.detach()).to(torch.float32)
            advantages = rewards - state_values.detach()
            advantages = torch.Tensor(w) @ advantages.reshape(TIME_SLOTS - 1,3,1)


            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return deepcopy(self)


def conduct(ppo,w,model_list,node_list,base_station):

    memory = Memory()

    time_step = 0

    for episode in range(n_iter):

        uav = UAV(node_list,base_station)

        running_reward = np.array([0.0, 0.0, 0.0])
        for t in range(TIME_SLOTS):
            time_step += 1
            state = torch.Tensor(uav.get_obs())
            action = ppo.select_action(state, memory)
            reward = uav.step(action)

            memory.rewards.append(reward)
            memory.is_terminals.append(0)
            running_reward += reward
            # print(reward)
            if time_step % (TIME_SLOTS - 1) == 0:
                model = ppo.update(memory,w)
                memory.clear_memory()
                offspring = Offspring(model,w)
                offspring.r = running_reward
                model_list.append(offspring)
                # print("runnr: ",running_reward)







