import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        #self.sigma = torch.zeros(1)  # TODO: Implement accordingly (T1, T2)
        #self.sigma = torch.tensor([5.0]) #constant variance sigma^2 = 5 T1 - Done
        #self.sigma = torch.tensor([10.]) #T2,
        self.sigma = torch.nn.Parameter(torch.tensor([10.])) #T2,

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, variance):
        x = self.fc1(x)
        x = F.relu(x)
        action_mean = self.fc2_mean(x)
        sigma = variance  # TODO: Is it a good idea to leave it like this? =>
        # No , according to formula we need to used sigma^2 = sigma_{0}^2 * exp... for T2

        # TODO: Instantiate and return a normal distribution
        # with mean mu and std of sigma (T1) - Done

        action_dist = Normal(action_mean, torch.sqrt(sigma))

        return action_dist

class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []

        self.baseline = 20
        self.variance = self.policy.sigma

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        #c = 5e-4 #T2
        #self.variance = self.policy.sigma * np.exp(-c * episode_number)

        # TODO: Compute discounted rewards (use the discount_rewards function)
        G = discount_rewards(r=rewards, gamma=self.gamma)
        #normalized rewards
        G = (G - torch.mean(G)) / torch.std(G)

        # TODO: Compute the optimization term (T1)
        loss = torch.sum(-G * action_probs) #basic REINFORCE
        #loss = torch.sum(-(G-self.baseline)*action_probs) #REINFORCE with baseline

        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        loss.backward()

        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1)
        #actions_probs = self.policy.forward(x)
        actions_probs = self.policy.forward(x,self.variance)

        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action = actions_probs.mean # np.mean(actions_probs)
        else:
            action = actions_probs.sample((1,))[0]

        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = actions_probs.log_prob(action)

        return action, act_log_prob

    def store_outcome(self, observation, action_prob, action_taken, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))

