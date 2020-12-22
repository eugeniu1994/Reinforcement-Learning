import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 16
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        # TODO: Add another linear layer for the critic
        self.value = torch.nn.Linear(self.hidden, 1)
        # TODO: Implement learned variance (or copy from Ex5)
        self.sigma = torch.nn.Parameter(torch.tensor([10.]))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        # Actor part
        action_mean = self.fc2_mean(x)
        sigma = F.softplus(self.sigma)  # TODO: Implement (or copy from Ex5)

        # TODO: Implement
        value = self.value(x)
        # TODO: Instantiate and return a normal distribution

        action_dist = Normal(action_mean, sigma)
        # TODO: Return state value in addition to the distribution
        return action_dist, value


class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.next_states = []
        self.done = []

    def update_policy(self, episode_number):
        self.optimizer.zero_grad()
        # Convert buffers to Torch tensors
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        # Clear state transition buffers
        self.states, self.action_probs, self.rewards = [], [], []
        self.next_states, self.done = [], []

        # TODO: Compute state values
        _, V = self.policy.forward(states)
        V = V.squeeze(-1)
        with torch.no_grad():
            V_ = self.policy.forward(next_states)[1].squeeze(-1)

        # TODO: Compute critic loss (MSE)
        q = rewards + self.gamma * V_.detach() * (1-done)
        critic_loss = torch.mean(torch.square(V-q))

        # Advantage estimates
        # TODO: Compute advantage estimates
        A = rewards + (self.gamma * V_ * (1-done)) - V

        # TODO: Calculate actor loss (very similar to PG)
        actor_loss = torch.mean((-action_probs) * A.detach())

        # TODO: Compute the gradients of loss w.r.t. network parameters
        (critic_loss+actor_loss).backward()

        # TODO: Update network parameters using self.optimizer and zero gradients
        self.optimizer.step()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)
        # TODO: Pass state x through the policy network
        dist, _ = self.policy.forward(x)
        # TODO: Return mean if evaluation, else sample from the distribution
        if evaluation:
            action = torch.mean(x)
        else:
            action = dist.sample()
        # TODO: Calculate the log probability of the action

        act_log_prob = dist.log_prob(action)
        return action, act_log_prob

    def store_outcome(self, state, next_state, action_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
