import gym
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

#from utils import ReplayMemory, Transition
from assignment4.utils import ReplayMemory, Transition

env = gym.make("CartPole-v0")
actions = env.action_space.n


class Agent(object):
    def __init__(self, num_actions, gamma=0.98, memory_size=5000, batch_size=32):
        self.scaler = None
        self.featurizer = None
        self.q_functions = None
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.memory = ReplayMemory(memory_size)
        self.initialize_model()

    def initialize_model(self):
        # Draw some samples from the observation range and initialize the scaler
        obs_limit = np.array([4.8, 5, 0.5, 5])
        samples = np.random.uniform(-obs_limit, obs_limit, (1000, obs_limit.shape[0]))
        self.scaler = StandardScaler()
        self.scaler.fit(samples)

        # Initialize the RBF featurizer
        self.featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=80)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=50)),
        ])
        self.featurizer.fit(self.scaler.transform(samples))

        # Create a value approximator for each action
        self.q_functions = [SGDRegressor(learning_rate="constant", max_iter=500, tol=1e-3)
                       for _ in range(self.num_actions)]

        # Initialize it to whatever values; implementation detail
        for q_a in self.q_functions:
            q_a.partial_fit(self.featurize(samples), np.zeros((samples.shape[0],)))

    def featurize(self, state):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        # Task 1a: TODO: Use (s, abs(s)) as features
        #featurized_state = np.concatenate((state, abs(state)), axis=1)

        # Task 1b: RBF features
        featurized_state =  self.featurizer.transform(self.scaler.transform(state))

        return featurized_state

    def get_action(self, state, epsilon=0.0):
        if np.random.random() < epsilon:
            a = int(np.random.random() * self.num_actions)
            return a
        else:
            featurized = self.featurize(state)
            qs = [q.predict(featurized)[0] for q in self.q_functions]
            qs = np.array(qs)
            a = np.argmax(qs, axis=0)
            return a

    def single_update(self, state, action, next_state, reward, done):
        # Calculate feature representations of the
        # Task 1: TODO: Set the feature state and feature next state
        featurized_state, featurized_next_state = self.featurize(state), self.featurize(next_state)

        # Task 1:  TODO Get Q(s', a) for the next state
        pred = np.array([q.predict(featurized_next_state)[0] for q in self.q_functions])
        next_qs = np.amax(pred, axis=0)

        # Calculate the updated target Q- values
        # Task 1: TODO: Calculate target based on rewards and next_qs
        if done:
            target = reward
        else:
            target = reward+self.gamma*next_qs

        # Update Q-value estimation
        #self.q_functions[action].partial_fit(featurized_state, target)
        self.q_functions[action].partial_fit(featurized_state, (target,))

    def update_estimator(self):
        if len(self.memory) < self.batch_size:
            # Use the whole memory
            samples = self.memory.memory
        else:
            # Sample some data
            samples = self.memory.sample(self.batch_size)

        # Task 2: TODO: Reformat data in the minibatch
        states = np.array([sample.state for sample in samples])
        action = np.array([sample.action for sample in samples])
        next_states = np.array([sample.next_state for sample in samples])
        rewards = np.array([sample.reward for sample in samples])
        dones = np.array([sample.done for sample in samples])

        # Task 2: TODO: Calculate Q(s', a)
        featurized_next_states = self.featurize(next_states)
        next_qs = np.array([np.amax(np.array([q.predict(s_.reshape(1,-1))[0] for q in self.q_functions]),axis=0) for s_ in featurized_next_states])

        # Calculate the updated target values
        # Task 2: TODO: Calculate target based on rewards and next_qs
        #(1 - dones) will be zero when done is True
        targets = rewards + self.gamma * next_qs * (1 - dones)

        # Calculate featurized states
        featurized_states = self.featurize(states)

        # Get new weights for each action separately
        for a in range(self.num_actions):
            # Find states where a was taken
            idx = action == a

            # If a not present in the batch, skip and move to the next action
            if np.any(idx):
                act_states = featurized_states[idx]
                act_targets = targets[idx]

                # Perform a single SGD step on the Q-function params
                self.q_functions[a].partial_fit(act_states, act_targets)

    def store_transition(self, *args):
        self.memory.push(*args)

