import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from agent import Agent, Policy
from utils import get_space_dim


#Please note that the tran() and test() function were changed,
#I added 'agent.desired_x = 1.9' to keep the current desired target (used when train multiple cartpoles, otherwise keep in args.desired_x)
#For the third reward function, I needed to know what was the last desired target, and it is taken from args.desired_x
#but there were some issues when using it in multiple_cartpole.py threads - args.desired_x was unknown, so I saved it as an agent attribute
#Note if you want to use the desired parameter from args.desired_x or agent.desired_x please exchange lines 68 and 69 from reward function and also 90 and 91 from the same function.

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default= None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default= 500,
                        help="Number of episodes to train for")
    parser.add_argument("--render_training", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test", default = True)

    parser.add_argument("--desired_x", type=float, default=1.9,
                        help="desired point for reward function, should be in (-2, 2)")

    return parser.parse_args(args)

# TODO: Definition of the modified reward function
from collections import deque
velocities = deque(maxlen=100)  #used to track the last 100 velocities of the car
velocities.append(0)
left_most, right_most = -1.9, 1.9 #limits of the x (actual limits are -2.4-2.4)
def new_reward(state,agent):
    # Solving Task 1 reward = 1
    #reward = 1

    x, curr_speed = state[0], state[1]
    # Solving Task 2.1 ----------------------------------
    '''reward = 0.01  # initial reward
    diff = np.abs(x - 0)  # compute distance to target
    if diff < 0.1:  # if close enough, + reward it again
        reward += 10 
    else:
        reward += 1/diff'''

    # Solving Task 2.2 - this version can be used for prev case (special case desired_x=0)
    '''reward = 0.001    #initial reward
    desired_x = args.desired_x
    diff = np.abs(x - desired_x) #compute distance to target
    s = (1 - (diff / 4.8)) #normalize diff to be in [0,1] and substract it from 1
    reward += s #add this small reward (closer to the target => bigger reward)
    if diff < 0.01: #if close enough, + reward it again
        reward += 2 # 1 here means that (diff / 4.8) = 0, and its on the desired x
    if x < left_most - .2 or x > right_most + .2:  # if out of the range, penalize by -5
        reward -= s'''

    #Solving Task 2.3 ----------------------------------------------------------------------
    reward = 0.001   #set some init reward (small number, because 0 causes problems, on discauted reward zero division)
    #desired_x = agent.desired_x
    desired_x = args.desired_x #set my desired x

    if x >= right_most: #if the car is on the right, set objective to go left
        desired_x = left_most
        reward += 1.
    elif x<= left_most:  #if car is on the left, set objective to go right
        desired_x = right_most
        reward += 1.

    diff = np.abs(x - desired_x)  # compute distance to target
    s = (1 - (diff / 4.8))  # normalize diff to be in [0,1] and substract it from 1
    reward += s  # add this small reward (closer to the target => bigger reward)
    if diff < 0.01:  # if close enough, + reward it again (0.01 predefined threshold)
        reward += 1.  # 1 here means that (diff / 4.8) = 0, and its on the desired x
    if x <= left_most-.2 or x >= right_most+.2:  # if out of the range, penalize by -5
        reward -= 5.

    curr_speed = np.abs(curr_speed)
    velocities.append(curr_speed)   #keep track the car velocity
    normalized_vel = (curr_speed - min(velocities)) / (max(velocities) - min(velocities)) #normalize vel
    reward += normalized_vel
    #agent.desired_x = desired_x
    args.desired_x = desired_x

    return reward

# Policy training function
def train(agent, env, train_episodes, early_stop=True, render=False,silent=False, train_run_id=0):
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []
    agent.desired_x = 1.9
    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)
            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            observation, reward, done, info = env.step(action)

            # TODO: Task 1 - change the reward function
            reward = new_reward(observation,agent)

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if render:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 15 full episodes, assume it's learned
        # (in the default setting)
        if early_stop and np.mean(timestep_history[-15:]) == env._max_episode_steps:
            if not silent:
                print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         "reward": reward_history,
                         "mean_reward": average_reward_history})
    return data

# Function to test a trained policy
def test(agent, env, episodes, render=False):
    test_reward, test_len = 0, 0
    agent.desired_x = 1.9
    for ep in range(episodes):
        done = False
        observation = env.reset()
        steps = 0
        vel_history = []
        while not done:
            steps +=1
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action)
            # TODO: New reward function
            reward = new_reward(observation, agent)
            vel_history.append(observation[1])
            if render:
                env.render()
            test_reward += reward
            test_len += 1
        #print('ep:{},steps:{}'.format(ep,steps))
        plt.plot(vel_history, label='velocity (max={})'.format(round(max(abs(np.array(vel_history))),2)))
        plt.xlabel = "timesteps"
        plt.ylabel = "velocity"
        plt.legend()
        plt.title("Velocity history  (%s)" % args.env)
        plt.show()

    print('test_reward:{}, test_len:{} '.format(test_reward,test_len))
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)

# The main function
def main(args):
    # Create a Gym environment
    env = gym.make(args.env)

    # Exercise 1
    # TODO: For CartPole-v0 - maximum episode length
    env._max_episode_steps = 200 #used for training
    if args.test is not None:
        env._max_episode_steps = 500 #test it for 500 steps per episode

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    # Print some stuff
    print("Environment:", args.env)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        training_history = train(agent, env, args.train_episodes, False, args.render_training)

        # Save the model
        model_file = "%s_params.ai" % args.env
        torch.save(policy.state_dict(), model_file)
        print("Model saved to", model_file)

        # Plot rewards
        sns.lineplot(x="episode", y="reward", data=training_history)
        sns.lineplot(x="episode", y="mean_reward", data=training_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("Reward history (%s)" % args.env)
        plt.show()
        print("Training finished.")
    else:
        print("Loading model from", args.test, "...")
        state_dict = torch.load(args.test)
        policy.load_state_dict(state_dict)
        print("Testing...")
        test(agent, env, args.train_episodes, args.render_test)

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

