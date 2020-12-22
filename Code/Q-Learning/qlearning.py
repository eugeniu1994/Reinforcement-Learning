import gym
import numpy as np
from matplotlib import pyplot as plt
import seaborn as seab
import sys

np.random.seed(123)
env = gym.make('CartPole-v0')
#env = gym.make('LunarLander-v2')
env.seed(321)

episodes = 20000

test_episodes = 10
num_of_actions = 2

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# For LunarLander, use the following values:
#         [  x     y     xdot ydot theta thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]
x_min, x_max = -1.2, 1.2
y_min, y_max = -.3, 1.2
xdot_min, xdot_max = -2.4, 2.4
ydot_min, ydot_max = -2., 2.
theta_min, theta_max = -6.28, 6.28
thetadot_min, thetadot_max = -8., 8.
cl_min, cl_max = 0, 1
cr_min, cr_max = 0, 1

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = 0  # TODO: Set the correct value.
a = int((target_eps*2000)/1-target_eps)
print('a ',a)
initial_q = 0  #
#initial_q = 50  # T3: Set to 50

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

#used for LunaLander
'''x_grid = np.linspace(x_min, x_max, discr)
y_grid = np.linspace(y_min, y_max, discr)
xdot_grid = np.linspace(xdot_min, xdot_max, discr)
ydot_grid = np.linspace(ydot_min, ydot_max, discr)
theta_grid = np.linspace(theta_min, theta_max, discr)
thetadot_grid = np.linspace(thetadot_min, thetadot_max, discr)
cl_grid = np.linspace(cl_min, cl_max, 2)
cr_grid = np.linspace(cr_min, cr_max, 2)'''

q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q #CartPole
#q_grid = np.zeros((discr, discr, discr, discr, discr, discr, 2, 2, num_of_actions)) + initial_q  #LunarLander

def find_nearest(array, value):
    return np.argmin(np.abs(array - value))

#used for Cartpole
def get_cell_index(state):
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av

#used for LunarLander
'''def get_cell_index(state):
    x = find_nearest(x_grid, state[0])
    y = find_nearest(y_grid, state[1])
    xdot = find_nearest(xdot_grid, state[2])
    ydot = find_nearest(ydot_grid, state[3])
    theta = find_nearest(theta_grid, state[4])
    thetadot = find_nearest(thetadot_grid, state[5])
    cl = find_nearest(cl_grid, state[6])
    cr = find_nearest(cr_grid, state[7])
    return x, y, xdot, ydot, theta, thetadot, cl, cr'''

def get_action(state, q_values, greedy=False):
    state = get_cell_index(state) #convert state from continuous to discrete
    # TODO: Implement epsilon-greedy
    if greedy or np.random.random() >= epsilon:
        #greedy is true -> we test the agent, or prob 1-epsilon
        action = np.argmax(q_values[state])
    else: #random action with prob epsilon
        action = int(np.random.random() * num_of_actions)
        
    return action

def update_q_value(old_state, action, new_state, reward, done, q_array, steps, on_policy=True):
    # TODO: Implement Q-value update
    s = get_cell_index(old_state)   #current state
    s_ = get_cell_index(new_state)  #next state

    #if done: #for LunarLander
    if done and steps < 199: #Q(terminal,:) is 0  - for Cartpole
        q_grid[s][:] = 0.
    else: #Q update formula
        if on_policy:
            a_ = get_action(new_state, q_grid, greedy=True)
            target = reward + (gamma * q_grid[s_][a_]) - q_grid[s][action]
        else: #off_policy
            target = reward + (gamma * max(q_grid[s_])) - q_grid[s][action]

        q_grid[s][action] += alpha * target

#values heatmap  before the training
values = np.amax(q_grid, axis=4) #
seab.heatmap(np.mean(values, axis=(1, 3)),xticklabels=True, yticklabels=True)
plt.xlabel("X")
plt.ylabel("theta")
plt.title('heatmap before the training')
plt.show()

# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    #epsilon = 0  # T3: Set to 0
    #epsilon = .2 #T1 constant
    epsilon = a/(a+ep) # T1: GLIE
    while not done:
        action = get_action(state, q_grid, greedy=test)
        new_state, reward, done, _ = env.step(action)
        if not test:
            update_q_value(state, action, new_state, reward, done, q_grid, steps)
        else:
            env.render()
        state = new_state
        steps += 1

    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))

    if ep == 1:
        values = np.amax(q_grid, axis=4)  # Task 2
        seab.heatmap(np.mean(values, axis=(1, 3)), xticklabels=True, yticklabels=True)
        plt.xlabel("X")
        plt.ylabel("theta")
        plt.title('heatmap after one episode')
        plt.show()
    elif ep == int(episodes/2):
        values = np.amax(q_grid, axis=4)  # Task 2
        seab.heatmap(np.mean(values, axis=(1, 3)), xticklabels=True, yticklabels=True)
        plt.xlabel("X")
        plt.ylabel("theta")
        plt.title('heatmap halfway through the training')
        plt.show()

# Save the Q-value array
#np.save("q_values.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# Calculate the value function
values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
values = np.amax(q_grid, axis=4) #Task 2
#np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY

# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib
seab.heatmap(np.mean(values, axis=(1, 3)),xticklabels=True, yticklabels=True)
plt.xlabel("X")
plt.ylabel("theta")
plt.title('heatmap after the training')
plt.show()

# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()

sys.exit()