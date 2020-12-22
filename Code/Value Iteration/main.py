# Copyright 2020 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#
#############################################################


import numpy as np
from time import sleep
from sailing import SailingGridworld
#from assignment2.sailing import SailingGridworld

epsilon = 10e-4  # TODO: Use this criteria for Task 3

# Set up the environment
env = SailingGridworld(rock_penalty=-2.)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)
gamma = 0.9

def Task1(value_est,policy):
    prev_value_est = value_est.copy()
    for iter in range(100):
        env.clear_text()
        #get all states
        for i in range(env.w):
            for j in range(env.h):
                history = []
                #compute the sum for all s'
                for tran in env.transitions[i,j]:
                    sum = 0.
                    for s_,r,done,p in tran:
                        sum += p * (r + (gamma * value_est[s_[0],s_[1]] if not done else 0))

                    history.append(sum)

                #update V(s), and pi(s)
                value_est[i,j] = np.max(history)
                policy[i,j] = np.argmax(history)

        if ((value_est - prev_value_est) < epsilon).all():
            print('Early stopping')
            break
        else:
            prev_value_est = value_est.copy()

    return value_est,policy

if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))

    value_est, policy = Task1(value_est,policy)

    # Show the values and the policy
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    sleep(1)

    # Save the state values and the policy
    fnames = "values.npy", "policy.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)

    # Run a single episode
    # TODO: Run multiple episodes and compute the discounted returns (Task 4)
    history_G = []
    for episod in range(1000):
        state = env.reset()

        done = False
        G, k = 0,0
        while not done:
            action = policy[state]
            state, reward, done, _ = env.step(action)
            G += (gamma**k)*reward
            k+=1
            #env.render()
            #sleep(0.1)

        history_G.append(G)
    print('Avg:{}, Std:{}'.format(np.average(history_G), np.std(history_G)))

