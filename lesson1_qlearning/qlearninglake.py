"""
Basic q-learning bot
"""
import numpy as np
import gym
import random


def main():

    env = gym.make('FrozenLake-v0')
    action_size = env.action_space.n
    state_size = env.observation_space.n
    qtable = np.zeros((state_size, action_size))

    total_episodes = 15000        # Total episodes
    learning_rate = 0.8           # Learning rate
    max_steps = 99                # Max steps per episode
    gamma = 0.95                  # Discounting rate

    # Exploration parameters
    epsilon = 1.0                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability
    decay_rate = 0.005            # Exponential decay rate for exploration prob

    """
    Code block copied from Thomas Simonini tutoral on RL
    Unique code below
    """
    post_training_trials = 2000  # Number of post traing experimental trials

    total_reward = 0.0
    for x in range(total_episodes):
        # Reset and grab the first observation
        state = env.reset()
        # Set epsilon for our epsilon greedy search
        if epsilon < min_epsilon:
            epsilon = min_epsilon
        else:
            epsilon = max_epsilon * np.exp(-x * decay_rate)
        for stepcount in range(max_steps):
            # The exploration case
            if epsilon >= random.uniform(0, 1):
                action = action = env.action_space.sample()
            # The exploit case
            else:
                # Random tie breaking
                action = np.random.choice(np.flatnonzero(
                    qtable[state] == qtable[state].max()))
            # perform the action
            new_state, reward, done, info = env.step(action)
            total_reward += reward
            qtable[state][action] = updateqtable(
                state, new_state, action, reward, qtable, learning_rate, gamma)
            state = new_state
            if done:
                break
    # Output data + stats
    print(qtable)
    print('Average reward during learning')
    print(total_reward / total_episodes)
    print('Average reward post learning over ' +
          str(post_training_trials) + ' trials')
    data = qtablestats(env, qtable, max_steps, post_training_trials)
    print(np.mean(data))
    print('Varance of post learning reward over ' +
          str(post_training_trials) + ' trials')
    print(np.var(data))

    pass


def updateqtable(state, new_state, action, reward, qtable, learning_rate, gamma):
    maxq = np.amax(qtable[new_state])
    deltaq = learning_rate * (reward + gamma * maxq - qtable[state][action])
    return qtable[state][action] + deltaq

# Run the algorithm after training to examine our results.


def qtablestats(env, qtable, max_steps, post_training_trials):
    data = np.array([0])
    for x in range(post_training_trials):
        state = env.reset()
        total_reward = 0.0
        for stepcount in range(max_steps):
            action = np.argmax(qtable[state])
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                data = np.append(data, total_reward)
                break
    # Hack here to drop the first dummy value
    return data[1:]


if __name__ == '__main__':
    main()
