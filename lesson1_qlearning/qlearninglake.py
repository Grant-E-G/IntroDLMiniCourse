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

    total_episodes = 30000        # Total episodes
    learning_rate = 0.8           # Learning rate
    max_steps = 99                # Max steps per episode
    gamma = 0.95                  # Discounting rate

    # Exploration parameters
    epsilon = 1.0                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability
    decay_rate = 0.003            # Exponential decay rate for exploration prob

    """
	Code block copied from Thomas Simonini tutoral on RL
	Unique code below
	"""
    reward_trials = 2000  # when report value the number of trials to average over
    
    total_reward = 0.0
    for x in range(total_episodes):
        #Reset and grab the first observation
        state = env.reset()
        #Set epsilon for our epsilon greedy search
        if epsilon < min_epsilon:
            epsilon = min_epsilon
        else:
            epsilon = max_epsilon * np.exp(-x * decay_rate)
        for stepcount in range(max_steps):
            #The exploration case
            if epsilon >= random.uniform(0, 1):
                action = random.randrange(0, action_size)
                new_state, reward, done, info = env.step(action)
                total_reward += reward
                qtable[state][action] = updateqtable(
                    state, new_state, action, reward, qtable, learning_rate, gamma)
                state = new_state
            #The exploit case
            else:
                action = np.argmax(qtable[state])
                new_state, reward, done, info = env.step(action)
                total_reward += reward
                qtable[state][action] = updateqtable(
                    state, new_state, action, reward, qtable, learning_rate, gamma)
                state = new_state
            if done:
                break

    print(qtable)
    print('Mean post learning reward is')
    print(meanreward(env, qtable, max_steps, reward_trials))
    print('Mean reward over trials')
    print(total_reward/total_episodes)
    
    pass


def updateqtable(state, new_state, action, reward, qtable, learning_rate, gamma):
    maxq = np.amax(qtable[new_state])
    deltaq = learning_rate * (reward + gamma * maxq - qtable[state][action])
    return qtable[state][action] + deltaq

#Run the algorithm after training to examine our average reward.
def meanreward(env, qtable, max_steps, reward_trials):
    mean_reward = 0.0
    for x in range(reward_trials):
        reward_total = 0.0
        state = env.reset()
        for stepcount in range(max_steps):
            action = np.argmax(qtable[state])
            state, reward, done, info = env.step(action)
            reward_total += reward
            if done:
                break
        mean_reward += reward_total / reward_trials

    return mean_reward


if __name__ == '__main__':
    main()
