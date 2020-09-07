###
# Group Members
# Clarise Poobalan : 383321
# Nicolaas Cawood: 2376182
# Shikash Algu: 2373769
# Byron Gomes: 0709942R
###


from collections import defaultdict
from collections import namedtuple
import sys
import itertools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


# Helper functions
##################
def blackjack_plot_value_function(V, title="Value Function", suptitle="MC Blackjack-v0"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(
        lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(
        lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))
    fig = plt.figure(figsize=(20, 15))
    st = fig.suptitle(suptitle, fontsize="large")

    def plot_surface(X, Y, Z, title, plot_positon=111):
        ax = fig.add_subplot(plot_positon, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title), 211)
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title), 212)
    fig.tight_layout()

    # shift subplots down:
    st.set_y(0.99)
    fig.subplots_adjust(top=0.93)

    fig.savefig(title + "_blackjack_plot_value_function.png")
    plt.show()


def td_plot_episode_stats(stats, algor_name="", smoothing_window=10):
    # Plot the episode length over time

    fig = plt.figure(figsize=(10, 10))
    st = fig.suptitle(algor_name, fontsize="large")

    # fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(311)
    ax1.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    ax1.set_title("Episode Length over Time")

    # Plot the episode reward over time
    # fig2 = plt.figure(figsize=(10, 5))
    ax2 = fig.add_subplot(312)
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    ax2.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    ax2.set_title("Episode Reward over Time (Smoothed over window size {})".format(
        smoothing_window))

    # Plot time steps and episode number
    # fig3 = plt.figure(figsize=(10, 5))
    ax3 = fig.add_subplot(313)
    ax3.plot(np.cumsum(stats.episode_lengths),
             np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    ax3.set_title("Episode per time step")

    fig.tight_layout()

    # shift subplots down:
    st.set_y(0.99)
    fig.subplots_adjust(top=0.93)

    fig.savefig(algor_name + "_td_plot_episode_stats.png")
    plt.show(block=False)


def td_plot_values(Q, algor_name="", state_shape=((4, 12))):
    """ helper method to plot a heat map of the states """

    values = np.zeros((4 * 12))
    max_a = [0 for _ in range(values.shape[0])]
    for key, value in Q.items():
        values[key] = max(value)
        max_a[key] = int(argmax(value))

    def optimal_move(i, j):
        left, right, down, up = (
                                    i, max(j - 1, 0)), (i, min(11, j + 1)), (min(3, i + 1), j), (max(0, i - 1), j)
        arr = np.array([values[up], values[right], values[down], values[left]])
        if i == 2 and j != 11:
            arr[2] = -9999
        if i == 0:
            arr[0] = -999
        if j == 0:
            arr[3] = -999
        if j == 11:
            arr[1] = -999
        return argmax(arr)

    # reshape the state-value function
    values = np.reshape(values, state_shape)
    # plot the state-value function
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    im = ax.imshow(values, cmap=matplotlib.cm.coolwarm)
    arrows = ['↑', '→', '↓', '←']
    index = 0
    for (j, i), label in np.ndenumerate(values):
        ax.text(i, j, np.round(label, 3),
                ha='center', va='center', fontsize=12)
        if j != 3 or i == 0:
            ax.text(i, j + 0.4, arrows[optimal_move(j, i)],
                    ha='center', va='center', fontsize=12, color='black')
        index += 1
    plt.tick_params(bottom=False, left=False,
                    labelbottom=False, labelleft=False)
    plt.title(algor_name + ': State-Value Function')
    fig.savefig(algor_name + "_td_plot_values.png")
    plt.show(block=False)


##################

def blackjack_sample_policy(observation):
    """
    A policy that sticks if the player score is >= 20 and hits otherwise.
    """
    player_score, _, _ = observation
    if player_score >= 20:
        # Stick
        return 0
    # Hit otherwise
    return 1


def mc_prediction(policy, env, num_episodes, discount_factor=1.0, max_steps_per_episode=9999, print_=False):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        print_: print every num of episodes - don't print anything if False

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    # Implements First-visit MC prediction from Sutton & Barto (Pg 92)
    v = defaultdict(float)
    returns = defaultdict(list)
    for episode_number in range(num_episodes):
        visted_states = {}
        current_state = env.reset()
        # Generate an episode following policy
        episode = []
        for step in range(max_steps_per_episode):
            action = policy(current_state)
            next_state, reward, done, info = env.step(action)
            episode.append((current_state, action, reward))
            # Busted or won
            if done:
                break
            current_state = next_state
        G = 0
        # Loop for each step of episode
        for step in reversed(episode):
            state, action, reward = step
            G = G * discount_factor + reward
            # First-visit only
            if state not in visted_states:
                returns[state].append(G)
                v[state] = sum(returns[state]) / len(returns[state])
                visted_states[state] = True
        if print_:
            if (episode_number + 1) % 1000 == 0:
                print('Episodes complete: {:d}'.format(episode_number + 1))
    return v


def argmax(numpy_array):
    """ argmax implementation that chooses randomly between ties """
    return np.random.choice(np.flatnonzero(numpy_array == numpy_array.max()))


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        probs = np.ones(nA) * epsilon / nA
        greedy_action = argmax(Q[observation])
        probs[greedy_action] += 1 - epsilon
        return probs

    return policy_fn


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1, max_steps_per_episode=100,
                              print_=False):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        print_: print every num of episodes - don't print anything if False
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    policy = make_epsilon_greedy_policy(Q, epsilon, nA)
    returns = defaultdict(lambda: defaultdict(list))
    for episode_number in range(num_episodes):
        visted_state_actions = {}
        current_state = env.reset()
        # Generate an episode following policy
        episode = []
        for step in range(max_steps_per_episode):
            action = np.random.choice(np.arange(nA), p=policy(current_state))
            next_state, reward, done, info = env.step(action)
            episode.append((current_state, action, reward))
            # Busted or won
            if done:
                break
            current_state = next_state
        G = 0
        # Loop for each step of episode
        for step in reversed(episode):
            state, action, reward = step
            G = G * discount_factor + reward
            # If S,A pair has not been visited
            if not (state, action) in visted_state_actions:
                returns[state][action].append(G)
                Q[state][action] = sum(returns[state][action]) / len(returns[state][action])
                visted_state_actions[(state, action)] = True
        if print_:
            if (episode_number + 1) % 1000 == 0:
                print('Episodes complete: {:d}'.format(episode_number + 1))
    return Q, policy


def SARSA(env, num_episodes, discount_factor=1.0, epsilon=0.1, alpha=0.5, print_=False):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        print_: print every num of episodes - don't print anything if False

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    nA = env.action_space.n
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Update statistics after getting a reward - use within loop, call the following lines
    # stats.episode_rewards[i_episode] += reward
    # stats.episode_lengths[i_episode] = t

    for episode_number in range(num_episodes):
        current_state = env.reset()
        action = np.random.choice(np.arange(nA), p=policy(current_state))
        t = 0
        while True:
            t += 1
            # Take action and observe
            next_state, reward, done, info = env.step(action)
            # Next state's action
            next_action = np.random.choice(np.arange(nA), p=policy(next_state))
            Q[current_state][action] += alpha * (reward + discount_factor * (Q[next_state][next_action] - Q[current_state][action]))
            current_state = next_state
            action = next_action
            # Update statistics
            stats.episode_rewards[episode_number] += reward
            stats.episode_lengths[episode_number] = t
            if done:
                break
    return Q, stats

def q_learning(env, num_episodes, discount_factor=1.0, epsilon=0.05, alpha=0.5, print_=False):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        print_: print every num of episodes - don't print anything if False

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    nA = env.action_space.n
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # Keeps track of useful statistics
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # Update statistics after getting a reward - use within loop, call the following lines
    # stats.episode_rewards[i_episode] += reward
    # stats.episode_lengths[i_episode] = t
    for episode_number in range(num_episodes):
        current_state = env.reset()
        t = 0
        while True:
            t += 1
            # Take action and observe
            action = np.random.choice(np.arange(nA), p=policy(current_state))
            next_state, reward, done, info = env.step(action)
            Q[current_state][action] += alpha * (reward + discount_factor * (max(Q[next_state]) - Q[current_state][action]))
            current_state = next_state
            # Update statistics
            stats.episode_rewards[episode_number] += reward
            stats.episode_lengths[episode_number] = t
            if done:
                break
    return Q, stats

def run_mc():
    # Exploring the BlackjackEnv
    # create env from https://gym.openai.com/envs/Blackjack-v0/
    blackjack_env = gym.make('Blackjack-v0')
    # let's see what's hidden inside this Object
    print(vars(blackjack_env))

    # how big is the action space ?2,  The action are  hit=1, stick=0
    print('number of actions:', blackjack_env.action_space.n)
    # let see the observation space - The observation of a 3-tuple of: the players current sum,
    #     the dealer's one showing card (1-10 where 1 is ace),
    #     and whether or not the player holds a usable ace (0 or 1).
    print('observation_space', blackjack_env.observation_space)
    observation = blackjack_env.reset()
    print('observation:', observation)
    # let's sample a random action
    random_action = blackjack_env.action_space.sample()
    print('random action:', random_action)
    # let's simulate one action
    next_observation, reward, done, _ = blackjack_env.step(random_action)
    print('next_observation:', next_observation)
    print('reward:', reward)
    print('done:', done)

    print("\nmc_prediction 10,000_steps\n")
    values_10k = mc_prediction(
        blackjack_sample_policy, blackjack_env, num_episodes=10000, print_=True)
    blackjack_plot_value_function(values_10k, title="10,000 Steps")

    print("\nmc_prediction 500,000_steps\n")
    values_500k = mc_prediction(
        blackjack_sample_policy, blackjack_env, num_episodes=500000, print_=True)
    blackjack_plot_value_function(values_500k, title="500,000 Steps")

    print("\nmc_control_epsilon_greedy\n")
    Q, policy = mc_control_epsilon_greedy(
        blackjack_env, num_episodes=500000, epsilon=0.1, print_=True)
    # For plotting: Create value function from action-value function
    # by picking the best action at each state
    values = defaultdict(float)
    for state, actions in Q.items():
        action_value = np.max(actions)
        values[state] = action_value
    blackjack_plot_value_function(values, title="Optimal Value Function")


def run_td():
    num_episodes = 1000
    discount_factor = 1.0
    epsilon = 0.1
    alpha = 0.5

    # create env : https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py
    cliffwalking_env = gym.make('CliffWalking-v0')
    cliffwalking_env.render()

    print('SARSA\n')
    sarsa_q_values, stats_sarsa = SARSA(cliffwalking_env, num_episodes=num_episodes,
                                        discount_factor=discount_factor, epsilon=epsilon,
                                        alpha=alpha, print_=True)
    td_plot_episode_stats(stats_sarsa, "SARSA")
    td_plot_values(sarsa_q_values, "SARSA")
    print('')

    print('Q learning\n')
    ql_q_values, stats_q_learning = q_learning(cliffwalking_env, num_episodes=num_episodes,
                                               discount_factor=discount_factor,
                                              epsilon=epsilon,
                                               alpha=alpha, print_=True)
    td_plot_episode_stats(stats_q_learning, "Q learning")
    td_plot_values(ql_q_values, "Q learning")
    print('')


if __name__ == '__main__':
    run_mc()
    run_td()
