###
# Group Members
# Clarise Poobalan : 383321
# Nicolaas Cawood : 2376182
# Shikash Algu : 2373769
# Byron Gomes : 0709942R
###

import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import matplotlib.pyplot as plt
from random import randrange



class Random:
    
    def __init__(self, world, state):
        self.world = world
        self.actions =[]
        self.states = [state]
    
    def play(self):
        action_dict = {0:'U', 1:'R', 2:'D', 3:'L'}
        done = False
        while not done:
            a = randrange(4)
            next_state, reward, done, n = self.world.step(a)
            self.actions.append(action_dict[a])
            self.states.append(next_state)
        self.plot_path()
    
    def plot_path(self):
        shape = self.world.shape
        game = np.array([['o']*shape[0]]*shape[1])
        for a, i in zip(self.actions, self.states[:-1]):
            x = i//shape[0]
            y = i - x*shape[0]
            if x == shape[0]-1 and y == shape[1]-1:
                game[x][y] = 'X'
            else:
                game[x][y] = a
        
        game_list = list(game)
        for g in game_list:
            print(*g, sep="  ")
        print("")
        


def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:

        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """
    
    V = np.zeros(env.observation_space.n)
    diff = 1
    while diff >= theta:
        diff = 0
        for state in range(env.observation_space.n):
            v = V[state]
            new_v = []
            for action in range(env.action_space.n):
                prob, next_state, reward, done = env.P[state][action][0]
                new_v.append(policy[state][action]*prob*(reward + discount_factor*V[next_state]))
            V[state] = sum(new_v)
            diff = max([diff, abs(v - V[state])])
    
    return V


def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        values = np.zeros(env.action_space.n)
        
        for action in range(env.action_space.n):
            prob, next_state, reward, done = env.P[state][action][0]
            values[action] = prob*(reward + discount_factor*V[next_state])
        
        actions = np.array([int(a) for a in values==values.max()])
        new_actions = actions/actions.sum()
        
        return new_actions
    
    
    # initialize arbitrary policy
    policy = np.ones([env.observation_space.n,
                      env.action_space.n])/env.action_space.n
    unstable = True
    while unstable:
        unstable = False
        V = policy_evaluation(env, policy, discount_factor)
        
        for state in range(env.observation_space.n):
            old_action = policy[state].copy()
            policy[state]  = one_step_lookahead(state, V)
            
            if not all(old_action==policy[state]):
                unstable = True

    return policy, V


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """
    
    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            prob, next_state, reward, done = env.P[state][action][0]
            values[action] = prob*(reward + discount_factor*V[next_state])
                
        return values

    V = np.zeros(env.observation_space.n)
    diff = 1
    while diff >= theta:
        diff = 0
        for state in range(env.observation_space.n):
            v = V[state]
            V[state] = one_step_lookahead(state, V).max()
            diff = max([diff, abs(v - V[state])])
    
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        policy[state] = np.argmax(one_step_lookahead(state, V))
    
    return policy, V


def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    state = env.reset()
    print("Initial game state")
    env.render()
    print("")
    
    # Q1.1 Play a round using the random policy and print trajectory
    print("Sample trajecory from a uniform random policy")
    rand_player = Random(env, state)
    rand_player.play()
    state = env.reset()

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    #Q2.1 evaluate random policy
    rand_policy = np.ones([env.observation_space.n,
                           env.action_space.n])/env.action_space.n
    
    v = np.floor(policy_evaluation(env, rand_policy)*100)/100

    # print state value for each state, as grid shape
    print("Converged value function for uniform random poliy")
    pv = np.around(v, 2).reshape(env.shape)
    print(pv)
    print("")

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")
    #Q3.1 use  policy improvement to compute optimal policy and state values
    policy, v = policy_iteration(env, policy_evaluation) # call policy_iteration
    
    # Print out best action for each state in grid shape
    action_dict = {0:'U', 1:'R', 2:'D', 3:'L'}
    best_move = np.array([action_dict[np.argmax(p)] for p in policy]).reshape(env.shape)
    print("Best action for each state")
    print(best_move)
    print("")
    # print state value for each state, as grid shape
    print("Value function for each state")
    print(v.reshape(env.shape))
    print("")
    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    #Q4.1.1 use  value iteration to compute optimal policy and state values
    policy, v = value_iteration(env)  # call value_iteration

    # Print out best action for each state in grid shape
    best_move = np.array([action_dict[p] for p in policy]).reshape(env.shape)
    print("Best action for each state")
    print(best_move)
    print("")
    # print state value for each state, as grid shape
    print("Value function for each state")
    print(v.reshape(env.shape))
    print("")
    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)
    
    #Q4.1.2 time the policy iteration and value iteration code
    gamma = np.logspace(-0.2, 0, num=30)
    
    pi_time = []
    vi_time = []
    for df in gamma:
        pi_time.append(timeit.timeit(lambda: policy_iteration(env, policy_evaluation, discount_factor=df), number=10)/10)
        vi_time.append(timeit.timeit(lambda: value_iteration(env, discount_factor=df), number=10)/10)
    
    plt.figure()
    plt.plot(gamma, pi_time, label="Policy Iteration")
    plt.plot(gamma, vi_time, label='Value Iteration')
    plt.xlabel('Discount Factor')
    plt.ylabel('Time (seconds)')
    plt.title('Average Execution Times for Variying Discount Factors')
    plt.legend()

if __name__ == "__main__":
    main()
