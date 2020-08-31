###
# Group Members
# Clarise Poobalan : 383321
# Nicolaas Cawood: 2376182
# Shikash Algu: 2373769
# Byron Gomes: 0709942R
###

import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import matplotlib.pyplot as plt


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
    val_fn = np.zeros(env.observation_space.n)

    while True:
        delta = 0

        for state in range(env.observation_space.n):
            val = 0
            for action in range(env.action_space.n):
                #print(state, action)
                prob, next_state, reward, done = env.P[state][action][0]
                #print(prob, next_state, reward, done)
                val += policy[state, action] * prob * (reward + discount_factor* (val_fn[next_state]))

            delta = max(delta, abs(val_fn[state] - val))
            val_fn[state] = val

        #print("Delta: {}".format(delta))
        #print(val_fn)
        if delta < theta:
            break
    #print(val_fn.shape)
    val_fn = np.round(val_fn, 3)
    return val_fn


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
        state_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            # print(state, action)
            prob, next_state, reward, done = env.P[state][action][0]
            state_values[action] = prob * (reward + discount_factor * V[next_state])
        return state_values

    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    #count = 0
    while True:
        val_fn = policy_evaluation(env, policy, discount_factor)
        policy_stability = True
        #count += 1
        for state in range(env.observation_space.n):
            old_action = np.argmax(policy[state])

            action_vals = one_step_lookahead(state, val_fn)
            max_action = np.argmax(action_vals)

            if old_action != max_action:
                policy_stability = False

            policy[state] = np.zeros(env.action_space.n)
            policy[state][max_action] = 1

        if policy_stability == True:
            #print("Count {}".format(count))
            break
    val_fn = np.round(val_fn, 2)
    return policy, val_fn


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
        state_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            # print(state, action)
            prob, next_state, reward, done = env.P[state][action][0]
            state_values[action] = prob * (reward + discount_factor* V[next_state])
        return state_values

    val_fn = np.zeros(env.observation_space.n)
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    while True:
        delta = 0

        for state in range(env.observation_space.n):
            old_val = val_fn[state]

            action_val = one_step_lookahead(state,val_fn)

            max_action = np.argmax(action_val)

            val_fn[state] = np.max(action_val)

            delta = max(delta, abs(old_val - val_fn[state]))

            policy[state] = np.zeros(env.action_space.n)
            policy[state][max_action] = 1

        if delta < theta:
            break

    return policy, val_fn


def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[24], terminal_reward=0, step_reward=-1)
    state = env.reset() # Resets the environment to initial state
    print("Question 1:")
    print("Starting Position")
    env.render() # Visual the current state
    print("")

    #### generate uniform random policy ####
    random_policy = [state]  # to keep track of the states visited
    random_policy_actions = [] # to keep track of the actions taken
    iterations = 0 # to keep track how many iterations it took to randomly find the terminal state
    accumulated_reward = 0 # to keep track of the return
    terminal_state_reached = False # to check if we are at the terminal state

    while terminal_state_reached == False:
        action_to_take = env.action_space.sample()
        if action_to_take == 0:
            action = 'U'
        elif action_to_take == 1:
            action = 'R'
        elif action_to_take == 2:
            action = 'D'
        elif action_to_take == 3:
            action = 'L'

        random_policy_actions.append(action)

        state, reward, done, info = env.step(action_to_take)

        accumulated_reward += reward

        random_policy.append(state)

        terminal_state_reached = done

        iterations += 1

    #Question 1.2 Print random trajectory on a grid
    random_trajectory = np.full(env.shape, 'O')
    it = np.nditer(random_trajectory, flags=['multi_index'])
    for position_index in range(len(random_policy)-1):
        #print(position_index)
        it.iterindex = random_policy[position_index]
        x,y = it.multi_index
        if (position_index == (len(random_policy)-2)):
            random_trajectory[x,y] = 'T'
        else:
            random_trajectory[x,y] = random_policy_actions[position_index]

    print("States Visited:")
    print(random_policy)
    print("")
    print("Actions Taken")
    print(random_policy_actions)
    print("")
    print("Trajectory")
    print(random_trajectory)
    print("")

    #Question 2.1 Policy Evaluation
    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")
    policy = np.ones([env.observation_space.n * env.action_space.n, env.action_space.n]) / env.action_space.n
    # evaluate random policy
    v = policy_evaluation(env, policy)
    # print state value for each state, as grid shape
    print("Calculated Policy Evaluation:")
    print(v.reshape(env.shape))

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

    print("")
    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")
    # TODO: use  policy improvement to compute optimal policy and state values

    policy, v = policy_iteration(env) # call policy_iteration

    #Print out best action for each state in grid shape
    optimal_policy = np.full(env.observation_space.n, 'O')
    for state in range(env.observation_space.n):
        index = np.where(policy[state] == 1)
        if index[0] == 0:
            action = 'U'
        elif index[0] == 1:
            action = 'R'
        elif index[0] == 2:
            action = 'D'
        elif index[0] == 3:
            action = 'L'
        optimal_policy[state] = action
    optimal_policy[env.observation_space.n-1] = 'T'
    #optimal_policy[env.__terminal_states] = 'T'
    print("Best action for each state in grid:")
    print(optimal_policy.reshape(env.shape))

    #Print state value for each state, as grid shape
    print("")
    print("Calculated Optimal Policy Evaluation:")
    print(v.reshape(env.shape))

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("")
    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    #Use  value iteration to compute optimal policy and state values
    policy, v = value_iteration(env)# call value_iteration

    #Print out best action for each state in grid shape
    optimal_policy = np.full(env.observation_space.n, 'O')
    for state in range(env.observation_space.n):
        index = np.where(policy[state] == 1)
        if index[0] == 0:
            action = 'U'
        elif index[0] == 1:
            action = 'R'
        elif index[0] == 2:
            action = 'D'
        elif index[0] == 3:
            action = 'L'
        optimal_policy[state] = action
    optimal_policy[env.observation_space.n - 1] = 'T'
    print("Best action for each state in grid:")
    print(optimal_policy.reshape(env.shape))
    #Print state value for each state, as grid shape
    print("")
    print("Calculated Optimal Policy Evaluation:")
    print(v.reshape(env.shape))

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)




if __name__ == "__main__":
    main()

# Question4.2: Plot average running times for discount factors
gamma_arr = np.logspace(-0.2, 0, num=30)
policy_iter_times = []
value_iter_times = []
policy_setup = '''
from __main__ import policy_iteration
from environments.gridworld import GridworldEnv
env = GridworldEnv(shape=[5, 5], terminal_states=[24], terminal_reward=0, step_reward=-1)
'''
value_setup = '''
from __main__ import value_iteration
from environments.gridworld import GridworldEnv
env = GridworldEnv(shape=[5, 5], terminal_states=[24], terminal_reward=0, step_reward=-1)
'''
for gamma in gamma_arr:
    policy_iter_times.append(timeit.timeit(setup = policy_setup, stmt = "policy_iteration(env,discount_factor=" + str(gamma) + ")", number = 10)/10)
    value_iter_times.append(timeit.timeit(setup = value_setup, stmt = "value_iteration(env,discount_factor=" + str(gamma) + ")", number=10)/10)

plt.plot(gamma_arr, policy_iter_times, color="purple", label='Policy iteration')
plt.plot(gamma_arr, value_iter_times, color="#42e3f5", label='Value iteration')
plt.title('Average Runtimes for Policy and Value Iteration over various Discount Rates')
plt.legend()
plt.xlabel('Discount rate')
plt.ylabel('Average time (seconds)')
plt.show()