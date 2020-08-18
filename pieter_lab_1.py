###
# Group Members
# Clarise Poobalan:383321
# Byron Gomes:0709942R
# Shikash Algu:2373769
# Nicolaas Cawood:2376182
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

    v = np.zeros(env.observation_space.n)
    # Guaranteed to converge
    while True:
        delta = 0
        # Loop through each state
        for state_index in range(env.observation_space.n):
            new_val = 0
            for action_id, policy_prob in enumerate(policy[state_index]):
                # for all new states from each action
                for prob, next_state, reward, done in env.P[state_index][action_id]:
                    new_val += policy_prob * prob * (reward + discount_factor * v[next_state])
            delta = max(delta, abs(new_val - v[state_index]))
            # In-place update
            v[state_index] = new_val
            # Max change is smaller than threshold
        if delta < theta:
            break
    return v

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
            for prob, next_state, reward, done in env.P[state][action]:
                state_values[action] += prob * (reward + discount_factor * V[next_state])
        return state_values

    # Uniform random policy: Give equal prob to taking each action
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    while True:
        v = policy_evaluation_fn(env, policy, discount_factor)
        policy_stable = True

        for state_index in range(env.observation_space.n):
            #Get the greedy action's index
            old_action = np.argmax(policy[state_index])

            #Check values for actions from this state
            action_values = one_step_lookahead(state_index, v)
            #Best lookahead action
            best_action = np.argmax(action_values)

            #Update policy to have 0 for all actions except best one
            policy[state_index] = np.zeros(env.action_space.n)
            policy[state_index][best_action] = 1

            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            return policy, v


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
        raise NotImplementedError
        # Start with a random policy




    raise NotImplementedError


def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
        24], terminal_reward=0, step_reward=-1)
    state = env.reset()
    print("")
    env.render()
    print("")

    # Uniform random policy
    policy = np.ones([env.observation_space.n * env.action_space.n, env.action_space.n]) / env.action_space.n

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    # Evaluate random policy
    v = policy_evaluation(env, policy)

    print(v)

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)


    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")

    policy, v = policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0)

    print(np.argmax(policy,axis=1).reshape(env.shape))
    print()
    print(v.reshape(env.shape))

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    # TODO: use  value iteration to compute optimal policy and state values
    policy, v = [], []  # call value_iteration

    # TODO Print out best action for each state in grid shape

    # TODO: print state value for each state, as grid shape

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)


if __name__ == "__main__":
    main()
