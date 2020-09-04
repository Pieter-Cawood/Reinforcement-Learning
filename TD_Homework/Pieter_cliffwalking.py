###
# Group Members
# Shikash Algu: 2373769
# Byron Gomes: 0709942R
# Clarise Poobalan: 383321
# Nicolaas Cawood: 2376182
#
#
#
#
#                 SARSA(LAMBDA)
# The idea in Sarsa(lmbda) is to apply the TD(lmbda)
# prediction method to state-action pairs rather than to states
###

import numpy as np
import matplotlib.pyplot as plt
import gym
import seaborn as sns
import cv2


def get_eps_greedy_probs(env, q, epsilon):
    probs = np.ones(env.nA) * epsilon / env.nA

    mask = q == np.max(q)
    greedy_choices = np.argwhere(mask)
    greedy_count = len(greedy_choices)

    probs[greedy_choices] = \
        (1 - ((env.nA - greedy_count) * epsilon / env.nA)) / greedy_count
    return probs


def sarsa_lmda(id, env, lmbda=0.5, n_episodes=500, epsilon=0.1, alpha=0.5, discount_rate=1):
    q = {}  # QValues
    e = {}  # Trace
    episodes = []
    fig = plt.figure()
    # Initialise Q & e dicts
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            q[state] = np.zeros(env.nA)
            e[state] = np.zeros(env.nA)

    # Loop over episodes
    for i_episode in range(1, n_episodes + 1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rInstance {}, Episode {}/{}".format(id, i_episode, n_episodes), end="")

        # Get state
        current_state = env.reset()

        # Current state, next action
        action_probabilities = get_eps_greedy_probs(env, q[current_state], epsilon)
        current_action = np.random.choice(
            np.arange(env.nA),
            p=action_probabilities)

        while True:
            # Take action and observe
            next_state, reward, done, info = env.step(current_action)

            # Next state, next action
            action_probabilities = get_eps_greedy_probs(env, q[next_state], epsilon)
            next_action = np.random.choice(
                np.arange(env.nA),
                p=action_probabilities)

            # TD-error
            delta = reward + discount_rate * q[next_state][next_action] - q[current_state][current_action]

            # Trace
            e[current_state][current_action] = e[current_state][current_action] + 1

            # Update Q & e
            for state in range(env.observation_space.n):
                for action in range(env.action_space.n):
                    q[current_state][current_action] = q[current_state][current_action] + \
                                                       alpha * delta * e[current_state][current_action]
                    e[current_state][current_action] = discount_rate * lmbda * e[current_state][current_action]

            # Update environment
            current_state = next_state.copy()
            current_action = next_action.copy()
            ims = []
            # Terminal reached
            if done:
                policy = np.array([np.argmax(q[key]) for key in np.arange(48)]).reshape(4, 12)
                hm_figure = sns.heatmap(policy).get_figure()
                file_name = "resources/" + str(id) + "episode" + str(i_episode)
                hm_figure.savefig(file_name)
                episodes.append(file_name)
                hm_figure.clear()
                plt.cla()
                break
    return episodes

def main():
    """
    Call aunty Sarsa for a video shoot

    """
    episode_list = []
    episodes = []
    lambdas = [0, 0.3, 0.5, 0.7, 0.9]
    i = 0
    # Record Sarsa episodes for different lambdas
    for lmbda in lambdas:
        i += 1
        env = gym.make('CliffWalking-v0')
        episode_list.append(sarsa_lmda(i, env))

    # 4 frames/sec video
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, (1280, 1280))
    # Make a video with subplots
    for i_episode in range(len(episode_list[0])):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rVideo episode {}/{}".format(i_episode, 500), end="")

        fig = plt.figure(figsize=(45, 45))
        # Make subplots for each lambda value
        for lmda_epi in range(len(episode_list)):
            file_name = episode_list[lmda_epi][i_episode]
            image = cv2.imread(file_name + ".PNG")
            ax = fig.add_subplot(1, 5, lmda_epi + 1)
            ax.set_title("Lamda (" + str(lambdas[lmda_epi]) + ")", {'fontsize': 28})
            plt.imshow(image)
            del image
        # Combine subplots to single fig
        fig.savefig("resources/combined" + str(i_episode))
        fig.clear()
        img = cv2.imread("resources/combined" + str(i_episode) + ".PNG")
        img = cv2.resize(img, (1280, 1280))
        # Use combined fig as video frame
        out.write(img)
        plt.close(fig)
        plt.cla()
    out.release()

if __name__ == "__main__":
    main()
