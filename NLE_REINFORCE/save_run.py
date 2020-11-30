import numpy as np
import gym
import nle
import random
import csv
import os
from Agent_1 import *

def run_episode(env,seed,episode_num):
    # create instance of MyAgent
    #from Agent_1 import MyAgent, SimplePolicy
    agent = MyAgent(env.observation_space, env.action_space)

    done = False
    episode_return = 0.0
    state = env.reset()
    stats_list = [np.zeros(25),np.zeros(25)]
    steps = 0
    max_depth = 0
    while not done:
        # pass state to agent and let agent decide action
        action = agent.act(state)
        new_state, reward, done, _ = env.step(action)
        steps += 1
        episode_return += reward
        # Update list of episode stats
        stats_list = update_stat_list(stats_list,state)
        if stats_list[0][12] > max_depth:
            max_depth = stats_list[0][12]
        # Check if done
        if done:
            row = get_stats(stats_list[0],steps,max_depth,seed,episode_num)

        state = new_state
    return episode_return,row

def get_stats(stats,steps,max_depth,seed,episode_num):
    return [1, # end_status, 1 = episode ended correctly. This is assumed here
        stats[9], # This is the ingame score, which may differ from the total returns
        stats[20], # time
        steps,stats[10], # steps in this episode
        stats[19], # health points
        stats[18], # experience
        stats[13], # experience level
        stats[21], # gold
        "UNK", # name of killer. env doesn't store this, so just left as unknown
        max_depth, # The furthest depth the agent went this episode
        episode_num, # The number of the episode
        seed, # The env seed
        "episode_"+str(episode_num)+".ttyrec"] # the name of the corresponding ttyrec file in stats.zip

def update_stat_list(stats_list,state):
    # Hacky stuff to get the right stats vector
    stats_list[0] = stats_list[1]
    stats_list[1] = list(state['blstats'])
    return stats_list



if __name__ == '__main__':
    # Directory
    dir = os.getcwd()

    # Seed
    seeds = [1,2,3,4,5]

    # Initialise environment
    env = gym.make("NetHackScore-v0")

    # Generate CSV
    stats_list = [['end_status','score','time','steps','hp','exp','exp_lev',
    'gold','hunger','killer_name','deepest_lev','episode','seeds','ttyrec']]

    # Run one episode
    rewards = []
    episode_num = 1
    for seed in seeds:
        env.seed(seed)
        seed_rewards = []
        reward,row = run_episode(env,seed,episode_num)
        stats_list.append(row)
        episode_num += 1
        rewards.append(reward)
    # Close environment and print average reward
    env.close()
    print("Average Reward: %f" %(np.mean(rewards)))

    # Write to csv
    # NOTE: Unfortunately, you will need to either rename each ttyrec file to
    #   match stats.csv or change the entry in stats.csv to point to the
    #   appropriate ttyrec file
    os.chdir(dir)
    file = open('stats.csv','w+',newline='')
    with file:
        writer = csv.writer(file)
        writer.writerows(stats_list)
    file.close()
