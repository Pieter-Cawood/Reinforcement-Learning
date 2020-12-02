import gym
import nle
import numpy as np
from MyAgent import MyAgent

env = gym.make("NetHackScore-v0")

agent = MyAgent(env.observation_space, env.action_space)

score = []

env.seed(42)
for i in range(0, 20):
    s = env.reset()
    rewards = []
    done = False
    while not done:
        a = agent.act(s)
        s, r, done, info = env.step(a)
        rewards.append(r)
    s = sum(rewards)
    print("\rRound:", i+1, "complete.", "Score:", s, end='')
    score.append(s)

env.seed(88)
for i in range(20, 40):
    s = env.reset()
    rewards = []
    done = False
    while not done:
        a = agent.act(s)
        s, r, done, info = env.step(a)
        rewards.append(r)
    s = sum(rewards)
    print("\rRound:", i+1, "complete.", "Score:", s, end='')
    score.append(s)

env.seed(3)
for i in range(40, 60):
    s = env.reset()
    rewards = []
    done = False
    while not done:
        a = agent.act(s)
        s, r, done, info = env.step(a)
        rewards.append(r)
    s = sum(rewards)
    print("\rRound:", i+1, "complete.", "Score:", s, end='')
    score.append(s)

env.seed(22)
for i in range(60, 80):
    s = env.reset()
    rewards = []
    done = False
    while not done:
        a = agent.act(s)
        s, r, done, info = env.step(a)
        rewards.append(r)
    s = sum(rewards)
    print("\rRound:", i+1, "complete.", "Score:", s, end='')
    score.append(s)

env.seed(0)
for i in range(80, 100):
    s = env.reset()
    rewards = []
    done = False
    while not done:
        a = agent.act(s)
        s, r, done, info = env.step(a)
        rewards.append(r)
    s = sum(rewards)
    print("\rRound:", i+1, "complete.", "Score:", s, end='')
    score.append(s)

print('')
print('Summary')
print('Scores per round:', score)
print('Minimum score:', min(score))
print('Average score:', round(sum(score)/len(score), 2))
print('Maximum score:', max(score))
print('Best episode number:', np.argmax(np.array(score)))

