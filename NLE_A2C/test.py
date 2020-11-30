import gym
import nle
# from MyAgent import MyAgent
from MyAgent2 import MyAgent


env = gym.make("NetHackScore-v0")
env.seed(42)

agent = MyAgent(env.observation_space, env.action_space)

score = []
for i in range(100):
    # print('')
    # print('Start round', i)
    s = env.reset()
    rewards = []
    count = 0
    done = False
    while not done:
        a = agent.act(s)
        s, r, done, info = env.step(a)
        # env.render()
        rewards.append(r)
        # print("Action:", a, "Reward:", r)
        count += 1
    # print('')
    # print("Round", i, "done")
    # print('')
    # positive_rewards = [r for r in rewards if r > 0]
    # print("Total reward:", sum(rewards))
    # print("Positive rewards:", positive_rewards)
    # print("Total actions taken:", count)
    s = sum(rewards)
    print("\rRound:", i+1, "complete.", "Score:", s, end='')
    score.append(s)

print('')
print('Summary')
print('Scores per round:', score)
print('Minimum score:', min(score))
print('Average score:', round(sum(score)/len(score), 2))
print('Maximum score:', max(score))

