import gym
import nle
from MyAgent import MyAgent

rewards = []
env = gym.make("NetHackScore-v0")
s = env.reset()
# env.render()

agent = MyAgent(env.observation_space, env.action_space)

print('')
print('Start')
count = 0
for i in range(10000):
    a = agent.act(s)
    s, r, d, info = env.step(a)
    # env.render()
    rewards.append(r)
    # print("Action:", a, "Reward:", r)
    count += 1
    if d:
        break

print('')
print("Done")
print('')
positive_rewards = [r for r in rewards if r > 0]
print("Total reward:", sum(rewards))
print("Positive rewards:", positive_rewards)
print("Total actions taken:", count)
