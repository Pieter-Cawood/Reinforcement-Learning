import gym
from gym import wrappers
import matplotlib.pyplot as plt
from collections import defaultdict
from value_function import ValueFunction



def sarsa_semi(episodes=500, runs=100, epsilon=0.1, alpha=0.1,
                        discount_factor=0.95):

    def record_vid(epi):
        return epi == episodes-1

    steps = defaultdict(float)

    for run in range(runs):
        env = gym.make('MountainCar-v0')
        
        if run == 0:
            env = wrappers.Monitor(env, video_callable=record_vid, force=True)
        
        Q = ValueFunction(alpha=alpha, n_actions=env.action_space.n,
                          num_of_tilings=8)

        for epi in range(episodes):
            
            state = env.reset()
            action = Q.act(state, epsilon)
            count = 0
            done = False
            while not done:
                count += 1
                new_state, reward, done, info = env.step(action)
                
                if epi <= (episodes - 1): #not finished
                    new_action = Q.act(new_state, epsilon)
                    target = reward + discount_factor*Q.__call__(new_state, new_action)
                    Q.update(target, state, action)
                    state = new_state
                    action = new_action
                else: #finished
                    env.render()
            
            Q.update(reward, state, action)
            steps[epi] += count
            
        env.close()
        print("Run", run+1, "complete.")
    
    average_steps = [value/runs for key, value in steps.items()]
    
    return average_steps


if __name__ == "__main__":
    average_steps = sarsa_semi()
    plt.title('Average steps per episode')
    plt.semilogy(range(len(average_steps)), average_steps)
    plt.xlabel('Episode number')
    plt.ylim([100,300])
    plt.savefig('Plot.png')
    plt.show()