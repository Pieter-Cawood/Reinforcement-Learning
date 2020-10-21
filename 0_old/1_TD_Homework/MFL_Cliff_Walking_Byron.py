# Group Members
# Clarise Poobalan: 383321
# Nicolaas Cawood: 2376182
# Shikash Algu: 2373769
# Byron Gomes: 0709942R

import gym
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation


    
def espilon_greedy(env, state, Q, epsilon=0.1):
    r = np.random.random()
    
    if r <= epsilon:
        action = env.action_space.sample()
        
    else:
        values = np.zeros(env.action_space.n)
        
        for act in range(env.action_space.n):
            prob, next_state, reward, done = env.P[state][act][0]
            values[act] = Q[next_state, act]
    
        action = np.argmax(values)
        
    return action

def Sarsa_lambda(env, episodes=500, epsilon=0.1, alpha=0.5, lambda_s=0.5, 
                 discount_factor=0.95):
    
    # initialize Q(S,A) and E(S,A)
    Q = np.zeros((env.nS, env.nA))  
    E = np.zeros((env.nS, env.nA))  
    Q_epochs = np.zeros((env.nS, env.nA, episodes))
    
    # Loop over episodes    
    for epi in range(episodes):
        state = env.reset()
        action = espilon_greedy(env, state, Q, epsilon)
        done = False
        count = 0
        
        # loop over steps
        while not done:   
            count += 1
            # take action A, observe R, S'
            S_dash, reward, done, info = env.step(action)

            # choose A' from S' using policy
            A_dash = espilon_greedy(env, S_dash, Q, epsilon)
            
            # update Q and E
            E[state, action] += 1
            delta = reward + discount_factor*Q[S_dash, A_dash] - Q[state, action]
            Q = Q + alpha*delta*E
            E = discount_factor*lambda_s*E
            
            # update state and action
            state = S_dash
            action = A_dash
            
            if count > 1000:
                break

        Q_epochs[:, :, epi] = Q
            
    return Q_epochs



def main():
    # Create environment
    env = gym.make('CliffWalking-v0')
    lambda_vals = np.array([0, 0.3, 0.5, 0.7, 0.9])
    
    # Compute Q with Sarsa Lambda algorithm
    Q = []
    for lam in lambda_vals:
        Q.append(np.max(Sarsa_lambda(env, lambda_s=lam), axis=1).reshape(4, 12, 500))
    
    
    # Create video
    def animate(i):
        plt.clf()
        plt.subplot(321)
        sns.heatmap(Q[0][:,:,i])
        plt.title('Lambda = 0')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(322)
        sns.heatmap(Q[1][:,:,i])
        plt.title('Lambda = 0.5')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(323)
        sns.heatmap(Q[2][:,:,i])
        plt.title('Lambda = 0.5')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(324)
        sns.heatmap(Q[3][:,:,i])
        plt.title('Lambda = 0.7')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(325)
        sns.heatmap(Q[4][:,:,i])
        plt.title('Lambda = 0.9')
        plt.xticks([])
        plt.yticks([])
    
    Writer = animation.writers['ffmpeg']
    writer = Writer()
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, animate)
    ani.save('Byron_Animation.mp4', writer=writer)
    print('Finished')
            
if __name__ == "__main__":
    main()