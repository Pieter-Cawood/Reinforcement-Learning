import random
import numpy as np
import gym
import torch
from collections import defaultdict

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *
from rpy2.robjects import r

def argmax(numpy_array):
    """ argmax implementation that chooses randomly between ties """
    return np.random.choice(np.flatnonzero(numpy_array == numpy_array.max()))


if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        print("This will be slow! No GPU available, use Colab instead")
    
    hyper_params = {
        "seed": 42,  # which seed to use
        "env": "PongNoFrameskip-v4",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e6),  # total number of steps to run the environment for
        "batch-size": 256,  # number of transitions to optimize at the same time
        "learning-starts": 10000,  # number of steps before learning starts
        "learning-freq": 5,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10,
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    env = gym.make(hyper_params["env"])
    env.seed(hyper_params["seed"])

    env = NoopResetEnv(env, noop_max=30)   # No co-op max, as described in paper's parameter list
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)  # As described in training details section, when lifes are up - so is the game
    env = FireResetEnv(env)  
    env = WarpFrame(env)    # Warp frame to 84x84 as described in paper
    env = PyTorchFrame(env) # Swap dimensions so channels are 1st -> Pytorch model needs this
    env = ClipRewardEnv(env)   #As described in Training details section
    env = FrameStack(env, 4)   # As described in hyperparameter list, agent history length
   # env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: episode_id % 10 == 0, force=True)
    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    agent = DQNAgent(env.observation_space,
                     env.action_space, 
                     replay_buffer, 
                     hyper_params["use-double-dqn"],
                     hyper_params["learning-rate"],
                     hyper_params["batch-size"],
                     gamma =  hyper_params["discount-factor"])
                     

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    state = env.reset()
    for t in range(hyper_params["num-steps"]):
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )
        # Epsilon greedy action selection
        sample = random.random()
        action = None
        if sample <= eps_threshold:
            action = env.action_space.sample()
        else:
            action = argmax(Q[state])
            
        # Take a leap of faith in the environment (done : float)
        next_state, reward, done, info = env.step(action)
        done = float(done)
        
        episode_rewards.append(reward)

        # Store agent experience at each timestep
        replay_buffer.add(state, action, reward, next_state, done)

        # Update state
        state = next_state

        if done:
            state = env.reset()
            episode_rewards.append(0.0)

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["learning-freq"] == 0
        ):
            agent.optimise_td_loss()

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["target-update-freq"] == 0
        ):
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if (
            done
            and hyper_params["print-freq"] is not None
            and len(episode_rewards) % hyper_params["print-freq"] == 0
        ):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
