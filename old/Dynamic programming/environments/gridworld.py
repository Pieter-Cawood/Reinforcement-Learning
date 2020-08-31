import io
import numpy as np
import sys
import gym
from gym import  spaces
from gym.utils import seeding

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NUMBER_OF_ACTIONS = 4

class GridworldEnv(gym.Env):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal state.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[4,4], terminal_states=[15], terminal_reward = 0, step_reward=-1):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape
        self.observation_space = spaces.Discrete(np.prod(shape))

        for t in terminal_states:
            assert 0 <= t < self.observation_space.n

        self.action_space = spaces.Discrete(NUMBER_OF_ACTIONS)
        self.__terminal_states = terminal_states
        self.__terminal_reward = terminal_reward
        self.__step_reward = step_reward

        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(self.observation_space.n).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(self.action_space.n)}

            is_done = lambda s: s in self.__terminal_states

            # stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, self.__terminal_reward, True)]
                P[s][RIGHT] = [(1.0, s, self.__terminal_reward, True)]
                P[s][DOWN] = [(1.0, s, self.__terminal_reward, True)]
                P[s][LEFT] = [(1.0, s, self.__terminal_reward, True)]
            # Not a terminal state
            else:
                reward = self.__step_reward
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward, False)]
                P[s][RIGHT] = [(1.0, ns_right, reward, False)]
                P[s][DOWN] = [(1.0, ns_down, reward, False)]
                P[s][LEFT] = [(1.0, ns_left, reward, False)]

            it.iternext()

        # Initial state distribution is uniform
        self.__initial_state_distribution = np.ones(self.observation_space.n) / self.observation_space.n

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__()


    def step(self, action):
        assert self.action_space.contains(action)
        prob, next_state, reward, done = self.P[self.__current_state][action][0]
        self.__current_state = next_state
        return next_state, reward, done, None

    def reset(self):
        self.__current_state =  np.random.choice(self.observation_space.n, p=self.__initial_state_distribution)
        return self.__current_state

    def render(self, mode='human', close=False):
        """ Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.observation_space.n).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.__current_state == s:
                output = " x "
            elif s in self.__terminal_states:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()

    def seed(self, seed=None):
        if(seed != None):
            np.random.seed(seed)

    def close(self):
        pass