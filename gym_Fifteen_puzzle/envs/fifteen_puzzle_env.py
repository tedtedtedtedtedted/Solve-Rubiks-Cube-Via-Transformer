import gym
from gym import spaces
import numpy as np
import random
from gym_Fifteen_puzzle.envs import fifteen_puzzle

actionList = fifteen_puzzle.FifteenPuzzle.action_list  # = ["l", "r", "u", "d"]


# CITATION: code structure+large sections of the code copy from https://github.com/RobinChiu/gym-Rubiks-Cube (i.e the cube environment we are using)
class FifteenPuzzleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # The empty space is represented as a 0
    # technically only allows for up to n=16.
    def __init__(self, n=3, episode_length=140):
        # the action is which of the four surrounding  to move into the empty space
        self.action_space = spaces.Discrete(4)
        # input is nxn array.
        self.n = n
        self.observation_space = spaces.Box(
            low=0, high=n*n - 1, shape=(n, n), dtype=np.uint8)
        self.step_count = 0
        self.max_steps = episode_length

        self.scramble_moves_min = 1
        self.scramble_high = 40
        self.doScramble = True

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.puzzle = fifteen_puzzle.FifteenPuzzle(n=self.n)
        if self.doScramble:
            self.scramble()
        self.state = self.getstate()
        self.step_count = 0
        self.action_log = []
        return self.state

    def step(self, action):
        self.action_log.append(action)
        self.puzzle.make_move(actionList[action])
        self.state = self.getstate()
        self.step_count += 1

        reward = 0.0
        done = False
        others = {}
        if self.puzzle.isSolved():
            reward = 1.0
            done = True

        if self.step_count > self.max_steps:
            done = True

        return self.state, reward, done, others

    def getstate(self):
        return np.array(self.puzzle.constructVectorArray())

    def render(self, mode='human', close=False):
        if close:
            return
        self.puzzle.displayPuzzle()

    def setScramble(self, low, high, doScramble=True):
        self.scramble_low = low
        self.scramble_high = high
        self.doScramble = doScramble

    def scramble(self):
        num_moves = random.randint(
            self.scramble_low, self.scramble_high)
        while self.puzzle.isSolved():
            self.scramble_log = []

            for _ in range(num_moves):
                move = random.randint(0, 3)
                self.puzzle.make_move(actionList[move])
                self.scramble_log.append(move)

    def getlog(self):
        return self.scramble_log, self.action_log
