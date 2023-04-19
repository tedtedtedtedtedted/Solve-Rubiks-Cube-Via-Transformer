from gym.envs.registration import register

register(
    id='FifteenPuzzle-v0',
    entry_point='gym_Fifteen_Cube.envs:FifteenCubeEnv',
)
