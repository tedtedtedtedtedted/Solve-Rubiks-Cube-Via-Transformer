import gym
from gym_Rubiks_Cube.envs.rubiks_cube_env import RubiksCubeEnv

class Rubiks():
    """A wrapper for the Rubiks Cube gym environment.
    Is built from https://github.com/RobinChiu/gym-Rubiks-Cube
    """
    def __init__(self, size=3, episode_length=40):
        """
        size: The rubiks cube created will be a size x size x size cube.
        episode_length: The episode will end after episode_length moves, and return negative reward
        """
        self.cube = RubiksCubeEnv(size)
        self.cube.doScramble = False
        self.cube.reset()

    def take_action(self, action):
        """Action is a number from 0 to 11 representing a possible quarter turn.

        The next state, reward, and boolean that is true when the episode has terminated is returned.

        For reference, the moves from 0 to 11 in standard cube notation are:
        F, R, L, U, D, B, F', R', L', U', D', B'
        https://en.wikipedia.org/wiki/Rubik%27s_Cube#Move_notation
        """
        if 0 <= action <= 11:
            next_state, reward, is_done, _ = self.cube.step(action)
            return next_state, reward, is_done
        else:
            raise Exception(f"Invalid Action: {action}")

    def scramble(self, scramble_length=10):
        """Scramble the cube
        scramble_length: The number of random moves that will be used to scramble the cube
        """
        self.cube.reset()
        self.cube.setScramble(scramble_length, scramble_length)
        self.cube.scramble()

    def render(self):
        """Render the cube"""
        self.cube.render()

    def reset(self):
        """Reset the cube"""
        self.cube.reset()

    def step(self):
        """Call the step method
        Here in case we need compatibility with other Gym environments.
        """
        return self.cube.step()


if __name__ == "__main__":
    # Example run for reference
    cube = Rubiks(4, 314)  # Create a 4 by 4 cube, that terminates after 314 moves
    current_state, reward, is_done = cube.take_action(0)  # Do a front turn clockwise
    print(f"This is the state we are in, represented as an ordered list of colors: {current_state}")
    cube.take_action(6)  # Undo the front turn
    cube.render()  # Display the cube on the terminal
    cube.scramble(100)  # Scramble the cube with 100 moves
    cube.render()  # Display the cube on the terminal