"""
How do we choose actions during training?
The main choice is to do an on-policy search, using the transformer to take moves.
We abstract out the decision making component of the model
to prove that it is indeed the best choice during consolidation,
as opposed to some other scheme, for example by simply taking random moves.
"""

import numpy as np

class Policy:
    def __init__(self, epsilon=0) -> None:
        self.possible_moves = {'U', 'D', 'L', 'R', 'F', 'B'}
        self.epsilon = epsilon

    def next_action(self, current_state):
        """
        current_state: The current state of the Rubik's cube
        Output: The next move to be taken, and a random move with probability epsilon
        """
        if np.random.uniform() > self.epsilon:
            return self._next_action(current_state)
        return np.random.choice(self.possible_moves)

    def _next_action(self, current_state):
        """
        current_state: The current state of the Rubik's cube
        Output: The next move to be taken
        """
        raise NotImplementedError

class RandomPolicy(Policy):
    """A policy that takes random moves"""
    def __init__(self, epsilon=0) -> None:
        super().__init__(1)

class OnPolicyTransformer(Policy):
    """The policy that uses the output of the transformer to choose moves"""
    def __init__(self, model, epsilon=0) -> None:
        self.model = model
        super().__init__(epsilon)

    def _next_action(self, current_state):
        # TODO: How do we get the transformer to give us a move?
        pass