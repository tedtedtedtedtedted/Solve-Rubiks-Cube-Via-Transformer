# Utilities to operate on 3x3 Rubik's cube and to address representation of cube state and action/permutation.
# For representation details, see "CubeGPT/README.md".





def color_to_internal():
    """
    - Input: A color representation of 3x3 Rubik's cube.
    - Output: Conversion to internal representation of the cube.
    """




def internal_to_color():
    """
    - Input: A internal representation of 3x3 Rubik's cube.
    - Output: Conversion to color representation of the cube. 
    """



def cube_permute():
    """
    - Assuming using internal representation of cube state.
    - Input:
        - A sequence of permutations.
        - An initial state to start from.
    - Output:
        - An arrived final state.
    """
    




def challenge_generator():
    """
    - Input:
        - Number of moves to permute.
        - Representation mode (color repr or internal repr).
    - Output:
        - A (internal or color representation) sequence of states and actions, where randomness from actions.
    """
    # TODO: Decide on whether always begin from initial state or should always first arrive to an random state (which we can use this function itself), then start to record permutation history. 
