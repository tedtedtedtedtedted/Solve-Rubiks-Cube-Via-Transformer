# Utilities to operate on 3x3 Rubik's cube and to address representation of cube state and action/permutation.
# For representation details, see "CubeGPT/README.md".





def color_to_internal(color_repr: list):
    """
    - Input: (str) A color representation of 3x3 Rubik's cube.
    - Output: (list) Conversion to internal representation of the cube.
    """
    internal_repr = [None] * 26 # 3x3 Rubik's cube has 26 small cubes excluding the core cube which has no color.

    # Upper layer:
    internal_repr[0] = color_repr[6] + color_repr[18] + color_repr[38]
    internal_repr[1] = color_repr[7] + color_repr[19]
    internal_repr[2] = color_repr[8] + color_repr[20] + color_repr[45]
    internal_repr[3] = color_repr[3] + color_repr[37]
    internal_repr[4] = color_repr[4]
    internal_repr[5] = color_repr[5] + color_repr[46]
    internal_repr[6] = color_repr[0] + color_repr[29] + color_repr[36]
    internal_repr[7] = color_repr[1] + color_repr[28]
    internal_repr[8] = color_repr[2] + color_repr[27] + color_repr[47]

    # Middle layer:
    internal_repr[9] = color_repr[21] + color_repr[41]
    internal_repr[10] = color_repr[22]
    internal_repr[11] = color_repr[23] + color_repr[48]
    internal_repr[12] = color_repr[40]
    internal_repr[13] = color_repr[49]
    internal_repr[14] = color_repr[32] + color_repr[39]
    internal_repr[15] = color_repr[31]
    internal_repr[16] = color_repr[30] + color_repr[50]

    # Lower layer:
    internal_repr[17] = color_repr[9] + color_repr[24] + color_repr[44]
    internal_repr[18] = color_repr[10] + color_repr[25]
    internal_repr[19] = color_repr[11] + color_repr[26] + color_repr[51]
    internal_repr[20] = color_repr[12] + color_repr[43]
    internal_repr[21] = color_repr[13]
    internal_repr[22] = color_repr[14] + color_repr[52]
    internal_repr[23] = color_repr[15] + color_repr[35] + color_repr[42]
    internal_repr[24] = color_repr[16] + color_repr[34]
    internal_repr[25] = color_repr[17] + color_repr[33] + color_repr[53]

    return internal_repr


def internal_to_color(internal_repr: list):
    """
    - Input: (list) A internal representation of 3x3 Rubik's cube.
    - Output: (str) Conversion to color representation of the cube. 
    """

    color_repr_list = [None] * 54

    # Up face.
    color_repr_list[0] = internal_repr[6][0]
    color_repr_list[1] = internal_repr[7][0]
    color_repr_list[2] = internal_repr[8][0]
    color_repr_list[3] = internal_repr[3][0]
    color_repr_list[4] = internal_repr[4][0]
    color_repr_list[5] = internal_repr[5][0]
    color_repr_list[6] = internal_repr[0][0]
    color_repr_list[7] = internal_repr[1][0]
    color_repr_list[8] = internal_repr[2][0]

    # Down face.
    color_repr_list[9] = internal_repr[17][0]
    color_repr_list[10] = internal_repr[18][0]
    color_repr_list[11] = internal_repr[19][0]
    color_repr_list[12] = internal_repr[20][0]
    color_repr_list[13] = internal_repr[21][0]
    color_repr_list[14] = internal_repr[22][0]
    color_repr_list[15] = internal_repr[23][0]
    color_repr_list[16] = internal_repr[24][0]
    color_repr_list[17] = internal_repr[25][0]

    # Front face.
    color_repr_list[18] = internal_repr[0][1]
    color_repr_list[19] = internal_repr[1][1]
    color_repr_list[20] = internal_repr[2][1]
    color_repr_list[21] = internal_repr[9][0]
    color_repr_list[22] = internal_repr[10][0]
    color_repr_list[23] = internal_repr[11][0]
    color_repr_list[24] = internal_repr[17][1]
    color_repr_list[25] = internal_repr[18][1]
    color_repr_list[26] = internal_repr[19][1]

    # Back face.
    color_repr_list[27] = internal_repr[8][1]
    color_repr_list[28] = internal_repr[7][1]
    color_repr_list[29] = internal_repr[6][1]
    color_repr_list[30] = internal_repr[16][0]
    color_repr_list[31] = internal_repr[15][0]
    color_repr_list[32] = internal_repr[14][0]
    color_repr_list[33] = internal_repr[25][1]
    color_repr_list[34] = internal_repr[24][1]
    color_repr_list[35] = internal_repr[23][1]

    # Left face.
    color_repr_list[36] = internal_repr[6][2]
    color_repr_list[37] = internal_repr[3][1]
    color_repr_list[38] = internal_repr[0][2]
    color_repr_list[39] = internal_repr[14][1]
    color_repr_list[40] = internal_repr[12][0]
    color_repr_list[41] = internal_repr[9][1]
    color_repr_list[42] = internal_repr[23][2]
    color_repr_list[43] = internal_repr[20][1]
    color_repr_list[44] = internal_repr[17][2]

    # Right face.
    color_repr_list[45] = internal_repr[2][2]
    color_repr_list[46] = internal_repr[5][1]
    color_repr_list[47] = internal_repr[8][2]
    color_repr_list[48] = internal_repr[11][1]
    color_repr_list[49] = internal_repr[13][0]
    color_repr_list[50] = internal_repr[16][1]
    color_repr_list[51] = internal_repr[19][2]
    color_repr_list[52] = internal_repr[22][1]
    color_repr_list[53] = internal_repr[25][2]



    return str(color_repr_list)
    



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
