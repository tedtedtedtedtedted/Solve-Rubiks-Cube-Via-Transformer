# Utilities to operate on 3x3 Rubik's cube and to address representation of cube state and action/permutation.
# For representation details, see "CubeGPT/README.md".


import random


def color_to_internal(color_repr: str):
    """
    - Input: (str) A color representation of 3x3 Rubik's cube.
    - Output: (list) Conversion to internal representation of the cube.
    """
    internal_repr = [""] * 26 # 3x3 Rubik's cube has 26 small cubes excluding the core cube which has no color.

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


def internal_to_color(internal_repr: list[str]):
    """
    - Input: (list) A internal representation of 3x3 Rubik's cube.
    - Output: (str) Conversion to color representation of the cube. 
    """

    color_repr_list = [""] * 54

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

    return "".join(color_repr_list)
    


def cube_permute(starting_state: str, moves: str):
    """
    - Assuming using color representation of cube state.
    - Input:
        - An starting state to start from.
        - A sequence of permutations.
    - Output:
        - An arrived final state.
    """
    # Convert one lower case to three upper case.
    stupid_moves = stupidize_permutations(moves)
    curr_state = starting_state


    def cube_permute_single(state: str, move: str):
        """
        - Assuming color representation of cube state.
        - Input:
            - An starting state to start with.
            - A single permutation.
        - Output:
            - An arrived final state.
        - Note: Don't care about fast performance. Thus will lessen the code to prioritize readability and correctness over speed performance.
        """
    

        def rotate_face_clockwise(starting_index: int):
            """
            - Rotate a face.
            - Assuming a face have consecutive indices.
            """
            state_transformed[starting_index + 0] = state[starting_index + 6]
            state_transformed[starting_index + 1] = state[starting_index + 3]
            state_transformed[starting_index + 2] = state[starting_index + 0]
            state_transformed[starting_index + 3] = state[starting_index + 7]
            # Skip center.
            state_transformed[starting_index + 5] = state[starting_index + 1]
            state_transformed[starting_index + 6] = state[starting_index + 8]
            state_transformed[starting_index + 7] = state[starting_index + 5]
            state_transformed[starting_index + 8] = state[starting_index + 2]
                 


        state_transformed = list(state)
    
        match move: 
            case "U": 
                # Convert to color repr then perform operation then convert back to internal repr. 
                # Stribe change:
                state_transformed[20] = state[47]
                state_transformed[19] = state[46]
                state_transformed[18] = state[45]
    
                state_transformed[38] = state[20]
                state_transformed[37] = state[19]
                state_transformed[36] = state[18]
    
                state_transformed[29] = state[38]
                state_transformed[28] = state[37]
                state_transformed[27] = state[36]
    
    
                state_transformed[47] = state[29]
                state_transformed[46] = state[28]
                state_transformed[45] = state[27]

                # Face rotation:
                rotate_face_clockwise(0)

            case "D": 
                state_transformed[24] = state[42]
                state_transformed[25] = state[43]
                state_transformed[26] = state[44]
    
                state_transformed[51] = state[24]
                state_transformed[52] = state[25]
                state_transformed[53] = state[26]
    
                state_transformed[33] = state[51]
                state_transformed[34] = state[52]
                state_transformed[35] = state[53]
    
                state_transformed[42] = state[33]
                state_transformed[43] = state[34]
                state_transformed[44] = state[35]

                # Face rotation:
                rotate_face_clockwise(9)

            case "F": 
                state_transformed[6] = state[44]
                state_transformed[7] = state[41]
                state_transformed[8] = state[38]
    
                state_transformed[45] = state[6]
                state_transformed[48] = state[7]
                state_transformed[51] = state[8]
    
                state_transformed[11] = state[45]
                state_transformed[10] = state[48]
                state_transformed[9] = state[51]
    
                state_transformed[44] = state[11]
                state_transformed[41] = state[10]
                state_transformed[38] = state[9]
                
                # Face rotation:
                rotate_face_clockwise(18)

            case "B":
                state_transformed[2] = state[53]
                state_transformed[1] = state[50]
                state_transformed[0] = state[47]
    
                state_transformed[36] = state[2]
                state_transformed[39] = state[1]
                state_transformed[42] = state[0]
    
                state_transformed[15] = state[36]
                state_transformed[16] = state[39]
                state_transformed[17] = state[42]
    
                state_transformed[53] = state[15]
                state_transformed[50] = state[16]
                state_transformed[47] = state[17]

                # Face rotation:
                rotate_face_clockwise(27)

            case "L":
                state_transformed[0] = state[35]
                state_transformed[3] = state[32]
                state_transformed[6] = state[29]
    
                state_transformed[18] = state[0]
                state_transformed[21] = state[3]
                state_transformed[24] = state[6]
    
                state_transformed[9] = state[18]
                state_transformed[12] = state[21]
                state_transformed[15] = state[24]
    
                state_transformed[35] = state[9]
                state_transformed[32] = state[12]
                state_transformed[29] = state[15]

                # Face rotation:
                rotate_face_clockwise(36)

            case "R":
                state_transformed[8] = state[26]
                state_transformed[5] = state[23]
                state_transformed[2] = state[20]
    
                state_transformed[27] = state[8]
                state_transformed[30] = state[5]
                state_transformed[33] = state[2]
    
                state_transformed[17] = state[27]
                state_transformed[14] = state[30]
                state_transformed[11] = state[33]
    
                state_transformed[26] = state[17]
                state_transformed[23] = state[14]
                state_transformed[20] = state[11]

                # Face rotation:
                rotate_face_clockwise(45)

            case "V":
                state_transformed[1] = state[34]
                state_transformed[4] = state[31]
                state_transformed[7] = state[28]
    
                state_transformed[19] = state[1]
                state_transformed[22] = state[4]
                state_transformed[25] = state[7]
    
                state_transformed[10] = state[19]
                state_transformed[13] = state[22]
                state_transformed[16] = state[25]
    
                state_transformed[34] = state[10]
                state_transformed[31] = state[13]
                state_transformed[28] = state[16]  
            case "H":
                state_transformed[23] = state[50]
                state_transformed[22] = state[49]
                state_transformed[21] = state[48]
    
                state_transformed[41] = state[23]
                state_transformed[40] = state[22]
                state_transformed[39] = state[21]
    
                state_transformed[32] = state[41]
                state_transformed[31] = state[40]
                state_transformed[30] = state[39]
    
                state_transformed[50] = state[32]
                state_transformed[49] = state[31]
                state_transformed[48] = state[30]
            case "S":
                state_transformed[3] = state[43]
                state_transformed[4] = state[40]
                state_transformed[5] = state[37]
    
                state_transformed[46] = state[3]
                state_transformed[49] = state[4]
                state_transformed[52] = state[5]
    
                state_transformed[14] = state[46]
                state_transformed[13] = state[49]
                state_transformed[12] = state[52]
    
                state_transformed[43] = state[14]
                state_transformed[40] = state[13]
                state_transformed[37] = state[12]
            
        return "".join(state_transformed)


    for move in stupid_moves:
        curr_state = cube_permute_single(curr_state, move)
    return curr_state 








def stupidize_permutations(moves: str):
    """
    - Input:
        - A sequence of moves that may contain inverse moves.
    - Output:
        - A stupid sequence (list) moves with inverse being replaced by 3 times non-inverse moves, and also for sandwiched moves.
    """
    stupid_moves = []
    for i in range(len(moves)):
        if moves[i].islower():
            stupid_moves.append(moves[i].upper())
            stupid_moves.append(moves[i].upper())
            stupid_moves.append(moves[i].upper())
        else:
             stupid_moves.append(moves[i])
    return "".join(stupid_moves)







def init_state(repr_mode: str):
    """
    - Input: Representation mode, either color repr or internal repr.
    - Output: The initial state of the cube. 
    """
    # For now assume color repr.
    state = [""] * 54

    colors = ["Y", "W", "O", "R", "G", "B"]
    for face in range(6):
        color = colors[face]
        for i in range(9):
            state[face * 9 + i] = color

    return "".join(state)


def challenge_generator(n: int, repr_mode: str, random_start: bool):
    """
    - Input:
        - Number of moves to permute.
        - Representation mode (color repr or internal repr).
        - Whether starting state is random or not. Otherwise, start from initial state.
    - Output:
        - A (internal or color representation) sequence of states and actions, where randomness from actions.
    """
    
    curr_state = init_state(repr_mode) # For now assumed color repr. // TODO: Later need internal repr.
    actions = ["U", "u", "D", "d", "F", "f", "B", "b", "L", "l", "R", "r", "V", "v", "H", "h", "S", "s"]
    if random_start:
        actions_permute_init_state = random.choices(actions, k=100)
        for i in range(100): # Ensure a very random starting state.
            curr_state = cube_permute(curr_state, actions_permute_init_state[i]) 

    record = [""] * (2 * n + 1)
    record[0] = curr_state

    actions_for_record = random.choices(actions, k=n)
    for i in range(n):
        record[2 * i + 1] = actions_for_record[i]
        record[2 * i + 2] = cube_permute(record[2 * i], actions_for_record[i]) # Don't directly use <cube_permute_single>!
        

    # Return type: List of states (color repr so str) and actions (str).
    return record 




def cube_visualize(state: str):
    cube_color_codes = {                                                          
        'Y': "\033[37m",
        'W': "", 
        'O': "\033[32m", 
        'R': "\033[31m",
        'G': "\033[33m",
        'B': "\033[34m"
    }   

    cube_layout = [
        [' ', ' ', ' ', ' ', state[0], state[1], state[2], ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', state[3], state[4], state[5], ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', state[6], state[7], state[8], ' ', ' ', ' '],
        [' ', state[36], state[37], state[38], state[18], state[19], state[20], state[45], state[46], state[47], state[27], state[28], state[29]],
        [' ', state[39], state[40], state[41], state[21], state[22], state[23], state[48], state[49], state[50], state[30], state[31], state[32]],
        [' ', state[42], state[43], state[44], state[24], state[25], state[26], state[51], state[52], state[53], state[33], state[34], state[35]],
        [' ', ' ', ' ', ' ', state[9], state[10], state[11], ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', state[12], state[13], state[14], ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', state[15], state[16], state[17], ' ', ' ', ' ']
    ]

    # Display cube:
    for row in cube_layout:
        for entry in row:
            if entry == ' ':
                print('  ', end='')
            else:
                print(cube_color_codes[entry] + '██' + '\033[0m', end='')
        print()





if __name__ == "__main__":
    #state = init_state("haha")
    #print(state)
    #state = cube_permute(state, "u")
    #print("u")
    #print(state)
    #state = cube_permute(state, "B")
    #print("B")
    #print(state)
    #state = cube_permute(state, "l")
    #print("l")
    #print(state)
    #print("YYYYYYYYYWWWWWWWWWGGGOOOOOOBBBRRRRRRRRRGGGGGGOOOBBBBBB")
    #print(cube_permute("YYYYYYYYYWWWWWWWWWGGGOOOOOOBBBRRRRRRRRRGGGGGGOOOBBBBBB", "B"))
   

    record = challenge_generator(3, "haha", False)
    print("State: " + record[0])
    cube_visualize(record[0])
    for i in range (3):
        print("Action: " + record[2 * i + 1]) # print action
        print("State: " + record[2 * i + 2]) # print resulting state
        cube_visualize(record[2 * i + 2])

    print()
    print()
    print()
    print("Interactive mode:")
    curr_state = init_state("haha")
    print("State: " + curr_state)
    cube_visualize(curr_state)
    while True:
        user_input = input("Enter an action: ")
        if user_input == "quit":
            break
        curr_state = cube_permute(curr_state, user_input)
        print("State: " + curr_state)
        cube_visualize(curr_state)



    #print(record)
    #print(color_to_internal(init_state("haha")))
    #print(internal_to_color(color_to_internal(init_state("haha"))))
    #print(init_state("haha"))
    #cube_visualize(init_state("haha"))
