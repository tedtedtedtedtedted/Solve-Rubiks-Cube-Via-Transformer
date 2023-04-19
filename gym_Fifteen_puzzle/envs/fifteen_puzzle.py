# A fifteen puzzle program (or n^2-1 puzzle).


import sys
import os
from math import ceil, log10


# CITATION: the basic code structure (and some of the helpers at the bottom) are taken from https://github.com/RobinChiu/gym-Rubiks-Cube

class FifteenPuzzle:
    # move a cell left, right, up, down into the emptyspace (so moving empty cell right,left,down,up)
    action_list = ["l", "r", "u", "d"]

    action_dict = {
        "l": (0, 1),
        "r": (0, -1),
        "u": (1, 0),
        "d": (-1, 0)
    }
    empty_space_char = "x"
    char_sep = ' | '
    line_sep_char = "-"
    # Define all the faces.

    def __init__(self, n):
        self.n = n
        self.puzzle = [[n*row + col for col in range(n)] for row in range(n)]
        self.empty_space = (0, 0)

    def displayPuzzle(self):
        width = ceil(log10(self.n*self.n))
        line_width = width*self.n + \
            len(self.char_sep)*self.n + len(self.char_sep.strip())
        line_width = ceil(line_width / (len(self.line_sep_char)))
        line_sep = self.line_sep_char * line_width + '\n'

        sys.stdout.write(line_sep)
        for row in self.puzzle:
            line = self.char_sep.join(
                [str(cell if cell != 0 else self.empty_space_char).ljust(width) for cell in row])
            line = self.char_sep + line + self.char_sep
            line = line.strip()
            sys.stdout.write(line)
            sys.stdout.write('\n')
            sys.stdout.write(line_sep)

    def make_move(self, action):
        if(action not in self.action_dict):  # action=left,right,up,down
            print(
                f"Action '{action}' is invalid. Action must be one of {set(self.action_dict.keys())}.")
            return

        empty_r, empty_c = self.empty_space
        move_r, move_c = self.action_dict[action]
        new_r, new_c = empty_r + move_r, empty_c + move_c
        if 0 <= new_r < self.n and 0 <= new_c < self.n:
            self.puzzle[empty_r][empty_c] = self.puzzle[new_r][new_c]
            self.puzzle[new_r][new_c] = 0
            self.empty_space = (new_r, new_c)

    # Interactive inerpreter for the puzzle.
    def client(self, isColor=False):
        while True:
            # clearScreen()
            self.displayPuzzle()
            userString = str(input("direction to move empty space: "))
            self.make_move(userString)
            print(self.puzzle)
            # print(self.constructVectorState(inBits=True))

    def constructVectorState(self):
        return [[cell for cell in row] for row in self.puzzle]

    # Given a vector state, arrange the cube to that state.
    def destructVectorState(self, puzzle_arr, checkValid=False):
        self.puzzle = [[cell for cell in row] for row in puzzle_arr]

    def isSolved(self):
        return all(self.puzzle[i][j] == self.n * i + j for i in range(self.n) for j in range(self.n))

# A useful clearscreen function


def clearScreen():
    if os.name == "nt":
        os.system('cls')
    else:
        os.system('clear')


def main():
    n = 'a'
    while not n.isdigit():
        clearScreen()
        n = input("\nEnter the size of the puzzle: ")
    cn = FifteenPuzzle(n=int(n))
    cn.client(isColor=True)


# Start the main program lmoa
if __name__ == "__main__":
    main()
