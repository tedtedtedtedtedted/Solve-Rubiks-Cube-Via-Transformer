# Summary of CubeGPT implementation details.



- Internal representation of 3x3 Rubik's cube (for transformer).
	- Vocab/token:
		- There are 26 outer colored cubes in 3x3 Rubik's cube.
		- Corner cube has 3 colors, edge cube has 2 colors, center cube has 1 colors.
		- Imagine you hold the cube, and permuting it without your fist twisted, so no whole cube rotation. We will fix this view.
			- In a sequence of tokens that represents a state of the cube, each token represents a small cube. We define the ordering of each cube as:	
				- U > D
				- F > B
				- L > R
				- U > F > L
			- For each token in the sequence, we must define a face ordering:
				- U > D > F > B > L > R
				- Notice there is no small cube/piece in the cube simultaneously colored by opposite faces.
    - Initial state:
        - U: Yellow.
        - D: White.
        - F: Orange.
        - B: Red.
        - L: Green.
        - R: Blue.



- Color representation of 3x3 Rubik's cube (for user, the most natural to user).
    - We will describe by faces:
        - U > D > F > B > L > R.
    - For each face, from the angle of directly facing the face to describe:
        - Notice it's easy to cause ambiguity, when you initially hold the cube, without rotating the cube, but rather move you head:
            1. Move your head up to describe upper face.
            2. Move your head fown to describe lower face.
            3. Move your head back to initial position and describe front face.
            4. Move your head to the back of the cube and describe the back face (i.e. horizontally rotate cube 180 degree).
            5. Move your head to left of the cube and describe the left face.
            6. Move your head to right of the cube and describe the right face.
            
        - Think of it as matrix:
            1  2  3
            4  5  6
            7  8  9



- Permutation representation of 3x3 Rubik's cube:
    - Layers = {U, D, F, B, L, R, V, H, S}.
    - Note:
        - Layers U, D, F, B, L, R are follows the orientation with respect to their faces.
        - V: Vertical middle layer; Follows orientation of left face.
        - H: Horizontal middle layer; Follows orientation of up face.
        - S: Sandwiched middle layer; Follows orientation of front face.
    - Inverse indicator:
        - Lower case is the inverse of upper case. E.g. "u" is the inverse of "U". Upper case follow above described standard orientation of permutation.
    - Permutation sequence:
        $$\{\text{\[layer + orientation\]}\}^k$$.
    - Will not parse syntax like "U2" or "U3"..., just do "UU".
        


- Utilities:
    - \<color\_to\_internal\>:
        - Input: A color representation (likley promped by user) of 3x3 Rubik's cube. 
        - Output: Conversion to internal representation of the cube.
    - \<internal\_to\_color\>:
        - Inverse of \<color\_to\_internal\>.
    - \<cube\_permute\>:
        - Assuming using internal representation of the cube state.
        - Input:
            - A sequence of action/permutation.
            - An initial state to start from.
        - Output:
            - A final state that we arrived.
    - \<challenge\_generator\>:
        - Input:
            - Number of moves to permute.
            - Representation mode (color repr or internal repr).
        - Output:
            - A (internal or color representation) sequence of states and actions, where randomness comes from actions.

        
