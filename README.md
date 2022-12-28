# Invariance-Learner-On-Rubik-Cube
Explore the idea of machine learning on invariance. Apply invariance learner on solving Rubik's cube.

<p align="center">
<img src="https://wallpaperaccess.com/full/1949972.jpg"
     width="700" />
</p>





- Theme: Invariance learning.

- Goal: No rush into finding an algorithm, but to explore (good) ways of learning & discovering (meaningful) invariance by messing around.

- Philosophy: 
	- Inspired by music for example, classical is harmoneous because it follows rules; even jazz is joyful because under the hood there is some invariance, even if it is invariance of dissonance
	- Inspired by problem solving, always play around to seek properties/invariance to solve a problem. 
	- Maybe the reason that human and animals play is to discover invariance of the world. Maybe that's why Shaul likes to play with fidget spinner (because spinning is a sort of invariance).

- Starter:
	1. Set up with OpenAI gym
		- Resources:
			- https://github.com/DoubleGremlin181/RubiksCubeGym
			- https://github.com/RobinChiu/gym-Rubiks-Cube/blob/master/image/play.png
			- https://github.com/marc131183/GymRubiksCube
	1. Given small tasks		
		1. Greedy
		2. Composite
		3. Random/genetic
		4. RL
	2. Explore new invariance itself and evaluate discovered invariance by some metric to decide whether it is useful enough or not to memorize it:
		- Resembleness/closedness
		- Symmetry
			- [ ] Stage 1: Human defined
			- [ ] Stage 2: Machine invented symmetry
	3. ~ Dynamic programming (store these trick to use later)
	4. Global strategy planner
		- [ ] Stage 1: Human instructed
		- [ ] Stage 2: Machine planner
