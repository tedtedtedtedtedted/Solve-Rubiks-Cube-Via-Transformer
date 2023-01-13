# Emotional-Learner-On-Rubik-Cube


<p align="center">
<img src="https://wallpaperaccess.com/full/1949972.jpg"
     width="700" />
</p>

- Name: Emotion.

- About: ML algorithm that learn with "emotion". Although the algorithm is designed to be a general AI, but in this specific project, we begin with solving rubik's cube. More precisely, our ML algorithm should have a high probability of solving any given initial states using its preexisting knowledge/tricks, if not solved in a reasonable amount of time, the model should continue learning until it gathers enough tricks to solve it. Note this does not guarantee correctness, and we don't expect it to, because it requires mathematical proof to solve rubik's cube

- Idea:
	- Human memorize emotional events the best.
	- Emotion $\iff$ information gain.
	- Inspired by the old idea "invariance learner" which learns invariance, but realized that invariance learning is a special case of emotion learning, since it gives you the most information gain, and because there sounds like a lot of hard-coded priors of probability, criterions for measures, beliefs of heuristic and we try to avoid these.

- Implementaion:
	- Two stages:
		- One area is for learning (invariance or more generally, tricks that contains most information gain).
			- Two steps:
				- [ ] Neural nets model probabilty distribution
				- Update tool box with promising tricks.
			- With good probability distribution learned in our neural nets, we hope our algo will be surprised by some of powerful trick like ???
			- [ ] Also must have a **variety** of tricks, so when already having a similar class of tricks, we value novel tricks more. This is to ensure it is at least possible to solve the rubik cube with "promising tricks" in the box, and hopefully never (although we will allow it because ML doesn't guarantee) run into issue where the tricks are too fancy and overlapping, and it is mathematically impossible to solve them through any combinations.
			- [ ] Above need notion of similarity of tricks. We will in addition need it for assembling because if a trick highly associated with another learned trick, then can try achieve that trick then reverse the trick.
		- One area is for assembling preexisting knowledge/tricks to solve the given problem.
			- Brute force is most intuitive, and as we gain more promising tricks, we have more tools so brute force isn't as bad. It could still be terrible though.
			- What human can do is inductive reasoning. For example, we conjecture, given a promising trick, whether we can inductively solve the problem or divide and conquer... I wonder how we can model that.
			- [ ] Also need to develop a notion of distance from the goal of original problem to make sure we don't just learn, but learn goal-oriented tricks. This can be naive and heuristic like the number of matches.
	- A memory of tricks that it learned. This bag of tricks grows dynamically in size. In particular, it collects the "most information-gained" tricks.
	- Parallelly attempt to solve original problem while learning tricks.
	- Dynamically determine/define what information gain is, instead of a fixed/hard-coded definition (this could be too early, but long-term).
	- Dynamically determine the underlying probability distribution on $p(S_{t+k} | S_t, A)$, where $A = (a_1, \dots, a_k)$ is a sequence of actions and $S_j$'s are states.
	- Probability distribution $p(S_{t+k} | S_t, A)$ depends on already seen transformations:
		- Initially, uniform distribution since no prior knowledge.
		- Problem is how do we update probability for unseen moves?
			- Attemp 1: Use neural nets to approximate probability distribution (which is a function and neural nets is universal function approximator). In the same spirit as using neural nets in Q-learning in reinforcement learning, we have an initialized probability distribution, then given current state and a sequence of moves, we ask neural nets to predict probability of resulting state and yields the state with highest probability. If it turns out to be
	
