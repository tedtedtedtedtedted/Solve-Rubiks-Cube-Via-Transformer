# Curious-Transformer-On-Rubik-Cube


<p align="center">
<img src="https://i.redd.it/i0mrjegmtau81.jpg"
     width="700" />
</p>

## First time setup
Install the hydra-core and transformers packages (ontop of the usual pytorch and numpy)

## File structure:
The important folders are:
- CubeGPT: The Regular Transformer
- Decision Transformer: The decision transformer and online decision transformer
Please enter these directories for more detailed README instructions to run the code.


## Name
- Curious Transformer.
     - Curious: Curiousity-driven learning.
     - Transformer: Attention mechanism.
     
## Core Idea:
- Transformer $\in$ Reinforcement Learning $\rightarrow$ Decision Transformer.
- Decision Transformer (offline) + Curiousity-driven Learning $\rightarrow$ Curious Transformer (online).
- No expert data.
- No problem-specific, hard-coded solver.
- No overkill with models like GPT-3.
- [ ] Apply Curious Transformer to solve $3 \times 3$ Rubik's cube.
- [ ] Generalize to $d \times d$ Rubik's cube.
- [ ] Generalize to any Reinforcement Learning tasks with deterministic, fully observable, discrete environment.
