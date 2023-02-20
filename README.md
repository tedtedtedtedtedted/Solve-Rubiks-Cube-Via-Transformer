# Curious-Transformer-On-Rubik-Cube


<p align="center">
<img src="https://wallpaperaccess.com/full/1949972.jpg"
     width="700" />
</p>

- Name: Curious Transformer.
     - Curious: Curiousity-driven learning.
     - Transformer: Attention mechanism.
     
- Core:
     - Transformer $\in$ RL $\rightarrow$ Decision Transformer.
     - Decision Transformer (offline) + Curiousity-driven Learning $\rightarrow$ Curious Transformer (online).

First time setup:
1. In the directory with the file setup.py, run the command:
```Shell
pip install -e .
```
or equivalent for your package manager (or `python3 -m pip install -e .` if it doesn't work).

2. Run rubiks.py to check if everything works.

3. Import Cube from rubiks to start building intelligent agents :)

- Proof of Concept:
     - [ ] Apply Curious Transformer to $3 \times 3$ Rubik's cube.
     - [ ] Generalize to $d \times d$ Rubik's cube.
     - [ ] Generalize to any reinforcement learning tasks.

- Handicaps:
     - No expert data.
     - No problem-specific, hard-coded algorithm.
     - No overkill with models like GPT-3.

