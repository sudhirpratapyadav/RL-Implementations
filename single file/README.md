This folder contains single file implementations of various RL algorithms.

# Algorithms Details

## 1. Reinforce
- File Name: reinforce.py
- This algorithm is the simplest policy gradient algorithm based directly on Policy Gradient theorem.
- Theory
  - Youtube Lecture: UC Berkely CS 285 Course by Sergey Levine [Lecture 5 Part 1](https://www.youtube.com/watch?v=GKoKNYaBvM0). This is the exact algorithm that have been implemented.
  - Open AI Spinning Up: [Part 3: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) read this to understand more about implementation details.
- Main isuue you will get is how to implement the policy gradient that is obtained using the auto-grad of the pytorch. For that a pseudo-loss function is deinfed, differentiating that (calling `loss.backword`) will give same gradients as the policy gradient theoram requires.
- Code Credits
  -  Modfied from Open AI's [Simple Policy Gradient](https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py)
