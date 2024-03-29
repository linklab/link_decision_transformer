# Decision Transformer


## Overview

Minimal code for [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) for mujoco control tasks in OpenAI gym.
Notable difference from official implementation are:

- Simple GPT implementation (causal transformer)
- Uses PyTorch's Dataset and Dataloader class and removes redundant computations for calculating rewards to go and state normalization for efficient training
- Can be trained and the results can be visualized and rendered
- https://github.com/nikhilbarhate99/min-decision-transformer