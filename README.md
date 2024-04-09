# Combining Reinforcement Learning (RL) and Evolutionary Strategies (ES) in Tetris

This repository contains the source code for a pytorch implementation of deep Q-networks (DQN), the cross entropy method (CE), and a combination of the 
two approaches. In this project we address the question: “To what extent does the combination of Reinforcement Learning and Cross-Entropy improve
the performance of Tetris?”


<p align="center">
  <img src="https://github.com/SirSebLancelot/Tetris-Reinforcement-Learning/raw/main/dqn_results/results20x10/tetris_animation_looped_long.gif" alt="Tetris Animation" />
</p>

Technologies used: Python3, PyTorch, Keras, NumPy, Matplotlib, scikit-learn.
- Developed a Tetris environment from scratch to train agents to play Tetris.
- Implemented machine learning algorithms with neural networks to improve performance up to 3 times.
- Ran experiments on high performance Linux systems, using SSH for access, script execution, and data retrieval.
- Created a novel algorithm to improve performance.


## Project Goal
This project aims to train an agent to clear as many lines in Tetris. Tetris is a classical puzzle game that provides an ideal testing ground for comparing the
effectiveness of reinforcement learning and evolutionary strategies. To accomplish our aim we:

- **Develop a Tetris Environment:**  We create a Tetris environment for analysis.
- **Implement and Evaluate CE:** We implement the CE method in Tetris and compare
the effect of adding noise. We observe fast convergence to a good policy that clears
three times more lines than DQN on the 10x10 grid.
- **Implement and Evaluate DQN:** We implement DQN in Tetris with a replay buffer,
target network and decaying epsilon greedy policy. Our results show steady improvement
in performance and highlight DQN’s sensitivity to hyper-parameters in Tetris.
- **Combine DQN and CE:** Building on the strengths of policy based (CE) and value
based (DQN) methods, we use CE’s direct policy search to identify promising solutions.
Then, we use DQN’s iterative updates to fine-tune the policy with the ultimate goal of
clearing more lines. We observe limited learning

## Running the code 

1) Install dependencies - install the required packages from the 'requirements.txt' file.

2) You can train an agent by running each methods python file:
   - `ce.py`
   - `e_constant_noise.py`
   - `ce_decreasing_noise.py`
   - `dqn.py`
   - `combine.py`
  
## Env:
The env folder contains all the files for the Tetris environment.

## Results:

There are 3 folders containing the results from training each of the 3 agents:
`ce_results`, `dqn_results`, and `combine_results`. These contain the trained agent
networks and storing the rewards for plotting.

The performance of an agent was visualised by comparing the number of lines for each agent. Below is an example of the DQN agent compared on the 6x5 and 10x10 grid:
<p align="center">
  <img src= https://github.com/SirSebLancelot/Tetris-Reinforcement-Learning/blob/main/dqn_results/compare_DQN.png alt="Tetris Animation" />
</p>








