# Combining Reinforcement Learning (RL) and Evolutionary Strategies (ES) in Tetris

This repository contains the source code for a pytorch implementation of deep Q-networks, the cross entropy method and a combination of the 
two approaches.

<div align="center">
  <img src="https://github.com/SirSebLancelot/Tetris-Reinforcement-Learning/raw/main/dqn_results/results20x10/tetris_animation.gif" alt="Tetris Animation" />
</div>


## Running the code 

You can train an agent by running each methods python file: 
-ce.py
-ce_constant_noise.py 
-ce_decreasing_noise.py
-dqn.py
-combine.py

## Results:

There are 3 folders containing the results from training each of the 3 agents:
ce_results, dqn_results, and combine_results. These contain the trained agent
networks and storing the rewards for plotting.

## Env:

The env folder contains all the files for the Tetris environment.



