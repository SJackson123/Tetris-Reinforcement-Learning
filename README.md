# Combining Reinforcement Learning (RL) and Evolutionary Strategies (ES) in Tetris

This repository contains the source code for a pytorch implementation of deep Q-networks, the cross entropy method and a combination of the 
two approaches.



<p align="center">
  <img src="https://github.com/SirSebLancelot/Tetris-Reinforcement-Learning/raw/main/dqn_results/results20x10/tetris_animation_looped_long.gif" alt="Tetris Animation" />
</p>

## Running the code 

1) Install dependencies - install the required packages from the 'requirements.txt' file.

2) You can train an agent by running each methods python file:
   - ce.py
   - e_constant_noise.py
   - ce_decreasing_noise.py
   - dqn.py
   - combine.py
  
## Env:
The env folder contains all the files for the Tetris environment.

## Results:

There are 3 folders containing the results from training each of the 3 agents:
'ce_results', 'dqn_results', and 'combine_results'. These contain the trained agent
networks and storing the rewards for plotting.





