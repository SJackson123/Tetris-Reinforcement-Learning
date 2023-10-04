Combining RL and ES in Tetris

## Description
This is the code for combining RL and ES in Tetris. The requirements file contains
the packages required to run the code.

## Table of Contents
-Running the code
-Directory structure

## Running the code 

You can run the code by running each of methods python files: 
-ce.py
-ce_constant_noise.py 
-ce_decreasing_noise.py
-dqn.py
-combine.py

## Directory Structure
--Results:

There are 3 folders containing the results from training each of the 3 agents;
ce_results, dqn_results, and combine_results. These contain the trained agent
networks and storing the rewards for plotting.

-- Env:

The env folder contains all the files for the Tetris environment.

-- ce.py, ce_constant_noise.py, ce_decreasing_noise.py

Implementation of the standard cross entropy method, cross entropy with constant
noise and, cross entropy with decreasing noise. You can run these files and it
should train a cross entropy agent.

-- dqn.py

Implementation of the DQN agent. Running this file you should be able to train a
DQN agent.

-- combine_dqn_ce.py

Implementation of combining DQN and CE. Running this file will attempt to train 
the agent. We make use of imports from sepCEM.py and models.py. sepCEM.py is another
implementation of the CE method. models.py contains useful functionality to 
save and retrieve the network parameters.



