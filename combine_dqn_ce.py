"""
Combining DQN with CE using actor-critic framework.
The code is available at: https://github.com/apourchot/CEM-RL
"""


import numpy as np
from copy import deepcopy
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from dqn import *
# provides useful info for tracking/saving
from models import RLNN
from sepCEM import sepCEM

try:
    from env.tetris import Tetris
    print("imported Tetris")
    env = Tetris(6, 5)
except ImportError:
    print("Tetris not loaded")


def evaluate(actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False):
    """Calculates the score (number of lines cleared) of an actor on a given number of runs."""
    scores = []
    steps = 0

    for _ in range(n_episodes):
        score = 0
        obs = deepcopy(env.reset())
        terminal = False

        while True:
            action = actor.select_greedy_action(obs)  # action - index
            next_obs, reward, terminal = env.step(actor.all_actions[action])

            score += reward
            steps += 1

            # adding to memory
            if memory is not None:
                memory.add_to_buffer( (obs,next_obs,[action],[reward],[terminal]) )
            obs = next_obs

            if terminal:
                env.reset()
                break

        scores.append(score)
    
    return np.mean(scores), steps

class Actor(RLNN):
    """Actor network which select actions. Outputs a probability distribution."""
    def __init__(self, state_dim, action_dim, hidden_dims=(64,64)):
        super(Actor, self).__init__(state_dim,action_dim)

        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.tau = 0.005
        self.discount = 0.95
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.all_actions =  [(col, rot) for col in range(env.game_grid.num_cols) for rot in range(4)]

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, x):
        x = self._format(x)
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        logits = self.output_layer(x)
        action_probs = F.softmax(logits, dim=-1)
        return action_probs
    
    def select_action(self, x):
        logits = self.forward(x)
        next_states = env.get_next_states(piece = env.current_piece, state=env.game_grid)
        valid_actions = set(next_states.keys())

        # apply a valid action mask such that invalid actions are not selected
        valid_action_mask = torch.tensor([1 if action in valid_actions else 0 for action in self.all_actions])
        valid_action_logits = logits * valid_action_mask

        dist = torch.distributions.Categorical(logits=valid_action_logits)
        action = dist.sample()
        return action.item() # index
    
    def select_greedy_action(self, state):
        action_probs = self.forward(state)
        next_states = env.get_next_states(piece = env.current_piece, state=env.game_grid)
        valid_actions = set(next_states.keys())

        # apply a valid action mask such that invalid actions are not selected
        valid_action_mask = torch.tensor([1 if action in valid_actions else 0 for action in self.all_actions])
        valid_action_probs = action_probs * valid_action_mask

        valid_action_probs = valid_action_probs / valid_action_probs.sum()
       
        return np.argmax(valid_action_probs.detach().numpy())

    def update(self, memory, batch_size, critic, actor_t):

        # Sample replay buffer
        states, _, _, _, _ = memory.sample_minibatch(batch_size)

        # Compute actor loss - using the critic net to update the policy
        # critic net just returns a q value for a state
        # advantage is achieved by using the critic's ouput to update the policy network directly
        actor_loss = -critic(states).mean()

        # Optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), actor_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class Critic(RLNN):
    """Critic network is a Q-network, output a single value for a state."""
    def __init__(self, state_dim, action_dim, h_layer_dim):
        super(Critic, self).__init__(state_dim, action_dim)

        self.x_layer = nn.Linear(state_dim, h_layer_dim)
        self.h_layer = nn.Linear(h_layer_dim, h_layer_dim)
        self.y_layer = nn.Linear(h_layer_dim, action_dim)
        # print(self.x_layer)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.tau = 0.005
        self.discount = 0.95
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, state):
        xh = F.relu(self.x_layer(state))
        hh = F.relu(self.h_layer(xh))
        state_values = self.y_layer(hh)
        return state_values
    
    def update(self, memory, batch_size, actor_t, critic_t):
        "Updating the critic network."
        # n_states - next states
        states, n_states, actions, rewards, dones = memory.sample_minibatch(batch_size)

        critic_t.eval()
        with torch.no_grad():
            # calculating the Q values for the next states with target net
            target_Q = critic_t(n_states)
        critic_t.train()
        # bellman update
        target_Q = rewards + (1 - dones) * self.discount * target_Q
        
        # Get current Q estimates
        current_Q = self(states)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q.detach())

        # Optimize the critic - updating the critic network
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


if __name__ == "__main__":
    torch.manual_seed(123)

    parser = argparse.ArgumentParser()

    parser.add_argument('--start_steps', default=10000, type=int)
    parser.add_argument('--n_grad', default=5, type=int)

    # DDPG parameters
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=100, type=int)

    # es parameters
    parser.add_argument('--pop_size', default=256, type=int)

    # training parameters
    parser.add_argument('--max_steps', default=1e6, type=int)
    parser.add_argument('--n_episodes', default=1, type=int)

    parser.add_argument('--checkpoint', default=5000, type=int)
    parser.add_argument('--save_all_models', dest="save_all_models", action="store_true")
    parser.add_argument('--output', default='results10x10', type=str)

    args = parser.parse_args()
    #####################################################################################
    state_dim = len(env.reset())
    action_dim = env.game_grid.num_cols * 4

    memory = ReplayBuffer(buffer_size=1e6)

    # make critic
    critic = Critic(state_dim, action_dim, 64)
    critic_t = Critic(state_dim, action_dim, 64)
    critic_t.load_state_dict(critic.state_dict())

    # make actor
    actor = Actor(state_dim, action_dim)
    actor_t = Actor(state_dim, action_dim)
    actor_t.load_state_dict(actor.state_dict())

    # CEM
    es = sepCEM(actor.get_size(), mu_init=actor.get_params(), pop_size=args.pop_size, elitism=True)

    # training
    step_cpt = 0
    total_steps = 0
    actor_steps = 0
    df = pd.DataFrame(columns=["total_steps", "average_score",
                               "average_score_rl", "average_score_ea", "best_score"])
    
    # while we are still interacting with the environment
    while total_steps < args.max_steps:
        fitness  = []
        fitness_ = []
        es_params = es.ask(args.pop_size)

        # updated the rl actor and the critic
        if total_steps > args.start_steps:
            
            for i in range(args.n_grad):

                # set params 
                actor.set_params(es_params[i])
                actor_t.set_params(es_params[i])
                actor.optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

                # critic update
                for _ in range(actor_steps // args.n_grad):
                    critic.update(memory, args.batch_size, actor, critic_t)

                for _ in range(actor_steps):
                    actor.update(memory, args.batch_size, critic, actor_t)

                # get params back in the population
                es_params[i] = actor.get_params()
        actor_steps = 0

        # evaluate all actors
        for params in es_params:
            actor.set_params(params)
            # only evaluating for 1 episode...
            f, steps = evaluate(actor, env, memory=memory, n_episodes=args.n_episodes)
            actor_steps += steps
            fitness.append(f)

            # print scores
            print(f'Actor fitness: {f}')
        
        # update es
        es.tell(es_params, fitness)

        # update step counts
        total_steps += actor_steps
        step_cpt += actor_steps

        # save stuff
        if step_cpt >= args.checkpoint:

            # evaluate mean actor over several runs. Memory not filled and steps not counted
            actor.set_params(es.mu)
            f_mu, _ = evaluate(actor, env, memory=None, n_episodes=30)
            print(f'Actor Mu average Fitness: {f_mu}')

            df.to_pickle(args.output + "/log.pkl")
            res = {"total_steps": total_steps,
                   "average_score": np.mean(fitness),
                   "average_score_half": np.mean(np.partition(fitness, args.pop_size // 2 - 1)[args.pop_size // 2:]),
                   "average_score_rl": np.mean(fitness[:args.n_grad]),
                   "average_score_ea": np.mean(fitness[args.n_grad:]),
                   "best_score": np.max(fitness),
                   "mu_score": f_mu}

            if args.save_all_models:
                os.makedirs(args.output + "/{}_steps".format(total_steps), exist_ok=True)
                critic.save_model(args.output + "/{}_steps".format(total_steps), "critic")
                actor.set_params(es.mu)
                actor.save_model(args.output + "{}_steps".format(total_steps), "actor_mu")
            else:
                critic.save_model(args.output, "critic")
                actor.set_params(es.mu)
                actor.save_model(args.output, "actor")
            df = df._append(res, ignore_index=True)
            step_cpt = 0
            print(res)

        print("Total steps", total_steps)

    

    
    

            

        
