"""
DQN with replay buffer and target network.
Code from: ajt80@bath.ac.uk
Q-network from: https://github.com/uvipen/Tetris-deep-Q-learning-pytorch/blob/master/src/deep_q_network.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import csv
from torchsummary import summary

try:
    from env.tetris import Tetris
    env = Tetris(6,5)
    print("Using Tetris env")
except ImportError:
    print('Tetris not loaded')


class ReplayBuffer(object):
    def __init__(self, buffer_size=1e6):
        self.buffer = []
        self.buffer_size = buffer_size

    def add_to_buffer(self, data):
        #data must be of the form (state,next_state,action,reward,terminal)
        self.buffer.append(data)
        # If buffer size exceeds the limit, remove oldest experiences
        while len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
    def sample_minibatch(self,minibatch_length):
        states = []
        next_states = []
        actions = []
        rewards = []
        terminals = []
        for i in range(minibatch_length):
            random_int = np.random.randint(0, len(self.buffer)-1) 
            transition = self.buffer[random_int]
            states.append(transition[0])
            next_states.append(transition[1])
            actions.append(transition[2])
            rewards.append(transition[3])
            terminals.append(transition[4])
        return torch.Tensor(states), torch.Tensor(next_states), torch.Tensor(actions), torch.Tensor(rewards), torch.Tensor(terminals)
        
    
class DeepQNetwork(nn.Module):
    def __init__(self, state_dim):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x

class Random_Agent():
    def __init__(self):
        pass
        # self.all_actions = [(col, rot) for col in range(env.game_grid.num_cols) for rot in range(4)]

    def select_action(self, valid_actions:set):
        valid_actions_list = list(valid_actions)
        index = random.randint(0, len(valid_actions_list) - 1)
        return valid_actions_list[index] # sample random action
    

class DQNAgent(object):
    def __init__(self, state_dim):
        self.qnet = DeepQNetwork(state_dim)
        self.qnet_target = copy.deepcopy(self.qnet)
        self.qnet_optim = torch.optim.Adam( self.qnet.parameters(), lr=0.001)
        self.discount_factor = 0.99
        self.MSELoss_function = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(20000)
        self.tau = 0.95
        self.random_agent = Random_Agent()
        self.all_actions = [(col, rot) for col in range(env.game_grid.num_cols) for rot in range(4)]

    def select_action(self, next_states:tuple, epsilon, next_actions):
        if np.random.uniform(0, 1) < epsilon:
            action = self.random_agent.select_action(next_actions)
        
            return action  # tuple - choose random action  
        else:
            # next_states which is a stack of tensors
            next_states = torch.FloatTensor(next_states)
            self.qnet.eval()
            with torch.no_grad():
                list_of_q_values_for_next_states = self.qnet(next_states)[:, 0]
            self.qnet.train()
            index = torch.argmax(list_of_q_values_for_next_states).item()
            action = next_actions[index]
            return action # tuple
    
    def soft_target_update(self,network,target_network,tau):
        for net_params, target_net_params in zip(network.parameters(), target_network.parameters()):
            target_net_params.data.copy_(net_params.data * tau + target_net_params.data * (1 - tau))


    def update_Q_Network(self, state, next_state, action, reward, terminals):
        qsa = self.qnet(state)
        self.qnet_target.eval()
        with torch.no_grad():
            qsa_next_action = self.qnet_target(next_state)
        self.qnet_target.train()
        # the max value
        qsa_next_action,_ = torch.max(qsa_next_action, dim=1, keepdim=True)
        not_terminals = 1 - terminals
        qsa_next_target = reward + not_terminals * self.discount_factor * qsa_next_action
        q_network_loss = self.MSELoss_function(qsa, qsa_next_target.detach())
        self.qnet_optim.zero_grad()
        q_network_loss.backward()
        self.qnet_optim.step()
        
    def update(self, update_rate):
        for i in range(update_rate):
            states, next_states, actions, rewards, terminals = self.replay_buffer.sample_minibatch(256)
            # print(f'states: {states}')
            states = torch.Tensor(states)
            # print(f'states after tensor: {states}')
            next_states = torch.Tensor(next_states)
            actions = torch.Tensor(actions)
            rewards = torch.Tensor(rewards)
            terminals = torch.Tensor(terminals)
            self.update_Q_Network(states, next_states, actions, rewards, terminals)
            self.soft_target_update(self.qnet, self.qnet_target, self.tau)


def run_worker(agent, process_num, episode_rewards_and_lines_cleared):
    np.random.seed(process_num)

    number_of_episodes = 1000
    max_time_steps = 500

    checkpoint = 1000

    for episode in range(number_of_episodes):
        reward_sum = 0
        state = env.reset()
        for i in range(max_time_steps):
            next_steps = env.get_next_states(piece = env.current_piece, state=env.game_grid)
            next_actions, next_states = zip(*next_steps.items())
            epsilon = 0.001 + (max(500 - episode, 0) * (1 - 0.001) / 500)

            action_tuple = agent.select_action(next_states, epsilon, next_actions=next_actions)
            action_index = agent.all_actions.index(action_tuple)

            next_state, reward, terminal = env.step(action_tuple)
            reward_sum += reward
            # episode_count += 1
            agent.replay_buffer.add_to_buffer( (state, next_state,[action_index],[reward],[terminal]) )
            state = next_state
            if terminal:
                break

        episode_rewards_and_lines_cleared.append((reward_sum, env.lines_cleared))

        if episode % checkpoint == 0 and episode > 0:
            save_data_to_csv(episode_rewards_and_lines_cleared, 'results10x10/episode_data_{}.csv'.format(episode))

        
        if episode % 1 == 0:
            print('episode:', episode, 'sum_of_rewards:', reward_sum, 'lines cleared:', env.lines_cleared, 'process_num:', process_num)
        #note: the agent weights are updated from all processes 
        agent.update(50)

def save_data_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Episode', 'Reward', 'Lines Cleared'])
        for episode, (reward, lines_cleared) in enumerate(data):
            csvwriter.writerow([episode, reward, lines_cleared])


state_dim = len(env.reset()) 
agent = DQNAgent(state_dim)
threads = 6


import torch.multiprocessing as mproc
import threading
if __name__ == '__main__':
    # share the network weights between the processes
    agent.qnet.share_memory()
    agent.qnet_target.share_memory()
    processes = []
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    ROLLING_EVENT.set()

    episode_rewards_and_lines_cleared = mproc.Manager().list()    

    for process_num in range(threads):
        p = mproc.Process(target=run_worker, args=(agent, process_num, episode_rewards_and_lines_cleared))
        p.start()
        processes.append(p)

    # wait for all processes to finish
    for p in processes:
        p.join()






   