import os.path

import utils.rl_utils as rl_utils
from envs.continuous_cartpole_v2 import ContinuousCartPole_V2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.gamma = gamma
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.actions = np.linspace(-1, 1, action_dim)

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action_id = action_dist.sample()
        action_id = action_id.item()
        action = self.actions[action_id]
        return action_id, action

    def update(self, transition_dict):
        state_list = transition_dict['states']
        reward_list = transition_dict['rewards']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action_id = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            reward = reward_list[i]
            log_prob = torch.log(self.policy_net(state).gather(1, action_id))
            G = self.gamma * G + reward
            loss = -G * log_prob
            loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    lr = 1e-3
    check_time = 100    # how often check the model to save
    iterations = 60     # number of round
    resolution = 21
    hidden_dim = 128
    gamma = 0.98
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    env = ContinuousCartPole_V2(penalise='Control Signal Increments')
    models_dir = '../models/REINFORCE_Penalise_Incremental_Signal'
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    # seeds set
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = resolution
    agent = REINFORCE(state_dim, hidden_dim, action_dim, lr, gamma, device)

    max_return = 0
    return_list = []
    for i in range(iterations):
        with tqdm(total = check_time, desc='Iteration %d' % i) as pbar:
            for i_episode in range(check_time):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state = env.reset()
                done = False
                while not done:
                    action_id, action = agent.take_action(state)
                    next_state, reward, done, info = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action_id)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return) # used for plot
                agent.update(transition_dict)
                avg_return = np.mean(return_list[-10:])
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (i * check_time + i_episode),
                        'return':
                            '%.3f' % avg_return
                    })
                pbar.update(1)


            # save models
            torch.save(agent.policy_net,os.path.join(models_dir,
                            'con_REINFORCE_res{}_iter{}_reward{}.pth'.format(resolution, i, int(avg_return))))
            max_return = avg_return

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on Continuous CarPole')
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on Continuous CarPole')
    plt.show()