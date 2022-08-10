import utils.rl_utils as rl_utils
from envs.continuous_CartPole import ContinuousCartPoleEnv
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json


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
    check_time = 100 # how often check the model to save
    iterations = 30  # number of round
    resolution = 21
    hidden_dim = 128
    gamma = 0.98
    random_len_std = 0.1
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_dir = '../models/random_len_every1_REINFORCE'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # logger
    logger_path = os.path.join(model_dir,'logger.json')
    log_content = []
    parameter_dict = {}
    parameter_dict['lr'] = 1e-3
    parameter_dict['check_time'] = 100
    parameter_dict['iterations'] = 30
    parameter_dict['resolution'] = 21
    parameter_dict['hidden_dim'] = 128
    parameter_dict['gamma'] = 0.98
    parameter_dict['random_len_std'] = 0.1
    log_content.append(parameter_dict)
    with open(logger_path, 'w') as file:
        json_file = json.dumps(log_content, indent=3)
        file.write(json_file)

    env = ContinuousCartPoleEnv()
    # env.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = resolution
    agent = REINFORCE(state_dim, hidden_dim, action_dim, lr, gamma, device)
    # seeds set
    torch.manual_seed(0)

    max_return = 0
    return_list = []
    for i in range(iterations):
        with tqdm(total = check_time, desc='Iteration %d' % i) as pbar:
            for i_episode in range(check_time):
                env = ContinuousCartPoleEnv(random_len=0.1)
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

            # # save the best model
            # if avg_return > max_return:
            #     # save models
            #     model_path = os.path.join(model_dir,
            #                          'random_REIN_res{}_iter{}_reward{}.pth'.format(resolution, i, int(avg_return)))
            #     torch.save(agent.policy_net,model_path)
            #     max_return = avg_return

            # save all models
            model_path = os.path.join(model_dir,
                                      'random_REIN_res{}_iter{}_reward{}.pth'.format(resolution, i,int(avg_return)))
            torch.save(agent.policy_net, model_path)
            max_return = avg_return

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('random length REINFORCE on Continuous CarPole')
    plt.savefig(os.path.join(model_dir, 'return.png'))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('random length REINFORCE on Continuous CarPole')
    plt.savefig(os.path.join(model_dir, 'average return.png'))
    plt.show()