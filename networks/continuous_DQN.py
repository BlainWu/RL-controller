import gym
import numpy as np
import torch
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils.rl_utils as rl_utils
from networks.DQN import Qnet
from utils.continuous_CartPole import ContinuousCartPoleEnv


class Continuous_DQN:

    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma,
                 epsilon, target_update, device):
        self.actions = np.linspace(-1, 1, action_dim)
        self.action_dim = action_dim
        # learning net
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # target net
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma  # discount
        self.epsilon = epsilon  # e-greedy
        self.target_update = target_update  # update frequency
        self.count = 0
        self.device = device

    def take_action(self, state):
        # return the value of action; e-greedy method
        if np.random.random() < self.epsilon:
            action_id = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action_id = self.q_net(state).argmax().item()
        action = self.actions[action_id]
        return action_id, action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        action_idx = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)
        q_values = self.q_net(states).gather(1, action_idx)
        """
        output of nn:
        [[q1, q2],
         [q1, q2],
        ...]
        size = batch_size * action
        """
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        # loss function
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad() # reset the grad
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict()) # update NN
        self.count += 1


if __name__ == "__main__":
    resolution = 21 # from -1 to 1, at 0.1 step size
    lr = 2e-3
    iterations = 50
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10  # how many episodes update the Q_target
    buffer_size = 10000
    minimal_size = 500 # the min number of data, when start to train
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("========= The training device is {}. =========".format(device))


    env = ContinuousCartPoleEnv()
    _ = env.reset()
    # random seeds
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = resolution  # the actions between -1 and 1
    agent = Continuous_DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

    max_return = 0
    return_list = []
    for i in range(iterations):
        with tqdm(total=int(num_episodes/iterations), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/iterations)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action_id, action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action_id, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s,b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                avg_return = np.mean(return_list[-10:])
                if(i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' %(num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % avg_return
                    })
                pbar.update(1)

        # save the best model
        if avg_return > max_return:
            # save models
            torch.save(agent.q_net, '../models/con_DQN_res{}_model_e{}_r{}.pth'.format(resolution, i, int(avg_return)))
            max_return = avg_return

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Continuous DQN on CartPole')
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Continuous DQN on CartPole')
    plt.show()