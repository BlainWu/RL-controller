import os.path
import utils.rl_utils as rl_utils
from envs.continuous_cartpole_v2 import ContinuousCartPole_V2
from envs.continuous_cartpole_v3 import ContinuousCartPole_V3
from utils.rl_utils import ReplayBuffer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json


class ActorNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, max_action):
        super(ActorNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.max_action


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, max_action):
        super(ActorNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.max_action


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        # x is states, a is action
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return self.fc2(x)


class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, max_action, noise_std, actor_lr, critic_lr, tau, gamma, device):
        # init 4 networks
        self.actor = ActorNet(state_dim, hidden_dim, action_dim, max_action).to(device)
        self.target_actor = ActorNet(state_dim, hidden_dim, action_dim, max_action).to(device)
        self.critic = QValueNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = QValueNet(state_dim, action_dim, hidden_dim).to(device)
        # copy the parameter
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        # init optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.noise_std = noise_std      # stand variance of noise
        self.max_action = max_action    # the max value of control signal
        self.tau = tau                  # rate of network updating
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        return action + self.noise_std * np.random.randn(self.action_dim)     # add random noise to action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        # unzip data
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # calculate loss
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        # update critics (minimize)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        #update actor (maxmize)
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

if __name__ == '__main__':
    env = ContinuousCartPole_V3(penalise='Angle Position Error with Control Signal', random_position=0.5, random_len=0.08)
    models_dir = '../models/DDPG_Angle_Position_Error_with_Control_rand_position_polelen01_0'  # make sure you change it !
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    # networks parameters
    hidden_dim = 128
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    max_action = 1
    actor_lr = 5e-4
    critic_lr = 5e-3
    gamma = 0.98
    tau = 0.005
    buffer_size = 10000
    minimal_size = 800
    batch_size = 64
    noise_std = 0.01
    # training parameters
    iterations = 40  # number of round
    check_time = 100  # how often check the model to save
    max_steps = 800
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    random_position = 2
    # seeds set
    # env.seed(0)
    # torch.manual_seed(0)
    agent = DDPG(state_dim, hidden_dim, action_dim, max_action, noise_std, actor_lr, critic_lr, tau, gamma, device)
    # whether use pre-trained model
    Resume = True
    if Resume:
        path_checkpoint = '../models/DDPG_Angle_Position_Error_with_rand_position_0/DDPG_iter7_reward586.pth'
        check_point = torch.load(path_checkpoint, map_location=torch.device('cuda'))
        agent.actor.load_state_dict(check_point.state_dict())
        agent.actor.train()
    # logger
    logger_path = os.path.join(models_dir, 'logger.json')
    log_content = []
    parameter_dict = {}
    parameter_dict['actor_lr'] = actor_lr
    parameter_dict['critic_lr'] = critic_lr
    parameter_dict['check_time'] = check_time
    parameter_dict['iterations'] = iterations
    parameter_dict['noise_std'] = noise_std
    parameter_dict['minimal_size'] = minimal_size
    parameter_dict['buffer_size'] = buffer_size
    parameter_dict['tau'] = tau
    parameter_dict['hidden_dim'] = hidden_dim
    parameter_dict['gamma'] = gamma
    parameter_dict['random_position'] = random_position
    parameter_dict['rewards'] = '1 * (6/(6 + abs(theta*(180 / math.pi)))-0.5) + 1.5 * (1.2/(1.2 + abs(x)) - 0.5) + \
                         0.25*(1/(1+abs(action)) - 0.5)'
    parameter_dict['info'] = 'random position normal 0.5 + control signal+ sensor failure'
    log_content.append(parameter_dict)
    with open(logger_path, 'w') as file:
        json_file = json.dumps(log_content, indent=3)
        file.write(json_file)

    max_return = 0
    return_list = []
    for i in range(iterations):
        with tqdm(total=check_time, desc='Iteration %d' % i) as pbar:
            for i_episode in range(check_time):
                # sensor_failure_index = i_episode % 4 # used for random sensor failure
                sensor_failure_freq = 4
                episode_return = 0
                env = ContinuousCartPole_V3(penalise='Angle Position Error with Control Signal', random_position=0.5,
                                            random_len=0.1)
                state = env.reset()
                _step_ct = 0
                done = False
                while not done and _step_ct <= max_steps:
                    _step_ct += 1
                    action = agent.take_action(state)
                    next_state, reward, done, info = env.step(action)
                    # if _step_ct % sensor_failure_freq == 0:               # used for random sensor failure
                    # if sensor_failure_index in [0, 1, 2]:   # used for random sensor failure
                    #     pass                        # used for random sensor failure
                    # else:                           # used for random sensor failure
                    #     next_state[sensor_failure_index] = 0    # used for random sensor failure
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)  # used for plot
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
            torch.save(agent.actor, os.path.join(models_dir, 'DDPG_iter{}_reward{}.pth'.format(i, int(avg_return))))

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on Continuous CarPole')
    plt.savefig(os.path.join(models_dir, 'return.png'))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on Continuous CarPole')
    plt.savefig(os.path.join(models_dir, 'mv_return.png'))
    plt.show()
