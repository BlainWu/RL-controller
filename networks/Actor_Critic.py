import os.path
import utils.rl_utils as rl_utils
from envs.continuous_cartpole_v2 import ContinuousCartPole_V2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)   # the 1 is the score

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        self.actions = np.linspace(-0.5, 0.5, action_dim)
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        # random action
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action_id = action_dist.sample()
        action_id = action_id.item()
        action = self.actions[action_id]
        return action_id, action

    def update(self, transition_dict):
        # unzip data
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        action_idx = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # calculate L(omega) using TD, r+gamma*V(omega_k+1)-V(omega_k)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        # calculate loss
        log_probs = torch.log(self.actor(states).gather(1, action_idx))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # reset the grad
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        # optimize two networks
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()


if __name__ == '__main__':
    env = ContinuousCartPole_V2(penalise='Angle Position Error')
    models_dir = '../models/Actor_Critics_Angle_Position_Error'  # make sure you change it !

    actor_lr = 1e-3
    critic_lr = 1e-2
    check_time = 100  # how often check the model to save
    iterations = 30  # number of round
    resolution = 21
    hidden_dim = 128
    gamma = 0.98
    max_steps = 800
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    # seeds set
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = resolution
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)
    # whether use pre-trained model
    Resume = False
    if Resume:
        path_checkpoint = '../models/REINFORCE_Angle_Position_Error_2/con_REINFORCE_res21_iter44_reward2007.pth'
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
    parameter_dict['resolution'] = resolution
    parameter_dict['hidden_dim'] = hidden_dim
    parameter_dict['gamma'] = gamma
    parameter_dict['rewards'] = '12/(12 + abs(theta*(180 / math.pi))) + 2.4/(2.4 + abs(x)) - 1.0; reward = 0'
    parameter_dict['info'] = 'action = -0.5~0.5;'
    log_content.append(parameter_dict)
    with open(logger_path, 'w') as file:
        json_file = json.dumps(log_content, indent=3)
        file.write(json_file)

    max_return = 0
    return_list = []
    for i in range(iterations):
        with tqdm(total=check_time, desc='Iteration %d' % i) as pbar:
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
                _step_ct = 0
                done = False
                while not done and _step_ct <= max_steps:
                    action_id, action = agent.take_action(state)
                    next_state, reward, done, info = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action_id)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                    _step_ct += 1
                return_list.append(episode_return)  # used for plot
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
            torch.save(agent.actor, os.path.join(models_dir,
                                                 'ActorCritic_res{}_iter{}_reward{}.pth'.format(resolution, i,
                                                                                                       int(avg_return))))
            max_return = avg_return

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('ActorCritic on Continuous CarPole')
    plt.savefig(os.path.join(models_dir, 'return.png'))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('ActorCritic on Continuous CarPole')
    plt.savefig(os.path.join(models_dir, 'mv_return.png'))
    plt.show()
