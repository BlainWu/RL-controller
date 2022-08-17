import utils.rl_utils as rl_utils
from envs.continuous_cartpole_v1 import ContinuousCartPole_V1
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from networks.REINFORCE import PolicyNet, REINFORCE

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
    parameter_dict['lr'] = lr
    parameter_dict['check_time'] = check_time
    parameter_dict['iterations'] = iterations
    parameter_dict['resolution'] = resolution
    parameter_dict['hidden_dim'] = hidden_dim
    parameter_dict['gamma'] = gamma
    parameter_dict['random_len_std'] = random_len_std
    parameter_dict['extra_info'] = 'None'
    log_content.append(parameter_dict)
    with open(logger_path, 'w') as file:
        json_file = json.dumps(log_content, indent=3)
        file.write(json_file)

    env = ContinuousCartPole_V1()
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
                env = ContinuousCartPole_V1(random_len=0.1)
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