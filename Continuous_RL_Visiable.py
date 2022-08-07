import gym
import torch
import numpy as np
from networks.continuous_DQN import Continuous_DQN
from networks.DQN import Qnet
from networks.continuous_REINFORCE import PolicyNet
from utils.continuous_CartPole import ContinuousCartPoleEnv
from utils.plot_utils import plot_action,plot_states
import time

"""Parameters"""
#model_path = './models/con_DQN_res21_model_e5_r429.pth'
model_path = './models/con_REINFORCE_res21_iter12_reward2080.pth'
num_steps = 800
resolution = 21  # IMPORTANT!!! Should be same as the value in the model
actions = np.linspace(-1, 1, resolution)

""" Init """
# env = ContinuousCartPoleEnv()
env = ContinuousCartPoleEnv(disturb_type='Gauss Noise', sensor_index=[0, 1, 2, 3],
                                disturb_starts=100, gaussian_std=0.3)
obs = env.reset()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
DRL_model = torch.load(model_path)
DRL_model.eval()
DRL_model.to(device)

""" Inference """
done = False
step_ct = 0

""" Plot Signal"""
action_record = []
obs_record = []

while(step_ct < num_steps):
    obs = torch.tensor([obs], dtype=torch.float).to(device)
    action_idx = DRL_model(obs).argmax().item()
    action = actions[action_idx]
    obs, reward, done, info = env.step(action)
    env.render(mode = "human")
    action_record.append(action)
    obs_record.append(obs)
    time.sleep(0.001)
    step_ct += 1
env.close()

# plot data
plot_action(action_record, num_steps, max_value=1)
plot_states(obs_record, num_steps)