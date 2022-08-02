import gym
import torch
import numpy as np
from networks.continuous_DQN import Continuous_DQN
from networks.DQN import Qnet
from networks.continuous_REINFORCE import PolicyNet
from utils.continuous_CartPole import ContinuousCartPoleEnv
import time

"""Parameters"""
#model_path = './models/con_DQN_res21_model_e5_r429.pth'
model_path = './models/con_REINFORCE_res21_iter12_reward2080.pth'
num_steps = 800
resolution = 21  # IMPORTANT!!! Should be same as the value in the model
actions = np.linspace(-1, 1, resolution)

""" Init """
env = ContinuousCartPoleEnv()
obs = env.reset()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
DRL_model = torch.load(model_path)
DRL_model.eval()
DRL_model.to(device)

""" Inference """
done = False
step_ct = 0
while(step_ct < num_steps and not done):
    obs = torch.tensor([obs], dtype=torch.float).to(device)
    action_idx = DRL_model(obs).argmax().item()
    action = actions[action_idx]
    obs, reward, done, info = env.step(action)
    env.render(mode = "human")
    time.sleep(0.001)
    step_ct += 1
env.close()