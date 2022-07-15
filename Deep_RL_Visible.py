import gym
import torch
import numpy as np
from networks.DQN import Qnet, DQN
import time

"""Parameters"""
model_path = './models/DQN_model_e2_r342.pth'
num_steps = 500

""" Init """
env = gym.make('CartPole-v0')
obs = env.reset()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
DRL_model = torch.load(model_path)
DRL_model.eval()
DRL_model.to(device)

""" Inference """
for step in range(num_steps):
    obs = torch.tensor([obs], dtype=torch.float).to(device)
    action = DRL_model(obs).argmax().item()
    obs, reward, done, info = env.step(action)
    env.render(mode = "human")
    time.sleep(0.001)

env.close()