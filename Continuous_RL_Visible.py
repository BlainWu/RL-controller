import torch
import numpy as np
from envs.continuous_cartpole_v1 import ContinuousCartPole_V1
from networks.DDPG import ActorNet
from networks.DDPG import ActorNet as PolicyNet
from utils.plot_utils import plot_action,plot_states
import time

"""Load Models"""
# model_path = './models/DDPG_Angle_Position_Error_1/DDPG_iter1_reward439.pth' # response
# model_path = './models/DDPG_Angle_Position_Error/DDPG_iter13_reward769.pth'
# model_path = './models/DDPG_Angle_Position_Error_with_Control_3/DDPG_iter2_reward346.pth'
# model_path = './models/DDPG_Angle_Position_Error_with_rand_position_0/DDPG_iter7_reward586.pth'
# model_path = './models/DDPG_Angle_Position_Error_with_Control_rand_position_1/DDPG_iter1_reward429.pth'
# model_path = './models/DDPG_Angle_Position_Error_with_Control_rand_position_random_sensor_failure_1/DDPG_iter7_reward374.pth'
# model_path = './models/DDPG_Angle_Position_Error_with_Control_rand_position_polelen008_0/DDPG_iter2_reward304.pth'
# model_path = './models/DDPG_Angle_Position_Error_with_Control_step_position/DDPG_iter1_reward533.pth' # step response
# model_path = './models/DDPG_Angle_Position_Error_Atlas/DDPG_iter4_reward476.pth'
model_path = './demo_models/DDPG_demo.pth'

"""Position Set-point"""
set_point_start = 200
set_point_position = 1

num_steps = 600

""" Init """
env = ContinuousCartPole_V1(length=0.5)
# env = ContinuousCartPoleEnv(disturb_type='Gauss Noise', sensor_index=[0, 1, 2, 3],
#                                 disturb_starts=100, gaussian_std=0.3)
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

while step_ct < num_steps:
    obs = torch.tensor([obs], dtype=torch.float).to(device)
    action = DRL_model(obs).item()
    obs, reward, done, info = env.step(action)
    obs_real = obs.copy()
    if step_ct >= 100:
        # obs[2] += np.random.normal(0, 0.05) # random noise
        obs[0] += -1                        # mis-place
    env.render(mode="human")
    action_record.append(action)
    obs_record.append(obs_real)
    time.sleep(0.001)
    step_ct += 1
env.close()

# plot data
plot_action(action_record, num_steps, max_value=1)
plot_states(obs_record, num_steps, show_fig=True)