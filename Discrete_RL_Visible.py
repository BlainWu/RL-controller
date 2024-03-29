import torch
import numpy as np
from envs.continuous_cartpole_v1 import ContinuousCartPole_V1
from networks.REINFORCE import PolicyNet
from utils.plot_utils import plot_action,plot_states
import time

"""Position Set-point"""
set_point_start = 200
set_point_position = 1


"""Load Models"""
# model_path = './models/REINFORCE_Angle_Position_Error_1/con_REINFORCE_res21_iter24_reward4080.pth'
# model_path = './models/REINFORCE_Integral_All_0/con_REINFORCE_res21_iter12_reward304.pth'
# model_path = './models/REINFORCE_Integral_All_1/con_REINFORCE_res21_iter17_reward470.pth' # action[-0.5, 0.5]
# model_path = './models/REINFORCE_Angle_Position_Error_5/con_REINFORCE_res21_iter37_reward364.pth' # action[-0.5, 0.5]
# model_path = './models/Actor_Critics_Angle_Position_Error/ActorCritic_res21_iter28_reward737.pth' # action[-0.5, 0.5]
# model_path = './models/REINFORCE_Angle_Position_Error_with_Control_4/con_REINFORCE_res21_iter8_reward344.pth'
# model_path = './models/REINFORCE_Angle_Position_Error_2/REINFORCE_demo.pth'
model_path = './demo_models/REINFORCE_demo.pth'


num_steps = 600
resolution = 21  # IMPORTANT!!! Should be same as the value in the model
action_range = 0.5
actions = np.linspace(-action_range, action_range, resolution)

""" Init """
env = ContinuousCartPole_V1()
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
    action_idx = DRL_model(obs).argmax().item()
    action = actions[action_idx]
    obs, reward, done, info = env.step(action)
    obs_controller = obs.copy()
    if step_ct >= set_point_start:
        obs[0] -= set_point_position
    env.render(mode = "human")
    action_record.append(action)
    obs_record.append(obs_controller)
    time.sleep(0.001)
    step_ct += 1
env.close()

# plot data
plot_action(action_record, num_steps, max_value=1)
plot_states(obs_record, num_steps, show_fig=True)