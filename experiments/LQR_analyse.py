import os.path

import numpy as np
import time
from envs.continuous_cartpole_v1 import ContinuousCartPole_V1
from utils.plot_utils import *
from scipy import linalg
from Optimal_Controller import generate_K_from_ARE, optimal_controller

def generate_LQR_multi_poles_test(figures_dir, info = '',num_steps=500, max_multi=20, resolution=20):
    """Parameters"""
    # config
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    # experiments parameters
    multi_series = generate_multi_series(max_multi=max_multi, resolution=resolution)
    num_steps = 500
    # generate log file
    log_content = []
    # log header
    info_experiment = {}
    info_experiment['num_steps'] = num_steps
    info_experiment['disturbances'] = 'None'
    info_experiment['variable'] = 'Pole Length'
    info_experiment['multiples'] = multi_series.tolist()
    info_experiment['extra'] = info
    log_content.append(info_experiment)
    result_dir = os.path.join(os.getcwd(), figures_dir)
    if not os.path.exists(result_dir):
        try:
            os.mkdir(result_dir)
        except FileNotFoundError:
            os.mkdir(os.path.dirname(result_dir))
            os.mkdir(result_dir)
    # init controller
    K = generate_K_from_ARE(len_pole=1)
    # record data
    action_record = []
    obs_record = []
    with open(os.path.join(result_dir, 'logger.json'), 'w') as file:
        for index, multi in enumerate(tqdm(multi_series)):
            # clear buffer
            obs_record = []
            action_record = []
            info = {}
            # init env
            env = ContinuousCartPole_V1(length=0.5 * multi)
            obs = env.reset()
            for step in range(num_steps):
                action = optimal_controller(K, obs)
                obs, reward, done, info = env.step(action)
                # record data
                obs_record.append(obs.tolist())
                action_record.append(action.tolist())
            info['Multiplier'] = multi
            info['Obs_Record'] = obs_record
            info['Action_Record'] = action_record
            env.close()
            # plot data
            plot_action(action_record, num_steps, max_value=1.2, show_fig=False,
                        save_path=os.path.join(result_dir, 'action_p{}_length_{}.png'.format(index, multi)))
            plot_states(obs_record, num_steps, show_fig=False,
                        save_path=os.path.join(result_dir, 'state_p{}_length_{}.png'.format(index, multi)))
            log_content.append(info)

        json_f = json.dumps(log_content, indent=3)
        file.write(json_f)

if __name__ == "__main__":
    figures_dir = '../experiments/LQR_PoleLen/angle_m20_r30'
    logger_path = os.path.join(figures_dir, 'logger.json')
    fig_save_path = os.path.join(os.path.dirname(figures_dir),'{}.png'.format(figures_dir.split('/')[-1]))

    #generate_LQR_multi_poles_test(figures_dir, max_multi=20, resolution=30)
    margin = analyse_angle(logger_path, save_path=fig_save_path)
    print(margin)


