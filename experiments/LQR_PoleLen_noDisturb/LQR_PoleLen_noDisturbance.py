import os
import numpy as np
import json
from tqdm import tqdm
from envs.continuous_cartpole_v1 import ContinuousCartPoleEnv
from utils.plot_utils import plot_action, plot_states, generate_multi_series
from Optimal_Controller import generate_K_from_ARE, optimal_controller


if __name__ == "__main__":
    # config numpy
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    # experiments parameters
    multi_series = generate_multi_series(max_multi=10,resolution=20,precision=3)
    num_steps = 500
    # init controller
    K = generate_K_from_ARE(len_pole=1)
    # record data
    action_record = []
    obs_record = []
    # generate log file
    log_content = []
    # log header
    info_experiment = {}
    info_experiment['num_steps'] = num_steps
    info_experiment['disturbances'] = 'None'
    info_experiment['variable'] = 'Pole Length'
    info_experiment['multiples'] = multi_series.tolist()
    log_content.append(info_experiment)
    result_dir = os.path.join(os.getcwd(), 'figures')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    with open(os.path.join(result_dir, 'logger.json'), 'w') as file:
        for index, multi in enumerate(tqdm(multi_series)):
            # clear buffer
            obs_record = []
            action_record = []
            info = {}
            # init env
            env = ContinuousCartPoleEnv(length=0.5*multi)
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