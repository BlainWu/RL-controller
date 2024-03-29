import os.path

import numpy as np
import matplotlib.pyplot as plt
import json
from utils.plot_utils import analyse_angle_pole_len
from utils.plot_utils import generate_RL_multi_poles_test
from utils.plot_utils import analyse_shift
from networks.REINFORCE import PolicyNet
from networks.DDPG import ActorNet, PolicyNet

if __name__ == "__main__":
    """Single Test"""
    # model_path = '../models/REINFORCE_Penalise_Angle_Error_1/con_REINFORCE_res21_iter24_reward68159.pth'
    # figure_dir = '../experiments/con/figure_1046'
    #
    #
    # generate_RL_multi_poles_test(model_path, figure_dir)
    #
    # analyse_angle(log_path=os.path.join(figure_dir,'logger.json'),
    #               save_path=os.path.join(os.path.dirname(figure_dir), 'anale_analysis_1046.png'))

    """Batch Test - length of pole"""
    model_path = '../models/DDPG_Angle_Position_Error_Atlas'
    figure_path = os.path.join('../experiments/', model_path.split('/')[-1])
    from utils.plot_utils import *

    analyse_all_models_pole_len(model_path, figure_path, max_multi=40, samplings=40, action_type='Continuous')
    # analyse_shift(figure_path)

    # """Batch Test - noise of angle"""
    # model_path = '../models/REINFORCE_Angle_Position_Error_2'
    figure_path = os.path.join('../experiments/', model_path.split('/')[-1]+'_noise')
    from utils.plot_utils import *
    from networks.REINFORCE import PolicyNet
    analyse_all_models_noise(model_path, figure_path, max_sigma=0.5, samplings=40, action_type='Continuous')
