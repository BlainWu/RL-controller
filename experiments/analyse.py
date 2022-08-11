import os.path

import numpy as np
import matplotlib.pyplot as plt
import json
from utils.plot_utils import analyse_angle
from utils.plot_utils import generate_RL_multi_poles_test
from networks.REINFORCE import PolicyNet


if __name__ == "__main__":
    model_path = '../models/REINFORCE_Penalise_Absolute_Signal/con_REINFORCE_res21_iter23_reward937.pth'
    figure_dir = './Absolute_Signal_REINFORCE/figures_937'


    generate_RL_multi_poles_test(model_path, figure_dir)

    analyse_angle(log_path=os.path.join(figure_dir,'logger.json'),
                  save_path=os.path.join(os.path.dirname(figure_dir), 'anale_analysis_4453.png'))