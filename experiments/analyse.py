import os.path

import numpy as np
import matplotlib.pyplot as plt
import json
from utils.plot_utils import analyse_angle


if __name__ == "__main__":
    # analyse_angle(log_path='./LQR_PoleLen_noDisturb/figures/logger.json',
    #               save_path='./LQR_PoleLen_noDisturb/angle_analysis.png')
    # analyse_angle(log_path='./REINFORCE_PoleLen_noDisturb/figures/logger.json',
    #               save_path='./REINFORCE_PoleLen_noDisturb/angle_analysis.png')
    # analyse_angle(log_path='./random_len_REINFORCE_PoleLen_noDisturb/figures_every1_8639/logger.json',
    #               save_path='./random_len_REINFORCE_PoleLen_noDisturb/angle_analysis_8639.png')
    figure_dir = './Incremental_Signal_REINFORCE/figures'
    analyse_angle(log_path=os.path.join(figure_dir,'logger.json'),
                  save_path=os.path.join(os.path.dirname(figure_dir), 'anale_analysis.png'))