import numpy as np
import matplotlib.pyplot as plt
import json
from utils.plot_utils import analyse_angle


if __name__ == "__main__":
    # analyse_angle(log_path='./LQR_PoleLen_noDisturb/figures/logger.json',
    #               save_path='./LQR_PoleLen_noDisturb/angle_analysis.png')
    analyse_angle(log_path='./REINFORCE_PoleLen_noDisturb/figures/logger.json',
                  save_path='./REINFORCE_PoleLen_noDisturb/angle_analysis.png')