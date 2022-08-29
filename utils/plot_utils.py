import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os
import torch
from tqdm import tqdm
from envs.continuous_cartpole_v1 import ContinuousCartPole_V1

def arc_to_degree(theta):
    return [i*(180 / math.pi) for i in theta]

def plot_states(data, steps, grid=False, threshold=True,
                auto_adjust=False, show_fig=False, save_path=None):
    data = np.array(data)
    axis = [i for i in range(steps)]
    # unzip data
    x, x_dot, theta, theta_dot = \
        np.hsplit(data, 4)[0].squeeze(), np.hsplit(data, 4)[1].squeeze(), \
        np.hsplit(data, 4)[2].squeeze(), np.hsplit(data, 4)[3].squeeze()
    theta = arc_to_degree(theta)
    state = plt.figure('Status', figsize=(7.2, 4.8))
    plt.subplot(221)
    plt.title('Position of Car')
    plt.grid(grid)
    plt.ylabel('m')
    if not auto_adjust:
        max_dist = 2.4
        plt.axis([0, steps, -max_dist * 1.2, max_dist * 1.2])
        if threshold:
            plt.axhline(y=max_dist, color='red',linestyle='--')
            plt.axhline(y=-max_dist, color='red',linestyle='--')
    plt.plot(axis, x)

    plt.subplot(222)
    plt.title('Velocity of Car')
    #plt.ylim((-0.5, 0.5))
    plt.grid(grid)
    plt.ylabel('m/s')
    plt.plot(axis, x_dot)

    plt.subplot(223)
    plt.title('Angle of Pole')
    plt.grid(grid)
    plt.ylabel('deg')
    if not auto_adjust:
        max_ang = 12
        plt.axis([0, steps, -max_ang * 1.2, max_ang * 1.2])
        if threshold:
            plt.axhline(y=max_ang, color='red',linestyle='--')
            plt.axhline(y=-max_ang, color='red',linestyle='--')
    plt.plot(axis, theta)

    plt.subplot(224)
    plt.title('Angular Velocity of Pole')
    plt.grid(grid)
    #plt.ylim((-0.2, 0.2))
    plt.ylabel('rad/s')
    plt.plot(axis, theta_dot)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10,
                        right=0.95, hspace=0.35, wspace=0.35)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    if show_fig:
        state.show()


def plot_action(data, steps, grid=False, max_value=0.5,
                show_fig=True, save_path=None):
    x_axis = [i for i in range(steps)]
    # calculate metric
    var = round(np.var(data), 4)
    mean = round(np.mean(data), 4)
    act_plot = plt.figure('Control Signal')
    ax = plt.gca()
    plt.title('Control Signal')
    plt.plot(x_axis, data)
    plt.axis([0, steps, -max_value, max_value])
    plt.text(0.1, 0.8, r'$\mu={},\ \sigma={}$'.format(mean, var), transform=ax.transAxes)
    plt.grid(grid)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()

    if show_fig:
        act_plot.show()


def analyse_angle_pole_len(log_path, show_fig=True, save_path=None, intersection=True):
    stable_margin = 0.0
    stable_angle = 12.0
    # load file
    log_content = []
    with open(log_path, 'r') as f:
        log_content = json.load(f)
    # iterate info
    multi_series = []
    max_angle = []
    for index, text in enumerate(log_content):
        if index == 0:  # skip the info list
            multiples = text['multiples']
        else:
            multi_series.append(text['Multiplier'])
            theta_all = np.hsplit(np.array(text['Obs_Record']), 4)[2].squeeze()
            theta_all = arc_to_degree(theta_all)
            max_angle.append(max(np.abs(theta_all)))
    # plot figure
    max_angle_plt = plt.figure()
    plt.axes(xscale="log")
    plt.plot(multi_series, max_angle)
    plt.ylim((0, 15))
    plt.xlim((0.05, 50))
    ax = plt.gca()
    plt.title('Maximum angle at different pole lengths')
    plt.xlabel('multiplier')
    plt.ylabel('maximum degree')
    plt.axhline(y=stable_angle, color='red', linestyle='--')

    # draw the intersections
    if intersection:
        _x = []
        _y = []
        inter_points = []
        for index in range(len(multi_series) - 1):
            if max_angle[index] >= stable_angle > max_angle[index + 1] or \
                    max_angle[index] <= stable_angle < max_angle[index + 1]:
                _x.append(multi_series[index])
                _y.append(max_angle[index])
                _x.append(multi_series[index + 1])
                _y.append(max_angle[index + 1])
                inter_points.append(np.interp(stable_angle, _y, _x))
                _x = []
                _y = []

        _margin_list = []
        inter_points = [round(i, 2) for i in inter_points]
        len_inter = len(inter_points)
        if len_inter >= 2:
            # points pre-process
            if len_inter == 2:
                pass
            else:
                _range = []
                for i in range(len_inter - 1):
                    _margin_list.append(inter_points[i + 1] - inter_points[i])
                    _margin_list = [round(i, 2) for i in _margin_list]
                _max_index = _margin_list.index(max(_margin_list))
                inter_points = inter_points[_max_index:_max_index + 2]
            # draw points and infobox
            for point in inter_points:
                plt.plot(point, stable_angle, 'ro')
                plt.axvline(x=point, color='red', linestyle=':')
            _min = min(inter_points)
            _max = max(inter_points)
            stable_margin = _max - _min
            info_box = '\n'.join((r'$min=%.2f$' % (_min, ),
                                  r'$max=%.2f$' % (_max,),
                                  r'$ranges=%.2f$' % (stable_margin, )))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.15, info_box, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    if show_fig:
        max_angle_plt.show()

    return stable_margin


def analyse_position_noise(log_path, show_fig=True, save_path=None, intersection=True):
    stable_margin = 0.0
    stable_position = 2.4
    # load file
    log_content = []
    with open(log_path, 'r') as f:
        log_content = json.load(f)
    # iterate info
    multi_series = []
    max_position = []
    for index, text in enumerate(log_content):
        if index == 0:  # skip the info list
            multiples = text['noise_series']
        else:
            multi_series.append(text['sigma'])
            all_position = np.hsplit(np.array(text['Obs_Record']), 4)[0].squeeze()
            max_position.append(max(np.abs(all_position)))
    # plot figure
    max_position_plot = plt.figure()
    plt.plot(multi_series, max_position)
    plt.xlim((0.05, 50))
    plt.ylim((0, 3))
    ax = plt.gca()
    plt.title('Maximum position at different sigma of Gaussian noise')
    plt.xlabel('sigma')
    plt.ylabel('maximum position')
    plt.axhline(y=stable_position, color='red', linestyle='--')

    # draw the intersections
    if intersection:
        _x = []
        _y = []
        inter_points = []
        for index in range(len(multi_series) - 1):
            if max_position[index] >= stable_position > max_position[index + 1] or \
                    max_position[index] <= stable_position < max_position[index + 1]:
                _x.append(multi_series[index])
                _y.append(max_position[index])
                _x.append(multi_series[index + 1])
                _y.append(max_position[index + 1])
                inter_points.append(np.interp(stable_position, _y, _x))
                _x = []
                _y = []

        _margin_list = []
        inter_points = [round(i, 2) for i in inter_points]
        len_inter = len(inter_points)
        if len_inter >= 1:
            # points pre-process
            inter_point = inter_points[0]  # pick up the first one
            # draw points and infobox

            plt.plot(inter_point, stable_position, 'ro')
            plt.axvline(x=inter_point, color='red', linestyle=':')
            stable_margin = inter_point
            info_box = '\n'.join((r'$max=%.2f$' % (inter_point,),
                                  r'$ranges=%.2f$' % (stable_margin,)))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.15, info_box, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    if show_fig:
        max_position_plot.show()

    return stable_margin

def analyse_angle_noise(log_path, show_fig=True, save_path=None, intersection=True):
    stable_margin = 0.0
    stable_angle = 12.0
    # load file
    log_content = []
    with open(log_path, 'r') as f:
        log_content = json.load(f)
    # iterate info
    multi_series = []
    max_angle = []
    for index, text in enumerate(log_content):
        if index == 0:  # skip the info list
            multiples = text['noise_series']
        else:
            multi_series.append(text['sigma'])
            theta_all = np.hsplit(np.array(text['Obs_Record']), 4)[2].squeeze()
            theta_all = arc_to_degree(theta_all)
            max_angle.append(max(np.abs(theta_all)))
    # plot figure
    max_angle_plt = plt.figure()
    plt.plot(multi_series, max_angle)
    plt.ylim((0, 15))
    ax = plt.gca()
    plt.title('Maximum angle at different sigma of Gaussian noise')
    plt.xlabel('sigma')
    plt.ylabel('maximum degree')
    plt.axhline(y=stable_angle, color='red', linestyle='--')

    # draw the intersections
    if intersection:
        _x = []
        _y = []
        inter_points = []
        for index in range(len(multi_series) - 1):
            if max_angle[index] >= stable_angle > max_angle[index + 1] or \
                    max_angle[index] <= stable_angle < max_angle[index + 1]:
                _x.append(multi_series[index])
                _y.append(max_angle[index])
                _x.append(multi_series[index + 1])
                _y.append(max_angle[index + 1])
                inter_points.append(np.interp(stable_angle, _y, _x))
                _x = []
                _y = []

        _margin_list = []
        inter_points = [round(i, 2) for i in inter_points]
        len_inter = len(inter_points)
        if len_inter >= 1:
            # points pre-process
            inter_point = inter_points[0]  # pick up the first one
            # draw points and infobox
            plt.plot(inter_point, stable_angle, 'ro')
            plt.axvline(x=inter_point, color='red', linestyle=':')
            stable_margin = inter_point
            info_box = '\n'.join((r'$max=%.2f$' % (inter_point,),
                                  r'$ranges=%.2f$' % (stable_margin,)))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.15, info_box, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    if show_fig:
        max_angle_plt.show()

    return stable_margin


def analyse_position_pole_len(log_path, show_fig=True, save_path=None, intersection=True):
    stable_margin = 0.0
    stable_position = 2.4
    # load file
    log_content = []
    with open(log_path, 'r') as f:
        log_content = json.load(f)
    # iterate info
    multi_series = []
    max_position = []
    for index, text in enumerate(log_content):
        if index == 0:  # skip the info list
            multiples = text['multiples']
        else:
            multi_series.append(text['Multiplier'])
            all_position = np.hsplit(np.array(text['Obs_Record']), 4)[0].squeeze()
            max_position.append(max(np.abs(all_position)))
    # plot figure
    max_position_plot = plt.figure()
    plt.axes(xscale="log")
    plt.plot(multi_series, max_position)
    plt.ylim((0, 3))
    ax = plt.gca()
    plt.title('Maximum position at different pole lengths')
    plt.xlabel('multiplier')
    plt.ylabel('maximum position')
    plt.axhline(y=stable_position, color='red', linestyle='--')

    # draw the intersections
    if intersection:
        _x = []
        _y = []
        inter_points = []
        for index in range(len(multi_series) - 1):
            if max_position[index] >= stable_position > max_position[index + 1] or \
                    max_position[index] <= stable_position < max_position[index + 1]:
                _x.append(multi_series[index])
                _y.append(max_position[index])
                _x.append(multi_series[index + 1])
                _y.append(max_position[index + 1])
                inter_points.append(np.interp(stable_position, _y, _x))
                _x = []
                _y = []

        _margin_list = []
        inter_points = [round(i, 2) for i in inter_points]
        len_inter = len(inter_points)
        if len_inter >= 2:
            # points pre-process
            if len_inter == 2:
                pass
            else:
                _range = []
                for i in range(len_inter - 1):
                    _margin_list.append(inter_points[i + 1] - inter_points[i])
                    _margin_list = [round(i, 2) for i in _margin_list]
                _max_index = _margin_list.index(max(_margin_list))
                inter_points = inter_points[_max_index:_max_index + 2]
            # draw points and infobox
            for point in inter_points:
                plt.plot(point, stable_position, 'ro')
                plt.axvline(x=point, color='red', linestyle=':')
            _min = min(inter_points)
            _max = max(inter_points)
            stable_margin = _max - _min
            info_box = '\n'.join((r'$min=%.2f$' % (_min,),
                                  r'$max=%.2f$' % (_max,),
                                  r'$ranges=%.2f$' % (stable_margin,)))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.15, info_box, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    if show_fig:
        max_position_plot.show()

    return stable_margin



def generate_multi_series(max_multi=20, resolution=20, precision=3):
    temp_a = np.linspace(1, max_multi, resolution).tolist()
    temp_b = np.linspace(1 / 15, 1, 25).tolist()
    multi_series = temp_b[0:len(temp_b)-1] + temp_a
    multi_series = np.round(multi_series, precision)
    return multi_series


def generate_RL_multi_poles_test(model_path, figures_dir, num_steps=500, max_multi=20, samplings=30, action_type=""):
    """Parameters"""
    resolution = 21  # IMPORTANT!!! Should be same as the value in the model
    actions = np.linspace(-1, 1, resolution)
    # config
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    DRL_model = torch.load(model_path)
    DRL_model.eval()
    DRL_model.to(device)
    # experiments parameters
    multi_series = generate_multi_series(max_multi=max_multi,resolution=samplings)
    # generate log file
    log_content = []
    # log header
    info_experiment = {}
    info_experiment['num_steps'] = num_steps
    info_experiment['disturbances'] = 'None'
    info_experiment['variable'] = 'Pole Length'
    info_experiment['multiples'] = multi_series.tolist()
    log_content.append(info_experiment)
    result_dir = os.path.join(os.getcwd(), figures_dir)
    if not os.path.exists(result_dir):
        try:
            os.mkdir(result_dir)
        except FileNotFoundError:
            os.mkdir(os.path.dirname(result_dir))
            os.mkdir(result_dir)
    assert action_type in ['Discrete', 'Continuous'], "The type of test only can be 'Discrete' and 'Continuous'"
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
                obs = torch.tensor([obs], dtype=torch.float).to(device)
                if action_type == "Discrete":
                    action_idx = DRL_model(obs).argmax().item()
                    action = actions[action_idx]
                    obs, reward, done, info = env.step(action)
                    # record data
                    obs_record.append(obs.tolist())
                    action_record.append(action.tolist())
                elif action_type == "Continuous":
                    action = DRL_model(obs).item()
                    obs, reward, done, info = env.step(action)
                    # record data
                    obs_record.append(obs.tolist())
                    action_record.append(action)

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

def generate_RL_noise_test(model_path, figures_dir, num_steps=600, max_sigma=0.5, samplings=40, action_type=""):
    """Parameters"""
    resolution = 21  # IMPORTANT!!! Should be same as the value in the model
    actions = np.linspace(-0.5, 0.5, resolution)
    # config
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    DRL_model = torch.load(model_path)
    DRL_model.eval()
    DRL_model.to(device)
    env = ContinuousCartPole_V1()
    # experiments parameters
    sigma_series = np.linspace(0, max_sigma, samplings)
    # generate log file
    log_content = []
    # log header
    info_experiment = {}
    info_experiment['num_steps'] = num_steps
    info_experiment['disturbances'] = 'None'
    info_experiment['variable'] = 'sigma of noise'
    info_experiment['noise_series'] = sigma_series.tolist()
    log_content.append(info_experiment)
    result_dir = os.path.join(os.getcwd(), figures_dir)
    if not os.path.exists(result_dir):
        try:
            os.mkdir(result_dir)
        except FileNotFoundError:
            os.mkdir(os.path.dirname(result_dir))
            os.mkdir(result_dir)
    assert action_type in ['Discrete', 'Continuous'], "The type of test only can be 'Discrete' and 'Continuous'"
    with open(os.path.join(result_dir, 'logger.json'), 'w') as file:
        for index, sigma in enumerate(tqdm(sigma_series)):
            # clear buffer
            obs_record = []
            action_record = []
            info = {}
            # init env
            obs = env.reset()
            obs_controller = obs.copy()
            for step in range(num_steps):
                # convert to tensor
                obs_controller = torch.tensor([obs_controller], dtype=torch.float).to(device)
                if action_type == "Discrete":
                    action_idx = DRL_model(obs_controller).argmax().item()
                    action = actions[action_idx]
                    obs, reward, done, info = env.step(action)
                    obs_controller = obs.copy()
                    obs_controller[2] += np.random.normal(0, sigma)
                    # record data
                    obs_record.append(obs.tolist())
                    action_record.append(action.tolist())
                elif action_type == "Continuous":
                    action = DRL_model(obs_controller).item()
                    obs, reward, done, info = env.step(action)
                    obs_controller = obs.copy()
                    obs_controller[2] += np.random.normal(0, sigma)
                    # record data
                    obs_record.append(obs.tolist())
                    action_record.append(action)

            info['sigma'] = sigma
            info['Obs_Record'] = obs_record
            info['Action_Record'] = action_record
            env.close()
            # plot data
            plot_action(action_record, num_steps, max_value=1.2, show_fig=False,
                        save_path=os.path.join(result_dir, 'action_p{}_sigma_{}.png'.format(index, sigma)))
            plot_states(obs_record, num_steps, show_fig=False,
                        save_path=os.path.join(result_dir, 'state_p{}_sigma_{}.png'.format(index, sigma)))
            log_content.append(info)

        json_f = json.dumps(log_content, indent=3)
        file.write(json_f)



def analyse_all_models_pole_len(models_path, figure_dir, max_multi=40, samplings=40, action_type=''):
    model_lists = os.listdir(models_path)
    model_lists.remove('logger.json')
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    logger_content = []
    margin_data = {}
    # mkdir
    angle_analysis_dir = os.path.join(figure_dir,'1_angle_analysis')
    position_analysis_dir = os.path.join(figure_dir, '2_position_analysis')
    if not os.path.exists(angle_analysis_dir):
        os.mkdir(angle_analysis_dir)
    if not os.path.exists(position_analysis_dir):
        os.mkdir(position_analysis_dir)

    with open(os.path.join(figure_dir, 'angle_position_record.json'), 'a') as file:
        for model in tqdm(model_lists):
            temp_margin = {}
            model_path = os.path.join(models_path, model)
            imgs_dir = os.path.join(figure_dir, model.split('.')[0])
            if not os.path.exists(imgs_dir):
                os.mkdir(imgs_dir)
                generate_RL_multi_poles_test(model_path, imgs_dir, max_multi=max_multi,
                                             samplings=samplings, action_type=action_type)
            else:
                pass
            angle_margin = analyse_angle_pole_len(log_path=os.path.join(imgs_dir, 'logger.json'),
                                                  save_path=os.path.join(angle_analysis_dir,
                                                                         'Angle_{}.png'.format(model.split('.')[0])))
            position_margin = analyse_position_pole_len(log_path=os.path.join(imgs_dir, 'logger.json'),
                                                        save_path=os.path.join(position_analysis_dir,
                                                                               'Position_{}.png'.format(model.split('.')[0])))
            temp_margin['Angle_Margin'] = angle_margin
            temp_margin['Position_Margin'] = position_margin
            margin_data[model.split('.')[0]] = temp_margin
        logger_content.append(margin_data)
        json_f = json.dumps(logger_content, indent=3)
        file.write(json_f)

def analyse_all_models_noise(models_path, figure_dir, max_sigma=0.5, samplings=40, action_type=''):
    assert os.path.exists(models_path), 'Please check the dirname of the models'
    model_lists = os.listdir(models_path)
    model_lists.remove('logger.json')
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    logger_content = []
    margin_data = {}
    # mkdir
    angle_analysis_dir = os.path.join(figure_dir,'1_angle_analysis')
    position_analysis_dir = os.path.join(figure_dir, '2_position_analysis')
    if not os.path.exists(angle_analysis_dir):
        os.mkdir(angle_analysis_dir)
    if not os.path.exists(position_analysis_dir):
        os.mkdir(position_analysis_dir)

    with open(os.path.join(figure_dir, 'angle_position_record.json'), 'a') as file:
        for model in tqdm(model_lists):
            temp_margin = {}
            model_path = os.path.join(models_path, model)
            imgs_dir = os.path.join(figure_dir, model.split('.')[0])
            if not os.path.exists(imgs_dir):
                os.mkdir(imgs_dir)
                generate_RL_noise_test(model_path, imgs_dir, max_sigma=max_sigma,
                                       samplings=samplings, action_type=action_type)
            else:
                pass
            angle_margin = analyse_angle_noise(log_path=os.path.join(imgs_dir, 'logger.json'),
                                                  save_path=os.path.join(angle_analysis_dir,
                                                                         'Angle_{}.png'.format(model.split('.')[0])))
            position_margin = analyse_position_noise(log_path=os.path.join(imgs_dir, 'logger.json'),
                                                        save_path=os.path.join(position_analysis_dir,
                                                                               'Position_{}.png'.format(model.split('.')[0])))
            temp_margin['Angle_Margin'] = angle_margin
            temp_margin['Position_Margin'] = position_margin
            margin_data[model.split('.')[0]] = temp_margin
        logger_content.append(margin_data)
        json_f = json.dumps(logger_content, indent=3)
        file.write(json_f)


def analyse_shift(figure_dir):
    # plot data
    max_angle_shift = []
    max_position_shift = []
    x_axis = []
    fig_path_angle = os.path.join(figure_dir, 'angle_shift.png')
    fig_path_position = os.path.join(figure_dir, 'position_shift.png')
    if os.path.exists(fig_path_position):
        os.remove(fig_path_position)
    if os.path.exists(fig_path_angle):
        os.remove(fig_path_angle)

    figure_dir_list = os.listdir(figure_dir)
    # filter
    figure_dir_list.remove('1_angle_analysis')
    figure_dir_list.remove('2_position_analysis')
    figure_dir_list.remove('angle_position_record.json')
    figure_dir_list.sort(key = lambda i:int(i.split('_')[-2].split('iter')[1]))
    # inter
    for _figure in tqdm(figure_dir_list):
        logger_path = os.path.join(figure_dir, os.path.join(_figure, 'logger.json'))
        with open(logger_path, 'r') as f:
            log_content = json.load(f)
        temp_angle = []
        temp_position = []
        for ind, record in enumerate(log_content):
            if ind == 0:
                continue
            else:
                if record["Multiplier"] == 1.0:
                    all_position = np.hsplit(np.array(record['Obs_Record']), 4)[0].squeeze()
                    all_angle = np.hsplit(np.array(record['Obs_Record']), 4)[2].squeeze()
                    temp_position.append(max(np.abs(all_position)))
                    temp_angle.append(max(np.abs(all_angle)))
                else:
                    continue
        max_angle_shift.append(max(temp_angle) *(180 / math.pi))
        max_position_shift.append(max(temp_position))
        assert len(max_angle_shift) == len(max_position_shift), \
            'the size of angle shift should be same as position shift'
        x_axis = [i+1 for i in range(len(max_angle_shift))]

    print(max_angle_shift)
    print(max_position_shift)
    plt.figure()
    plt.plot(x_axis, max_angle_shift)
    plt.title('Max angle shift of all models')
    plt.xlabel('iter')
    plt.ylabel('degree')
    plt.ylim((0, 12))
    plt.grid(True)

    plt.savefig(fig_path_angle)
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(x_axis, max_position_shift)
    plt.title('Max position shift of all models')
    plt.xlabel('iter')
    plt.ylabel('m')
    plt.ylim((0, 2.4))
    plt.grid(True)
    plt.savefig(fig_path_position)
    plt.show()
    plt.close()
