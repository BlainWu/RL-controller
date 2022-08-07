import numpy as np
import matplotlib.pyplot as plt
import math


def plot_states(data, steps, grid=False, threshold=True, auto_adjust=False):
    data = np.array(data)
    axis = [i for i in range(steps)]
    # unzip data
    x, x_dot, theta, theta_dot = \
        np.hsplit(data, 4)[0].squeeze(), np.hsplit(data, 4)[1].squeeze(), \
        np.hsplit(data, 4)[2].squeeze(), np.hsplit(data, 4)[3].squeeze()
    theta = [i*(180 / math.pi) for i in theta]
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
    plt.ylabel('rad/s')
    plt.plot(axis, theta_dot)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10,
                        right=0.95, hspace=0.35, wspace=0.35)

    state.show()


def plot_action(data, steps, grid=False, max_value=0.5):
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
    act_plot.show()