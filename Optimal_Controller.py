import gym
import matplotlib.pyplot as plt
import numpy as np
import time
from utils.continuous_CartPole import ContinuousCartPoleEnv
from scipy import linalg


def generate_K_from_ARE(weight_R = 1, weight_Q = 4, m_cart = 1, m_pole = 0.1, len_pole = 0.5,
                      grav = 9.8):
    # system equation
    m_all = m_cart + m_pole
    temp = grav / (len_pole * (4.0 / 3.0 - m_pole / m_all))
    A = np.array([[0, 1, 0, 0],
                  [0, 0, temp, 0],
                  [0, 0, 0, 1],
                  [0, 0, temp, 0]])
    temp = -1 / (len_pole*(4.0 / 3.0 - m_pole/m_all))
    B = np.array([[0],
                  [1/m_all],
                  [0],
                  [temp]])

    # cost function
    R = np.eye(weight_R, dtype=int)
    Q = np.eye(weight_Q, dtype=int)

    # solve Riccati Equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # get the optimal K = inv(R) B^T P
    K = np.dot(np.linalg.inv(R), np.dot(B.T, P))
    return K


def optimal_controller(K, state):
    u = -np.dot(K, state)
    u = u / 30.0  # the force range [-30, 30] N
    return u


def plot_action(data, steps, grid = False, max_value = 0.5):
    x_axis = [i for i in range(steps)]
    import matplotlib.pyplot as plt
    import numpy as np
    # calculate metric
    var = round(np.var(data), 4)
    mean = round(np.mean(data), 4)
    plt.figure()
    ax = plt.gca()
    plt.title('Control Signal')
    plt.plot(x_axis, data)
    plt.axis([0, steps, -max_value, max_value])
    plt.text(0.1, 0.8, r'$\mu={},\ \sigma={}$'.format(mean, var), transform=ax.transAxes)
    plt.grid(grid)
    plt.show()


def plot_states(data, steps):
    x_axix = [i for i in range(steps)]
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure()
    plt.subplots(411)


if __name__ == "__main__":
    env = ContinuousCartPoleEnv()
    obs = env.reset()
    num_steps = 300
    K = generate_K_from_ARE()
    action_record = []
    for step in range(num_steps):
        action = optimal_controller(K, obs)
        action_record.append(action)
        obs, reward, done, info = env.step(action)
        env.render(mode = 'human')
        time.sleep(0.001)

    env.close()

    # plot datas
    plot_action(action_record, num_steps)