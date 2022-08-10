import numpy as np
import time
from envs.continuous_CartPole import ContinuousCartPoleEnv
from utils.plot_utils import plot_action, plot_states
from scipy import linalg


def generate_K_from_ARE(weight_R=1, weight_Q=4, m_cart=1, m_pole=0.1,
                        len_pole=0.5, grav=9.8):
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


if __name__ == "__main__":
    # init env
    # env = ContinuousCartPoleEnv(disturb_type='Gauss Noise', sensor_index=[0, 1, 2, 3],
    #                             disturb_starts=100, gaussian_std=0.3)
    env = ContinuousCartPoleEnv()
    obs = env.reset()
    num_steps = 500
    # init controller
    K = generate_K_from_ARE(len_pole=1)
    # record data
    action_record = []
    obs_record = []
    for step in range(num_steps):
        action = optimal_controller(K, obs)
        obs, reward, done, info = env.step(action)
        # record data
        obs_record.append(obs)
        action_record.append(action)
        # UI
        env.render(mode='human')
        time.sleep(0.001)
    env.close()
    # plot data
    plot_action(action_record, num_steps, max_value=1.2)
    plot_states(obs_record, num_steps)

