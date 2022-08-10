"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

Continuous version by Ian Danforth
Modified by Peilin Wu
"""

import math
import sys

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import utils.rendering as rendering


class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, mass_cart=1.0, mass_pole=0.1, length=0.5,
                 disturb_type=None, disturb_starts=None, sensor_index=[0, 1, 2, 3], gaussian_std=0.1,
                 random_len=None):
        """
        :param disturb_type: chose the type of disturbances, 'Gauss Noise', 'Sensor Failure'
        :param disturb_starts: when the disturbances starts
        :param sensor_index: sensors effected by disturbances, 0->position, 1->velocity, 2->angle, 3->angular velocity
        :param random_len: the standard variance of normal distribution, e.g. 0.1
        :param penalise: the ways to generate rewards,
        """
        self.tick = 0
        self.gravity = 9.8
        self.masscart = mass_cart
        self.masspole = mass_pole
        self.total_mass = (self.masspole + self.masscart)
        # actually half the pole's length
        self.length = np.random.normal(length, random_len) if random_len is not None else length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0
        self.disturb_starts = disturb_starts if disturb_starts is not None else sys.maxsize
        self.disturb_type = disturb_type
        self.sensor_index = sensor_index
        self.gaussian_std = gaussian_std

        # check the disturbances configuration
        assert self.disturb_type in [None, 'Gauss Noise', 'Sensor Failure'], \
            "The following types of disturbances are available: 'Gauss Noise', 'Sensor Failure'"
        for i in self.sensor_index:
            assert i in [0, 1, 2, 3], '{} is not a valid index of sensor.'.format(i)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None       # real states
        self.obs_state = None   # observed state, includes disturbances

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        self.tick += 1
        # Cast action to float to strip np trappings
        action = np.clip(action, self.min_action, self.max_action)  # ensure the action in a reasonable range
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        self.obs_state = self.state
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)

        # add disturbances
        temp_state = list(self.obs_state)
        if self.tick >= self.disturb_starts:
            if self.disturb_type == 'Gauss Noise':
                for ind_sensor in self.sensor_index:
                    temp_state[ind_sensor] *= (1 + np.random.normal(0, self.gaussian_std))
            elif self.disturb_type == 'Sensor Failure':
                for ind_sensor in self.sensor_index:
                    temp_state[ind_sensor] = 0.0 # no signal return
            else:
                pass
        self.obs_state = tuple(temp_state) # update state

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                # logger.warn("""
                #     You are calling 'step()' even though this environment has already returned
                #     done = True. You should always call 'reset()' once you receive 'done = True'
                #     Any further steps are undefined behavior.
                # """)
                pass
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.obs_state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width /world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            # from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()