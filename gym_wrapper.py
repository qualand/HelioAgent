# Gymnasium wrapper around AimingEnv, so any standard RL library (Stable-Baselines3 PPO, RecurrentPPO, ...) can drive it.
#   action      : MultiDiscrete([12] * n_heliostats), one menu point per mirror
#   observation : AimingEnv._obs() scaled to about 0..1
#   terminated  : a wall crossed 1200 C (failure); truncated: sun down or episode time up

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from environment_class import AimingEnv


class HelioAimingGym(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, env_kwargs=None, reset_kwargs=None, shutdown_penalty=2.0):
        super().__init__()
        self.env = AimingEnv(**(env_kwargs or {}))
        self.shutdown_penalty = shutdown_penalty                # taken off the reward when a wall passes 1200 C and the episode ends:
                                                                # without it, ending a bad episode (future = 0) beats continuing at a negative reward
        self.reset_kwargs = dict(reset_kwargs or {})            # empty = the training episode defined in AimingEnv.reset; the analyses pass day, hour_range, start

        n_points = len(self.env.grid)
        self.action_space = spaces.MultiDiscrete([n_points] * self.env.n)

        half_width = self.env.grid[:, 0].max()
        n_columns = self.env.field.thermal.receiver.nx                  # curtain columns in Janna's grid (5)
        self.scale = np.concatenate([
            [24.0, 360.0, 90.0, 1000.0],                       # hour, sun azimuth, sun elevation, DNI
            np.full(2 * self.env.n, half_width),               # aim offsets on the aperture
            [3000.0, 1500.0, 1.0, 1000.0, 1200.0, 1200.0],     # power kW, peak flux kW/m2, intercept, particle out C, receiver wall C, SNOUT wall C
            np.full(n_columns, 1000.0),                        # outlet temperature per curtain column, C
        ]).astype(np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=self.scale.shape, dtype=np.float32)   # scaled values sit within about -1..3
        self.metrics = None                                     # full metrics of the last step, for logging

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset(seed=seed, **{**self.reset_kwargs, **(options or {})})
        self.metrics = self.env.last
        return obs / self.scale, {}

    def step(self, action):
        obs, reward, done, metrics = self.env.step(np.asarray(action, dtype=int))
        self.metrics = metrics
        terminated = bool(done and metrics['why'] == 'wall_over_temp')
        truncated = bool(done and not terminated)
        if terminated:
            reward -= self.shutdown_penalty
        info = {key: value for key, value in metrics.items() if np.isscalar(value)}    # scalars only, as loggers expect
        return obs / self.scale, float(reward), terminated, truncated, info