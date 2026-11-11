import numpy as np
import gymnasium as gym
from gymnasium import spaces
from environment_class import AimingEnv
from reward import local_credit


#--- Gymnasium wrapper around AimingEnv, so a standard RL loop (train_ppo.py, or Stable-Baselines3) can drive it
class HelioAimingGym(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, env_kwargs = None, reset_kwargs = None, shutdown_penalty = 2.0):
        super().__init__()

        #--- The environment and the episode it starts
        self.env = AimingEnv(**(env_kwargs or {}))
        self.reset_kwargs = dict(reset_kwargs or {})            # Empty = the training episode defined in AimingEnv.reset; the analyses pass day, hour_range, start
        self.shutdown_penalty = shutdown_penalty                # Taken off the reward when a wall passes the limit and the episode ends:
                                                                # without it, ending a bad episode (future = 0) beats continuing at a negative reward

        #--- Spaces: one menu point per heliostat; observation = AimingEnv._obs() scaled to about 0..1
        n_points = len(self.env.grid)
        self.action_space = spaces.MultiDiscrete([n_points] * self.env.n)
        half_width = self.env.grid[:, 0].max()
        n_columns = self.env.field.thermal.receiver.nx          # Curtain columns in Janna's grid (5)
        self.scale = np.concatenate([
            [24.0, 360.0, 90.0, 1000.0],                       # Hour, sun azimuth, sun elevation, DNI
            np.full(2 * self.env.n, half_width),               # Aim offsets on the aperture
            [3000.0, 1500.0, 1.0, 1000.0, 1200.0, 1200.0],     # Power kW, peak flux kW/m2, intercept, particle out C, receiver wall C, SNOUT wall C
            np.full(n_columns, 1000.0),                        # Outlet temperature per curtain column, C
        ]).astype(np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape = self.scale.shape, dtype = np.float32)   # Scaled values sit within about -1..3

        #--- Results
        self.metrics = None                                     # Full metrics of the last step, for logging
        return

    #--- Reset: options override the stored reset arguments (day, hour_range, start)
    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        obs = self.env.reset(seed = seed, **{**self.reset_kwargs, **(options or {})})
        self.metrics = self.env.last
        return obs / self.scale, {}

    #--- Step: terminated = a wall crossed the limit (failure, penalty applied); truncated = sun down or time up
    def step(self, action):
        obs, reward, done, metrics = self.env.step(np.asarray(action, dtype = int))
        self.metrics = metrics
        terminated = bool(done and metrics['why'] == 'wall_over_temp')
        truncated = bool(done and not terminated)
        if terminated:
            reward -= self.shutdown_penalty
        info = {key: value for key, value in metrics.items() if np.isscalar(value)}    # Scalars only, as loggers expect
        info['heliostat_credit'] = local_credit(metrics, penalty = self.shutdown_penalty if terminated else 0.0, **self.env.weights)   # Each mirror's share of the reward, for the learner
        return obs / self.scale, float(reward), terminated, truncated, info