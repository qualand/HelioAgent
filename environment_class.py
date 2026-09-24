# Environment
#
# Holds a class, because an episode has to remember things between calls:
# what time it is, where every heliostat is currently aiming, what the last
# flux was. AimingEnv stores those, and its step() is what changes them. It
# calls the helio_env functions to do the physics, then keeps the results.
#
# No reward and no receiver thermal yet.

import numpy as np
from functions_4_environment import build_field, receiver_axes, aim_grid, set_aims_discrete, trace, sun_at

class AimingEnv:

    def __init__(self, dt_min=1.0, nray=1e5, peak_limit_kw_m2=900.0):
        # Build field, once. This is the slow part and it never repeats.
        self.field, self.PT = build_field()
        self.axes = receiver_axes(self.field)
        self.n = len(self.field.heliostats)
        self.grid = aim_grid(self.field, n_cols=3, n_rows=4, margin=0.5)
        
        self.dt = dt_min                    # minutes of plant time per step
        self.nray = nray
        self.peak_limit = peak_limit_kw_m2  # hard limit, ends the episode if exceeded

        # state, filled by reset()
        self.day = None
        self.hour = None
        self.choice = None # One integer per heliostat. 12 options. 
        self.offsets = None
        self.last = None

    # ---- Reset box -------------------------------------------------------
    def reset(self, day=172, hour=8.0, seed=None):
        # Start a new episode. Return the first observation.
        rng = np.random.default_rng(seed)
        self.day = day
        self.hour = hour + rng.uniform(0.0, 0.5)     # random start within 30 min
        self.choice = np.full(self.n, 4)
        self.offsets = set_aims_discrete(self.field, self.choice, self.grid, self.axes)
        self.last = dict(power_kw=0.0, peak_kw_m2=0.0, intercept=0.0)
        return self._obs()

    # ---- one step: Action -> Ray trace -> Terminal ------------------------
    def step(self, choice):
        # Choice : array (n,) of ints 0..11, menu point per heliostat..
        # Returns (obs, metrics, done, why).

        # Action: put every heliostat on its grid point
        self.choice = np.asarray(choice, dtype = int)
        self.offsets = set_aims_discrete(self.field, self.choice, self.grid, self.axes)

        # Clock
        self.hour += self.dt / 60.0
        az, el = sun_at(self.day, self.hour)

        # Ray trace
        res, flux = trace(self.PT, self.field, az, el, nray=self.nray)

        metrics = dict(
            power_kw=res['Absorbed power (kW)'],
            peak_kw_m2=float(flux.max()),
            intercept=res['Intercept efficiency (%)'] / 100.0,
            sun_el=el,
        )
        self.last = metrics

        done, why = self._terminal(metrics)
        return self._obs(), metrics, done, why

    # ---- Observation box ----------------------------------------------------
    def _obs(self):
        # What the agent sees. One flat float vector, fixed length.
        az, el = sun_at(self.day, self.hour)
        return np.concatenate([
            [self.hour, az, el],                     # clock and sun
            self.offsets.ravel(),                    # where everyone is aiming
            [self.last['power_kw'], self.last['peak_kw_m2'], self.last['intercept']],
        ]).astype(np.float32)

    # ---- Terminal check box ---------------------------------------------------
    def _terminal(self, m):
        if m['peak_kw_m2'] > self.peak_limit:
            return True, 'flux_limit'
        if m['sun_el'] < 7.0:
            return True, 'sun_down'
        return False, ''

if __name__ == "__main__":
    # The sun moves, the aims do not.
    import time
    env = AimingEnv()
    obs = env.reset(seed=0)
    print("obs length:", len(obs), " first 6:", np.round(obs[:6], 3))

    for k in range(5):
        t0 = time.time()
        obs, m, done, why = env.step(np.full(env.n, 4))
        print(f"step {k}  hour {env.hour:7.4f}  power {m['power_kw']:8.1f}"
              f"  peak {m['peak_kw_m2']:7.1f}  done={done} {why}  ({time.time()-t0:.1f} s)")
        if done:
            break