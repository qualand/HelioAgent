# Environment
#
# Holds a class, because an episode has to remember things between calls:
# what time it is, where every heliostat is currently aiming, what the last
# flux was. AimingEnv stores those, and its step() is what changes them. It
# calls the helio_env functions to do the physics, then keeps the results.
#
# No reward and no receiver thermal yet.

import numpy as np
from functions_4_environment import build_field, receiver_axes, aim_grid, offsets_to_xyz, set_aim_xyz, set_aims_discrete, trace, sun_at
from helio_kinematics import HelioKinematics

class AimingEnv:

    def __init__(self, dt_min=1.0, slew_mrad_s = 1.3, nray=1e5, peak_limit_kw_m2=900.0):
        # Build field, once. This is the slow part and it never repeats.
        self.field, self.PT = build_field()
        self.axes = receiver_axes(self.field)
        self.n = len(self.field.heliostats)
        self.grid = aim_grid(self.field, n_cols=3, n_rows=4, margin=0.15)
        self.kin = HelioKinematics(self.field, slew_mrad_s)
        
        self.dt = dt_min                    # minutes of plant time per step
        self.nray = nray
        self.peak_limit = peak_limit_kw_m2  # hard limit, ends the episode if exceeded

        # State: what the environment remembers between steps. Filled by reset(), changed by step()
        self.day = None # day of year, fixed for the episode
        self.hour = None # solar time; step() adds dt/60
        self.choice = None # One integer per heliostat. 12 options. 
        self.offsets = None # (n, 2) floats: the (dx, dv) on the plate for those menu points
        self.last = None # dict: power_kw, peak_kw_m2, intercept from the last ray trace

    # ---- Reset box -------------------------------------------------------
    def reset(self, day=172, hour=8.0, seed=None, stow=False):
        # Start a new episode. Return the first observation.
        rng = np.random.default_rng(seed)
        self.day = day
        self.hour = hour + rng.uniform(0.0, 0.5)     # random start within 30 min
        self.choice = np.full(self.n, 4)
        self.offsets = self.grid[self.choice]
        
        az, el = sun_at(self.day, self.hour)
        target = offsets_to_xyz(self.offsets, self.axes)
        az_t, el_t = self.kin.target_angles(target, az, el)
        if stow:
            self.kin.snap(*self.kin.stow_angles(target)) # face up, as in heliostat.create_geometry
        else:
            self.kin.snap(az_t, el_t) # already tracking point 4
        
        self.last = dict(power_kw=0.0, peak_kw_m2=0.0, intercept=0.0)
        return self._obs()

    # ---- one step: Action -> Kinematics -> Ray trace -> Terminal ------------------------
    def step(self, choice):
        # Choice : array (n,) of ints 0..11, menu point per heliostat.
        # Returns (obs, metrics, done, why).

        # Action: put every heliostat on its grid point
        self.choice = np.asarray(choice, dtype = int)
        self.offsets = self.grid[self.choice]
        target = offsets_to_xyz(self.offsets, self.axes)

        # Clock
        self.hour += self.dt / 60.0
        az, el = sun_at(self.day, self.hour)
        
        # Kinematics: motors move toward the wanted normal, speed limited
        az_t, el_t = self.kin.target_angles(target, az, el)
        self.kin.move_toward(az_t, el_t, self.dt*60)
        dist = np.linalg.norm(target - self.kin.pos, axis =1) #?????????????? Check
        set_aim_xyz(self.field, self.kin.aim_points(az, el, dist))
        
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
    import matplotlib.pyplot as plt
    env = AimingEnv(peak_limit_kw_m2=1e9)
    env.reset(seed=0, stow=True)
    minutes, power, peak = [], [], []
    for k in range(18):
        obs, m, done, why = env.step(np.full(env.n, 4))
        minutes.append(k + 1); power.append(m['power_kw']); peak.append(m['peak_kw_m2'])
        print(f"min {k+1:2d}  power {m['power_kw']:8.1f}  peak {m['peak_kw_m2']:7.1f}")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    a1.plot(minutes, power, marker='o'); a1.set_ylabel('absorbed power (kW)'); a1.grid(alpha=0.3)
    a2.plot(minutes, peak, marker='o', color='C1'); a2.axhline(900, color='k', ls='--', label='900 kW/m2')
    a2.set_ylabel('peak flux (kW/m2)'); a2.set_xlabel('minute after 08:00 (solar)'); a2.grid(alpha=0.3); a2.legend()
    fig.suptitle('Startup from stow, all heliostats on point 4, 1.3 mrad/s')
    fig.tight_layout(); plt.savefig('startup_from_stow.png', dpi=150)