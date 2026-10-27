# Environment
#
# Holds a class, because an episode has to remember things between calls:
# what time it is, where every heliostat is aiming, what the last flux and
# temperatures were. AimingEnv stores those, and step() changes them. It calls
# functions_4_environment for the physics and keeps the results.
#
# Episode = normal operation: mirrors tracking, receiver at setpoint, SNOUT walls hot.
# Startup from stow (stow=True) is kept for later.

import numpy as np
from functions_4_environment import build_field, receiver_axes, aim_grid, offsets_to_xyz, set_aim_xyz, trace, warm_up_snout, sun_at, spread_by_distance, DNI_MODELS
from helio_kinematics import HelioKinematics
from reward import score

class AimingEnv:

    def __init__(self, dt_min=1.0, slew_mrad_s=1.3, nray=1e5, episode_minutes=120, sun_min_el=7.0, dni_model='clearsky',
                 wall_limit_c=1200.0, low_temp_c=765.0, w_uneven=1.0, w_travel=0.05, w_setpoint=0.5):
        # Build field, once. This is the slow part and it never repeats.
        self.field, self.PT = build_field()
        self.axes = receiver_axes(self.field)
        self.n = len(self.field.heliostats)
        self.grid = aim_grid(self.field, n_cols=3, n_rows=4, margin=0.15)
        self.kin = HelioKinematics(self.field, slew_mrad_s)

        self.dt = dt_min                        # minutes of plant time per step
        self.nray = nray
        self.episode_minutes = episode_minutes
        self.wall_limit_c = wall_limit_c        # RSLE-57 limit, receiver walls and SNOUT: stop above it
        self.low_temp_c = low_temp_c            # hot storage needs 765 C (G3P3): stop below it
        self.sun_min_el = sun_min_el            # degrees; episode ends when the sun drops below it
        self.dni_at = DNI_MODELS[dni_model]     # 'constant', 'clearsky' or 'tmy': function (day, hour, sun_el) -> W/m2
        self.weights = dict(w_uneven=w_uneven, w_travel=w_travel, w_setpoint=w_setpoint)

        # State: what the environment remembers between steps. Filled by reset(), changed by step()
        self.day = None        # day of year, fixed for the episode
        self.hour = None       # solar time; step() adds dt/60
        self.minute = None     # minutes since reset
        self.dni = None        # W/m2 of the current minute
        self.choice = None     # one integer per heliostat, 12 options
        self.offsets = None    # (n, 2) floats: the (dx, dv) on the aperture for those menu points
        self.last = None       # metrics from the last ray trace

    # ---- Reset box -------------------------------------------------------
    def reset(self, day=355, hour_range=(14.0, 14.5), seed=None, start='random', stow=False):
        rng = np.random.default_rng(seed)
        self.day = day
        self.hour = rng.uniform(*hour_range)      # random start inside operating hours
        self.minute = 0
        if start == 'random':
            self.choice = rng.integers(0, len(self.grid), self.n)    # as Carballo et al. 2025: random aim points at reset
        elif start == 'spread':
            self.choice = spread_by_distance(self.kin.pos, len(self.grid))
        else:
            self.choice = np.full(self.n, 4)
        self.offsets = self.grid[self.choice]

        az, el = sun_at(self.day, self.hour)
        target = offsets_to_xyz(self.offsets, self.axes)
        az_t, el_t = self.kin.target_angles(target, az, el)
        if stow:
            self.kin.snap(*self.kin.stow_angles(target))   # face up, as in heliostat.create_geometry
        else:
            self.kin.snap(az_t, el_t)                        # already tracking their points

        res, flux = self._trace_now(az, el, target)
        res.update(warm_up_snout(self.field, self.PT, t_aperture_c=res['particle_out_c'], dt_s=self.dt * 60))   # SNOUT walls at steady state
        self.last = self._metrics(res, flux, el)
        return self._obs()

    # ---- one step: Action -> Kinematics -> Ray trace -> Thermal -> Reward -> Terminal ----
    def step(self, choice):
        # choice: array (n,) of ints 0..11, menu point per heliostat. Returns (obs, reward, done, metrics).
        self.choice = np.asarray(choice, dtype=int)
        self.offsets = self.grid[self.choice]
        target = offsets_to_xyz(self.offsets, self.axes)

        self.hour += self.dt / 60.0
        self.minute += self.dt
        az, el = sun_at(self.day, self.hour)

        az_t, el_t = self.kin.target_angles(target, az, el)
        self.kin.move_toward(az_t, el_t, self.dt * 60)          # motors, speed limited
        res, flux = self._trace_now(az, el, target)

        metrics = self._metrics(res, flux, el)
        self.last = metrics
        done, why = self._terminal(metrics)
        reward, terms = score(metrics, **self.weights)
        metrics.update(terms, reward=reward, why=why)
        return self._obs(), reward, done, metrics

    # Where the mirrors are now -> beam landing points -> DNI of the minute -> ray trace + thermal
    def _trace_now(self, az, el, target):
        dist = np.linalg.norm(target - self.kin.pos, axis=1)      # mirror to wanted point, m
        set_aim_xyz(self.field, self.kin.aim_points(az, el, dist))
        self.dni = max(self.dni_at(self.day, self.hour, el), 1.0)   # 1 W/m2 floor keeps the efficiency ratio finite under clouds
        return trace(self.PT, self.field, az, el, nray=self.nray, dni=self.dni, dt_s=self.dt * 60)

    def _metrics(self, res, flux, sun_el):
        return dict(
            power_kw=res['Absorbed power (kW)'],
            peak_kw_m2=float(flux.max()),
            intercept=res['Intercept efficiency (%)'] / 100.0,
            optical_efficiency=res['Field efficiency (%)'] / 100.0,       # curtain power ÷ (DNI × mirror area)
            sun_el=sun_el,
            dni_w_m2=self.dni,
            particle_out_c=res['particle_out_c'],
            particle_max_c=res['particle_max_c'],
            mass_flow=res['mass_flow'],                                    # kg/s the controller chose to hold 775 C
            particle_kw=res['particle_kw'],                                # power that went into the particles
            receiver_efficiency=res['efficiency'],                         # particle_kw ÷ incident
            wall_max_c=res['wall_max_c'],                                  # receiver cavity walls
            snout_wall_max_c=res['SNOUT wall T max (C)'],
            snout_kw=np.array([res[f'SNOUT {p} absorbed (kW)'] for p in ('top', 'bottom', 'east', 'west')]),
            snout_peak_kw_m2=max(res[f'SNOUT {p} peak flux (kW/m^2)'] for p in ('top', 'bottom', 'east', 'west')),
            motor_travel=self.kin.travel / (2 * self.n * self.kin.rate * self.dt * 60),   # 0 = nobody moved, 1 = every motor at full speed all minute
        )

    # ---- Observation box ----------------------------------------------------
    def _obs(self):
        az, el = sun_at(self.day, self.hour)
        m = self.last
        return np.concatenate([
            [self.hour, az, el, self.dni],            # clock, sun, DNI
            self.offsets.ravel(),                     # where everyone is aiming
            [m['power_kw'], m['peak_kw_m2'], m['intercept'], m['particle_out_c'], m['wall_max_c'], m['snout_wall_max_c']],
        ]).astype(np.float32)

    # ---- Terminal check box ---------------------------------------------------
    def _terminal(self, m):
        if max(m['wall_max_c'], m['snout_wall_max_c']) > self.wall_limit_c:
            return True, 'wall_over_temp'
        if m['particle_out_c'] < self.low_temp_c:
            return True, 'particles_too_cold'
        if m['sun_el'] < self.sun_min_el:
            return True, 'sun_down'
        if self.minute >= self.episode_minutes:
            return True, 'time_up'
        return False, ''

if __name__ == "__main__":
    env = AimingEnv()
    env.reset(seed=0)
    for k in range(30):
        obs, reward, done, m = env.step(np.full(env.n, 4))
        print(f"min {k+1:2d}  DNI {m['dni_w_m2']:5.0f}  power {m['power_kw']:7.1f} kW  flow {m['mass_flow']:5.2f} kg/s  Tout {m['particle_out_c']:6.1f} C  "
              f"wall {m['wall_max_c']:6.1f} C  snout {m['snout_wall_max_c']:6.1f} C  reward {reward:6.3f}")
        if done:
            print('episode ended:', m['why'])
            break