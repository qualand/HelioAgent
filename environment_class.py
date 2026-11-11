import numpy as np
from functions_4_environment import build_field, receiver_axes, aim_grid, center_point, offsets_to_xyz, set_aim_xyz, trace, warm_up_snout, sun_at, hours_above, spread_by_distance, dni_at
from helio_kinematics import HelioKinematics
from reward import score


class AimingEnv:
    def __init__(self, dt_min = 1.0, slew_mrad_s = 1.3, nray = 1e6, episode_minutes = 120, sun_min_el = 7.0, operating_min_el = 15.0,
                 wall_limit_c = 1200.0, w_uneven = 1.0, w_travel = 0.05, w_setpoint = 0.5, trace_seed = 123):

        #--- Plant, built once (the slow part, never repeated)
        self.field, self.PT = build_field()         # Heliostat field, receiver, SNOUT and the SolTrace scene
        self.axes = receiver_axes(self.field)       # Aperture frame: center, sideways, up
        self.n = len(self.field.heliostats)         # 218 heliostats
        self.grid = aim_grid(self.field)            # 3 x 5 menu, 15 aim points on the aperture
        self.center_point = center_point(self.grid) # Point 7: (0, 0), the aperture center
        self.kin = HelioKinematics(self.field, slew_mrad_s)   # Motors: where every mirror points now, slew-rate limited

        #--- Step and episode
        self.dt = dt_min                            # Minutes of plant time per step
        self.nray = nray                            # Rays per trace (1e6: SNOUT maximum within ~20 C of converged on the 5 cm mesh)
        self.trace_seed = trace_seed                # SolTrace random seed; None = a new seed every trace, so ray noise differs step to step
        self.episode_minutes = episode_minutes      # Episode length (min); 120 = 2 h of regular operation
        self.sun_min_el = sun_min_el                # Degrees; the episode ends when the sun drops below it
        self.operating_min_el = operating_min_el    # Degrees; regular operation = sun at least this high, no startup or shutdown

        #--- Safety and reward
        self.wall_limit_c = wall_limit_c            # RSLE-57 limit (C) on the receiver walls and the SNOUT: the episode stops above it
        self.weights = dict(w_uneven = w_uneven, w_travel = w_travel, w_setpoint = w_setpoint)   # Reward weights, see reward.score

        #--- State: what the environment remembers between steps, filled by reset(), changed by step()
        self.day = None        # Day of the year, fixed for the episode
        self.hour = None       # Solar time; step() adds dt / 60
        self.minute = None     # Minutes since reset
        self.dni = None        # W/m2 of the current step
        self.choice = None     # One integer per heliostat, the menu point
        self.offsets = None    # (n, 2) the (dx, dv) on the aperture of those menu points
        self.last = None       # Metrics of the last ray trace
        return

    #--- Reset: start a new episode. The training episode is defined by the defaults: a random day of the year, a random start time such that
    #    the whole episode is inside regular operation hours, every heliostat on the center point (random day and time as Carballo et al. 2025).
    #    The analysis scripts pass day, hour_range and start explicitly. stow = True (mirrors face up) is kept for startup studies.
    def reset(self, day = None, hour_range = None, seed = None, start = 'center', stow = False):
        rng = np.random.default_rng(seed)
        self.day = int(rng.integers(1, 366)) if day is None else day
        if hour_range is None:
            first, last = hours_above(self.day, self.operating_min_el)
            hour_range = (first, max(first, last - self.episode_minutes / 60.0))   # The episode ends before the sun gets low
        self.hour = rng.uniform(*hour_range)
        self.minute = 0
        if start == 'center':
            self.choice = np.full(self.n, self.center_point)
        elif start == 'spread':
            self.choice = spread_by_distance(self.kin.pos, self.grid)                # Image-size rule
        else:
            self.choice = rng.integers(0, len(self.grid), self.n)                    # Random aim points
        self.offsets = self.grid[self.choice]

        az, el = sun_at(self.day, self.hour)
        target = offsets_to_xyz(self.offsets, self.axes)
        az_t, el_t = self.kin.target_angles(target, az, el)
        if stow:
            self.kin.snap(*self.kin.stow_angles(target))   # Face up, as in heliostat.create_geometry
        else:
            self.kin.snap(az_t, el_t)                        # Already tracking their points

        res, flux = self._trace_now(az, el, target)
        res.update(warm_up_snout(self.field, self.PT, t_aperture_c = res['aperture_eff_c'], dt_s = self.dt * 60))   # SNOUT walls at steady state
        self.last = self._metrics(res, flux, el)
        return self._obs()

    #--- Step: action -> kinematics -> ray trace -> thermal -> reward -> terminal check
    #    choice: array (n,) of ints 0..14, menu point per heliostat. Returns (obs, reward, done, metrics).
    def step(self, choice):
        self.choice = np.asarray(choice, dtype = int)
        self.offsets = self.grid[self.choice]
        target = offsets_to_xyz(self.offsets, self.axes)

        self.hour += self.dt / 60.0
        self.minute += self.dt
        az, el = sun_at(self.day, self.hour)

        az_t, el_t = self.kin.target_angles(target, az, el)
        self.kin.move_toward(az_t, el_t, self.dt * 60)          # Motors move toward the new targets, speed limited
        res, flux = self._trace_now(az, el, target)

        metrics = self._metrics(res, flux, el)
        self.last = metrics
        done, why = self._terminal(metrics)
        reward, terms = score(metrics, **self.weights)
        metrics.update(terms, reward = reward, why = why)      # The per-heliostat credit is added by gym_wrapper, which also knows the shutdown penalty
        return self._obs(), reward, done, metrics

    #--- Where the mirrors point now -> beam landing points -> DNI of the step -> ray trace and thermal models
    def _trace_now(self, az, el, target):
        dist = np.linalg.norm(target - self.kin.pos, axis = 1)      # Mirror to wanted point (m)
        set_aim_xyz(self.field, self.kin.aim_points(az, el, dist))
        self.dni = max(dni_at(self.day, self.hour, el), 1.0)        # Clear sky or weather file, chosen in functions_4_environment.py; 1 W/m2 floor keeps the efficiency ratio finite under clouds
        seed = self.trace_seed if self.trace_seed is not None else int(np.random.default_rng().integers(1, 2**31 - 1))
        return trace(self.PT, self.field, az, el, nray = self.nray, dni = self.dni, dt_s = self.dt * 60, seed = seed)

    #--- Metrics of one trace, the dictionary the reward, the credit and the loggers read
    def _metrics(self, res, flux, sun_el):
        return dict(
            power_kw = res['Absorbed power (kW)'],
            peak_kw_m2 = float(flux.max()),
            intercept = res['Intercept efficiency (%)'] / 100.0,
            optical_efficiency = res['Field efficiency (%)'] / 100.0,       # Curtain power / (DNI x mirror area)
            sun_el = sun_el,
            dni_w_m2 = self.dni,
            particle_out_c = res['particle_out_c'],
            particle_max_c = res['particle_max_c'],
            particle_out_columns_c = res['particle_out_columns_c'],         # Outlet temperature per curtain column, for the unevenness term
            aperture_c = res['aperture_eff_c'],                              # What the SNOUT sees through the aperture
            mass_flow = res['mass_flow'],                                    # kg/s the controller chose to hold 775 C
            particle_kw = res['particle_kw'],                                # Power that went into the particles
            receiver_efficiency = res['efficiency'],                         # particle_kw / incident
            wall_max_c = res['wall_max_c'],                                  # Receiver cavity walls
            snout_wall_max_c = res['SNOUT wall T max (C)'],
            snout_kw = np.array([res[f'SNOUT {p} absorbed (kW)'] for p in ('top', 'bottom', 'east', 'west')]),
            snout_peak_kw_m2 = max(res[f'SNOUT {p} peak flux (kW/m^2)'] for p in ('top', 'bottom', 'east', 'west')),
            heliostat_intercept = res['heliostat_intercept'],                     # (n,) each mirror's own rays absorbed on the curtain
            heliostat_column_rays = res['heliostat_column_rays'],                 # (n, 5) its rays absorbed per curtain column
            heliostat_snout_rays = res['heliostat_snout_rays'],                   # (n, 4) its rays landing on each SNOUT panel
            snout_panel_max_c = np.array([res[f"SNOUT {p['name']} wall T max (C)"] for p in self.field.snout_panels]),   # (4,) same panel order
            motor_travel = self.kin.travel / (2 * self.n * self.kin.rate * self.dt * 60),   # 0 = nobody moved, 1 = every motor at full speed all step
        )

    #--- Observation: clock and sun, where everyone is aiming, plant state, column temperatures
    def _obs(self):
        az, el = sun_at(self.day, self.hour)
        m = self.last
        return np.concatenate([
            [self.hour, az, el, self.dni],            # Clock, sun, DNI
            self.offsets.ravel(),                     # Aim offsets of all heliostats
            [m['power_kw'], m['peak_kw_m2'], m['intercept'], m['particle_out_c'], m['wall_max_c'], m['snout_wall_max_c']],
            m['particle_out_columns_c'],              # Outlet temperature per curtain column: what the unevenness term looks at
        ]).astype(np.float32)

    #--- Terminal check: wall over the limit, sun down, or time up
    def _terminal(self, m):
        if max(m['wall_max_c'], m['snout_wall_max_c']) > self.wall_limit_c:
            return True, 'wall_over_temp'
        if m['sun_el'] < self.sun_min_el:
            return True, 'sun_down'
        if self.minute >= self.episode_minutes:
            return True, 'time_up'
        return False, ''


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#--- Example: 30 steps with every heliostat on the center point
if __name__ == "__main__":
    env = AimingEnv()
    env.reset(seed = 0)
    for k in range(30):
        obs, reward, done, m = env.step(np.full(env.n, env.center_point))
        print(f"min {k+1:2d}  DNI {m['dni_w_m2']:5.0f}  power {m['power_kw']:7.1f} kW  flow {m['mass_flow']:5.2f} kg/s  Tout {m['particle_out_c']:6.1f} C  "
              f"wall {m['wall_max_c']:6.1f} C  snout {m['snout_wall_max_c']:6.1f} C  reward {reward:6.3f}")
        if done:
            print('episode ended:', m['why'])
            break