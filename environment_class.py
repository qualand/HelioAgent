import numpy as np
from functions_4_environment import build_field, receiver_axes, aim_grid, center_point, offsets_to_xyz, set_aim_xyz, trace, warm_up_snout, sun_at, hours_above, spread_by_distance, dni_at
from helio_kinematics import HelioKinematics
from reward import score


class AimingEnv:
    def __init__(self, dt_min = 1.0, slew_mrad_s = 1.3, nray = 1e6, episode_minutes = 120, sun_min_el = 7.0, operating_min_el = 15.0,
                 wall_limit_c = 1200.0, w_optical = 1.0, w_thermal = 1.0, w_travel = 5.0, w_margin = 0.5, margin_start_c = 1100.0, trace_seed = 123):

        #--- Plant, built once (the slow part, never repeated)
        self.field, self.PT = build_field()         # Heliostat field, receiver, SNOUT and the SolTrace scene
        self.axes = receiver_axes(self.field)       # Aperture frame: center, sideways, up
        self.n = len(self.field.heliostats)         # 218 heliostats
        self.kin = HelioKinematics(self.field, slew_mrad_s)   # Motors: where every mirror points now, slew-rate limited

        #--- Actions of one heliostat: a menu point on the aperture, or standby
        self.grid = aim_grid(self.field)            # 3 x 5 menu, 15 aim points on the aperture; action k = menu point k
        self.center_point = center_point(self.grid) # Point 7: (0, 0), the aperture center
        self.tower_top_m = 61.0                     # Top of the NSTTF tower above the ground (m); the aperture center is at 45.48 m. To confirm with Bill
        self.standby_clearance_m = 5.0              # The standby point sits this far above the tower top: the beam passes over the tower into the sky and heats nothing
        self.standby_offset = np.array([0.0, self.tower_top_m + self.standby_clearance_m - self.field.tower_height])   # (dx, dv) from the aperture center in the aperture plane: (0, 20.5 m)
        self.standby_point = len(self.grid)         # Action 15 = standby: the mirror tracks the standby point, its light is off the receiver. In the scene the rays hit nothing
                                                    # (the tower face strips stop 10 m above the aperture). The move takes 30 to 100 s at 1.3 mrad/s, within one 5 min step
        self.n_actions = len(self.grid) + 1

        #--- Step and episode
        self.dt = dt_min                            # Minutes of plant time per step
        self.nray = nray                            # Rays per trace (1e6: SNOUT maximum within ~20 C of converged on the 5 cm mesh)
        self.trace_seed = trace_seed                # SolTrace random seed; None = a new seed every trace, so ray noise differs step to step
        self.episode_minutes = episode_minutes      # Episode length (min); 120 = 2 h of regular operation
        self.sun_min_el = sun_min_el                # Degrees; the episode ends when the sun drops below it
        self.operating_min_el = operating_min_el    # Degrees; regular operation = sun at least this high, no startup or shutdown

        #--- Safety and reward
        self.wall_limit_c = wall_limit_c            # RSLE-57 limit (C) on the receiver walls and the SNOUT: the episode stops above it
        self.snout_uncertainty = True               # True: the SNOUT temperature used for the limit, the margin p and the observation is T + dT, the cell temperature plus its
                                                    # propagated ray-count uncertainty (about 19 C at 5 cm and 1e6 rays). False: the measured T alone. The measured values stay in the metrics
        self.weights = dict(w_optical = w_optical, w_thermal = w_thermal, w_travel = w_travel,                    # Reward weights, see reward.score
                            w_margin = w_margin, margin_start_c = margin_start_c, wall_limit_c = wall_limit_c)   # SNOUT margin: costs w_margin per step at the limit, 0 below margin_start_c
        # w_travel = 5: at 0.05 (night_04 to night_11) a hop between neighbouring menu points cost nothing visible and the sampled policy moved
        # 180 mirrors every step. At 5, sun tracking costs a constant 0.085 per step, 180 hops about 0.03, and a 26-mirror standby swap about 0.05 once

        #--- State: what the environment remembers between steps, filled by reset(), changed by step()
        self.day = None        # Day of the year, fixed for the episode
        self.hour = None       # Solar time; step() adds dt / 60
        self.minute = None     # Minutes since reset
        self.dni = None        # W/m2 of the current step
        self.choice = None     # One integer per heliostat: menu point 0..14 or 15 = standby
        self.standby = None    # (n,) True where the heliostat is in standby
        self.offsets = None    # (n, 2) the (dx, dv) each heliostat tracks: its menu point, or the standby offset
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
            choice = np.full(self.n, self.center_point)
        elif start == 'spread':
            choice = spread_by_distance(self.kin.pos, self.grid)                     # Image-size rule
        else:
            choice = rng.integers(0, self.n_actions, self.n)                         # Random actions, standby included
        target = self._targets(choice)

        az, el = sun_at(self.day, self.hour)
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
    #    choice: array (n,) of ints 0..15 per heliostat (15 = standby). Returns (obs, reward, done, metrics).
    def step(self, choice):
        target = self._targets(choice)

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

    #--- Actions -> the point each mirror tracks (global xyz). Also stores choice, standby and offsets
    def _targets(self, choice):
        self.choice = np.asarray(choice, dtype = int)
        self.standby = self.choice == self.standby_point
        on_menu = np.minimum(self.choice, len(self.grid) - 1)                       # Standby mirrors borrow a menu index here, replaced on the next line
        self.offsets = np.where(self.standby[:, None], self.standby_offset, self.grid[on_menu])
        return offsets_to_xyz(self.offsets, self.axes)

    #--- Where the mirrors point now -> beam landing points -> DNI of the step -> ray trace and thermal models
    def _trace_now(self, az, el, target):
        dist = np.linalg.norm(target - self.kin.pos, axis = 1)      # Mirror to wanted point (m)
        set_aim_xyz(self.field, self.kin.aim_points(az, el, dist))
        self.dni = max(dni_at(self.day, self.hour, el), 1.0)        # Clear sky or weather file, chosen in functions_4_environment.py; 1 W/m2 floor keeps the efficiency ratio finite under clouds
        seed = self.trace_seed if self.trace_seed is not None else int(np.random.default_rng().integers(1, 2**31 - 1))
        return trace(self.PT, self.field, az, el, nray = self.nray, dni = self.dni, dt_s = self.dt * 60, seed = seed)

    #--- Metrics of one trace, the dictionary the reward, the credit and the loggers read
    #    The optical chain comes from Bill's process_soltrace_results (copylot_to_soltrace.py): the ray trace already includes cosine, shading,
    #    blocking, mirror reflectivity, atmospheric attenuation (DELSOL3 clear day, as a reflectivity per distance group) and spillage.
    def _metrics(self, res, flux, sun_el):
        return dict(
            power_kw = res['Absorbed power (kW)'],
            peak_kw_m2 = float(flux.max()),
            optical_efficiency = res['Field efficiency (%)'] / 100.0,                   # Curtain power / (DNI x mirror area): the product of the four lines below
            cosine_shading_efficiency = res['Shadowing and cosine efficiency (%)'] / 100.0,
            blocking_efficiency = res['Blocking efficiency (%)'] / 100.0,
            reflection_attenuation_efficiency = res['Reflection and attenuation efficiency (%)'] / 100.0,
            intercept = res['Intercept efficiency (%)'] / 100.0,                        # Rays on the curtain / rays leaving the field; standby mirrors lower it
            sun_el = sun_el,
            dni_w_m2 = self.dni,
            particle_out_c = res['particle_out_c'],
            particle_max_c = res['particle_max_c'],
            particle_out_columns_c = res['particle_out_columns_c'],         # Outlet temperature per curtain column (Janna's 5 columns)
            aperture_c = res['aperture_eff_c'],                              # What the SNOUT sees through the aperture
            mass_flow = res['mass_flow'],                                    # kg/s the controller chose to hold 775 C
            particle_kw = res['particle_kw'],                                # Power that went into the particles
            thermal_efficiency = res['efficiency'],                          # particle_kw / curtain power, from Janna's model
            wall_max_c = res['wall_max_c'],                                  # Receiver cavity walls
            snout_wall_max_c = res['SNOUT wall T max upper (C)' if self.snout_uncertainty else 'SNOUT wall T max (C)'],   # What the limit, the margin and the agent use
            snout_wall_max_measured_c = res['SNOUT wall T max (C)'],         # The model value without the uncertainty
            snout_wall_max_uncertainty_c = res['SNOUT T max uncertainty (C)'],   # dT = (dT/dq) q / sqrt(n) of the hottest cell
            snout_kw = np.array([res[f'SNOUT {p} absorbed (kW)'] for p in ('top', 'bottom', 'east', 'west')]),
            snout_peak_kw_m2 = max(res[f'SNOUT {p} peak flux (kW/m^2)'] for p in ('top', 'bottom', 'east', 'west')),
            heliostat_intercept = res['heliostat_intercept'],                     # (n,) each mirror's own rays absorbed on the curtain
            heliostat_column_rays = res['heliostat_column_rays'],                 # (n, 5) its rays absorbed per curtain column
            heliostat_snout_rays = res['heliostat_snout_rays'],                   # (n, 4) its rays landing on each SNOUT panel
            heliostat_hot_band_kw_m2 = res['heliostat_hot_band_kw_m2'],           # (n, 4) its incident flux in the 10 cm band next to the aperture of each panel, where the maximum sits
            snout_panel_max_c = np.array([res[f"SNOUT {p['name']} wall T max {'upper ' if self.snout_uncertainty else ''}(C)"] for p in self.field.snout_panels]),   # (4,) same panel order, same choice
            snout_temperature_per_flux = res['SNOUT dT/dq (K per W/m2)'],         # How much the hottest SNOUT cell cools per unit of flux taken away (about 0.35 mK per W/m2)
            n_standby = int(self.standby.sum()),                                   # Heliostats in standby this step
            standby_fraction = float(self.standby.mean()),
            motor_travel = self.kin.travel / (2 * self.n * self.kin.rate * self.dt * 60),   # 0 = nobody moved, 1 = every motor at full speed all step
            heliostat_travel = self.kin.travel_per_mirror / (2 * self.kin.rate * self.dt * 60),   # (n,) the same per mirror; its mean is motor_travel
        )

    #--- Observation: clock and sun, then 3 numbers per heliostat, then the plant state
    #    Per heliostat: (dx, dv) of its menu point and a standby flag; a standby mirror shows (0, 0, 1). The policy reads this block by position (policy_network.split).
    #    Plant: power, peak flux, intercept, thermal efficiency, particle outlet, receiver wall maximum, the four SNOUT panel maxima, the five column temperatures.
    def _obs(self):
        az, el = sun_at(self.day, self.hour)
        m = self.last
        per_mirror = np.c_[np.where(self.standby[:, None], 0.0, self.offsets), self.standby.astype(float)]   # (n, 3)
        return np.concatenate([
            [self.hour, az, el, self.dni],            # Clock, sun, DNI
            per_mirror.ravel(),                       # dx, dv, standby for every heliostat
            [m['power_kw'], m['peak_kw_m2'], m['intercept'], m['thermal_efficiency'], m['particle_out_c'], m['wall_max_c']],
            m['snout_panel_max_c'],                   # Top, bottom, east, west: which SNOUT wall is getting hot
            m['particle_out_columns_c'],              # Outlet temperature per curtain column
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
#--- Example: 10 steps with every heliostat on the center point, then 10 more with the 60 heliostats nearest the tower in standby
if __name__ == "__main__":
    env = AimingEnv()
    env.reset(seed = 0)
    nearest_60 = np.argsort(np.linalg.norm(env.kin.pos[:, :2], axis = 1))[:60]
    for k in range(20):
        choice = np.full(env.n, env.center_point)
        if k >= 10:
            choice[nearest_60] = env.standby_point
        obs, reward, done, m = env.step(choice)
        print(f"min {k+1:2d}  standby {m['n_standby']:3d}  DNI {m['dni_w_m2']:5.0f}  power {m['power_kw']:7.1f} kW  eta_opt {m['optical_efficiency']:.3f}  "
              f"eta_th {m['thermal_efficiency']:.3f}  Tout {m['particle_out_c']:6.1f} C  snout {m['snout_wall_max_c']:6.1f} C  reward {reward:6.3f}")
        if done:
            print('episode ended:', m['why'])
            break