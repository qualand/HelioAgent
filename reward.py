import numpy as np


#--- Reward of one step from the metrics the environment produced: r = w_optical * eta_opt + w_thermal * eta_th - w_travel * m - w_margin * p
#    eta_opt = power on the curtain / (DNI x mirror area): the whole optical chain of the ray trace (cosine, shading, blocking, reflectivity,
#              attenuation, spillage). Heliostats in standby count as a loss here, so the agent uses them when it safely can.
#    eta_th  = particle power / power on the curtain, from Janna's receiver model: radiation and convection losses of the cavity
#    m       = motor travel, 0 = nobody moved, 1 = every motor at full speed all step; sun tracking alone is about 0.02
#    p       = SNOUT margin, 0 while the hottest SNOUT cell is below margin_start_c, 1 at the wall limit: a warning every step before the trip.
#              night_09 (w_margin = 0): the one-time -2 came too late, the agent kept every mirror on center and tripped in most episodes
#    The shutdown penalty (-2 when a wall passes the limit) is added by gym_wrapper, which owns the episode end
def score(metrics, w_optical = 1.0, w_thermal = 1.0, w_travel = 0.05, w_margin = 0.5, margin_start_c = 1100.0, wall_limit_c = 1200.0):
    optical_efficiency = metrics['optical_efficiency']
    thermal_efficiency = metrics['thermal_efficiency']
    motor_travel = metrics['motor_travel']
    snout_margin = snout_margin_term(metrics, margin_start_c, wall_limit_c)
    reward = w_optical * optical_efficiency + w_thermal * thermal_efficiency - w_travel * motor_travel - w_margin * snout_margin
    terms = dict(optical_efficiency = optical_efficiency, thermal_efficiency = thermal_efficiency, motor_travel = motor_travel, snout_margin = snout_margin)
    return reward, terms


#--- SNOUT margin p: how far the hottest SNOUT cell is into the band between margin_start_c and the limit (0 below the band, 1 at the limit)
def snout_margin_term(metrics, margin_start_c, wall_limit_c):
    return float(np.clip((metrics['snout_wall_max_c'] - margin_start_c) / (wall_limit_c - margin_start_c), 0.0, 1.0))


#--- Credit of each heliostat: how much of the reward the field would lose if this one mirror went to standby (a difference reward,
#    Wolpert and Tumer 2001), term by term, to first order:
#      eta_opt    loses the mirror's share n_i / N of the curtain rays (exact: eta_opt is a sum over mirrors)
#      eta_th     the receiver losses are nearly fixed, so taking power away lowers eta_th by (1 - eta_th) n_i / N
#      margin     the hottest SNOUT edge cools by dT/dq x (the mirror's flux there), which lowers p by that over the 100 C band
#      travel     the mirror's own motor travel over the n mirrors (m is their mean): a hop between menu points is charged to the mirror that hopped
#      penalty    split by the mirrors' flux on the hottest panel's edge
#    night_10 split the margin term by shares instead (0.5 p / 218, about 0.001 per mirror): ten times too small, no mirror ever found standby
#    worth its 0.002 of efficiency, so the agent never used it. The learner adds the credit, centered and scaled, to the shared advantage (train_ppo.py).
def local_credit(metrics, penalty = 0.0, w_optical = 1.0, w_thermal = 1.0, w_travel = 5.0, w_margin = 0.5, margin_start_c = 1100.0, wall_limit_c = 1200.0, **unused):
    column_rays = np.asarray(metrics['heliostat_column_rays'], dtype = float)         # (n, 5) rays of each mirror absorbed in each curtain column
    total = max(column_rays.sum(), 1.0)                                                # All curtain rays of the field
    curtain_share = column_rays.sum(axis = 1) / total                                  # n_i / N
    credit = (w_optical * metrics['optical_efficiency'] + w_thermal * (1.0 - metrics['thermal_efficiency'])) * curtain_share
    heliostat_travel = np.asarray(metrics['heliostat_travel'])                             # (n,) each mirror's travel, 0 still to 1 full speed all step
    credit = credit - w_travel * heliostat_travel / len(heliostat_travel)                  # What the field's travel term loses to this mirror's motion
    hot_band = np.asarray(metrics['heliostat_hot_band_kw_m2'])                          # (n, 4) each mirror's incident flux in the band next to the aperture, per panel (kW/m2)
    panel_max_c = np.asarray(metrics['snout_panel_max_c'])
    on_hottest = hot_band[:, panel_max_c.argmax()]
    cooling_c = 1e3 * metrics['snout_temperature_per_flux'] * on_hottest               # C the hottest SNOUT edge cools without this mirror
    if snout_margin_term(metrics, margin_start_c, wall_limit_c) > 0:                   # Only while the SNOUT is inside the margin band
        credit = credit - w_margin * cooling_c / (wall_limit_c - margin_start_c)
    if penalty:                                                                         # The step the episode ends over temperature
        share = on_hottest / max(on_hottest.sum(), 1e-9) if panel_max_c.max() >= metrics['wall_max_c'] else curtain_share   # SNOUT tripped: those mirrors pay; a cavity wall: by light into the cavity
        credit = credit - penalty * share
    return credit


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#--- Previous reward (night_04 to night_08): r = eta_opt + 0.5 * s - 1.0 * u - 0.05 * m, with s = on setpoint and u = column temperature unevenness.
#    Kept for reference. To go back: uncomment, and put w_uneven and w_setpoint back in AimingEnv.weights.
#
# from receiver_thermal import G3P3_RECEIVER
# particle_in_c = G3P3_RECEIVER['Tpin'] - 273.15                  # Particle inlet temperature (C), 555
# particle_target_c = G3P3_RECEIVER['Tpout_target'] - 273.15      # Particle outlet setpoint (C), 775
# temperature_rise_c = particle_target_c - particle_in_c          # Design rise (C), 220: makes the setpoint and unevenness terms dimensionless and of order one
#
# def score(metrics, w_uneven = 1.0, w_travel = 0.05, w_setpoint = 0.5):
#     optical_efficiency = metrics['optical_efficiency']                                                    # Curtain power / (DNI x mirror area)
#     on_setpoint = 1.0 - max(particle_target_c - metrics['particle_out_c'], 0.0) / temperature_rise_c     # 1 at 775 C out, 0 a full rise below; above 775 does not count
#     on_setpoint = min(max(on_setpoint, 0.0), 1.0)
#     columns_c = np.asarray(metrics['particle_out_columns_c'])                                             # Outlet temperature of each curtain column (C)
#     temperature_unevenness = min((columns_c.max() - columns_c.min()) / temperature_rise_c, 1.0)           # Hottest minus coldest column over the 220 C rise, capped at 1
#     motor_travel = metrics['motor_travel']
#     reward = optical_efficiency + w_setpoint * on_setpoint - w_uneven * temperature_unevenness - w_travel * motor_travel
#     terms = dict(optical_efficiency = optical_efficiency, on_setpoint = on_setpoint,
#                  temperature_unevenness = temperature_unevenness, motor_travel = motor_travel)
#     return reward, terms
#
# def local_credit(metrics, penalty = 0.0, w_uneven = 1.0, **unused):
#     column_rays = np.asarray(metrics['heliostat_column_rays'], dtype = float)         # (n, 5) rays of each mirror absorbed in each curtain column
#     total = max(column_rays.sum(), 1.0)
#     curtain_share = column_rays.sum(axis = 1) / total                                  # n_i / N
#     efficiency_share = metrics['optical_efficiency'] * curtain_share                   # eta_i = eta * n_i / N
#     columns_c = np.asarray(metrics['particle_out_columns_c'])
#     hotter = (columns_c - columns_c.mean()) / temperature_rise_c                        # h_c: how much hotter column c is than the mean, over the 220 C rise
#     feeds_hot = column_rays @ hotter / total                                            # Sum_c n_ic h_c / N: positive for a mirror feeding hot columns
#     unevenness_share = metrics['temperature_unevenness'] * feeds_hot / max(feeds_hot.sum(), 0.01)   # u_i = u * (mirror's hot feed) / (field's hot feed)
#     credit = efficiency_share - w_uneven * unevenness_share
#     if penalty:
#         snout_rays = np.asarray(metrics['heliostat_snout_rays'], dtype = float)
#         panel_max_c = np.asarray(metrics['snout_panel_max_c'])
#         if panel_max_c.max() >= metrics['wall_max_c']:
#             on_panel = snout_rays[:, panel_max_c.argmax()]
#             share = on_panel / max(on_panel.sum(), 1.0)
#         else:
#             share = curtain_share
#         credit = credit - penalty * share
#     return credit