import numpy as np
from receiver_thermal import G3P3_RECEIVER

#--- Particle temperatures that scale the reward terms
particle_in_c = G3P3_RECEIVER['Tpin'] - 273.15                  # Particle inlet temperature (C), 555
particle_target_c = G3P3_RECEIVER['Tpout_target'] - 273.15      # Particle outlet setpoint (C), 775
temperature_rise_c = particle_target_c - particle_in_c          # Design rise (C), 220: makes the setpoint and unevenness terms dimensionless and of order one


#--- Reward of one step from the metrics the environment produced: r = efficiency + w_setpoint * s - w_uneven * u - w_travel * m
#    The shutdown penalty (-2 when a wall passes the limit) is added by gym_wrapper, which owns the episode end
def score(metrics, w_uneven = 1.0, w_travel = 0.05, w_setpoint = 0.5):
    optical_efficiency = metrics['optical_efficiency']                                                    # Curtain power / (DNI x mirror area)
    on_setpoint = 1.0 - max(particle_target_c - metrics['particle_out_c'], 0.0) / temperature_rise_c     # 1 at 775 C out, 0 a full rise below; above 775 does not count
    on_setpoint = min(max(on_setpoint, 0.0), 1.0)
    columns_c = np.asarray(metrics['particle_out_columns_c'])                                             # Outlet temperature of each curtain column (C)
    temperature_unevenness = min((columns_c.max() - columns_c.min()) / temperature_rise_c, 1.0)           # Hottest minus coldest column over the 220 C rise, capped at 1
    motor_travel = metrics['motor_travel']                                                                # Fraction of full motor speed used in the step; sun tracking alone is about 0.02
    reward = optical_efficiency + w_setpoint * on_setpoint - w_uneven * temperature_unevenness - w_travel * motor_travel
    terms = dict(optical_efficiency = optical_efficiency, on_setpoint = on_setpoint,
                 temperature_unevenness = temperature_unevenness, motor_travel = motor_travel)
    return reward, terms


#--- Credit of each heliostat: its share of the reward terms its own rays change, c_i = eta_i - w_uneven * u_i (- its share of the penalty)
#    The shares add up to the field terms. The learner adds them, centered and scaled, to the shared advantage (train_ppo.py).
#    Setpoint and motor travel stay shared. Scale matters: one mirror moves the field efficiency by 1/218 of its intercept change but the
#    column temperatures by about 5 x its power share, so the unevenness piece is the larger lever per mirror (night_05 had it 20 x too small).
def local_credit(metrics, penalty = 0.0, w_uneven = 1.0, **unused):
    column_rays = np.asarray(metrics['heliostat_column_rays'], dtype = float)         # (n, 5) rays of each mirror absorbed in each curtain column
    total = max(column_rays.sum(), 1.0)                                                # All curtain rays of the field
    curtain_share = column_rays.sum(axis = 1) / total                                  # n_i / N
    efficiency_share = metrics['optical_efficiency'] * curtain_share                   # eta_i = eta * n_i / N
    columns_c = np.asarray(metrics['particle_out_columns_c'])
    hotter = (columns_c - columns_c.mean()) / temperature_rise_c                        # h_c: how much hotter column c is than the mean, over the 220 C rise
    feeds_hot = column_rays @ hotter / total                                            # Sum_c n_ic h_c / N: positive for a mirror feeding hot columns
    unevenness_share = metrics['temperature_unevenness'] * feeds_hot / max(feeds_hot.sum(), 0.01)   # u_i = u * (mirror's hot feed) / (field's hot feed)
    credit = efficiency_share - w_uneven * unevenness_share
    if penalty:                                                                         # The step the episode ends over temperature
        snout_rays = np.asarray(metrics['heliostat_snout_rays'], dtype = float)         # (n, 4) rays of each mirror on each SNOUT panel
        panel_max_c = np.asarray(metrics['snout_panel_max_c'])
        if panel_max_c.max() >= metrics['wall_max_c']:
            on_panel = snout_rays[:, panel_max_c.argmax()]                               # The SNOUT tripped: the mirrors with light on the hottest panel pay
            share = on_panel / max(on_panel.sum(), 1.0)
        else:
            share = curtain_share                                                       # A receiver cavity wall tripped: by the light sent into the cavity
        credit = credit - penalty * share
    return credit