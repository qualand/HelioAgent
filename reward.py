# One number per minute from the metrics the environment already produces. RL route: the reward.

import numpy as np
from receiver_thermal import G3P3_RECEIVER

particle_in_c = G3P3_RECEIVER['Tpin'] - 273.15
particle_target_c = G3P3_RECEIVER['Tpout_target'] - 273.15
temperature_rise_c = particle_target_c - particle_in_c

def score(metrics, w_uneven=1.0, w_travel=0.05, w_setpoint=0.5):
    optical_efficiency = metrics['optical_efficiency']
    on_setpoint = 1.0 - max(particle_target_c - metrics['particle_out_c'], 0.0) / temperature_rise_c   # only being below 775 C costs
    on_setpoint = min(max(on_setpoint, 0.0), 1.0)
    columns_c = np.asarray(metrics['particle_out_columns_c'])                                           # outlet temperature of each curtain column
    temperature_unevenness = min((columns_c.max() - columns_c.min()) / temperature_rise_c, 1.0)         # hottest column minus coldest, over the 220 C rise
    motor_travel = metrics['motor_travel']
    reward = optical_efficiency + w_setpoint * on_setpoint - w_uneven * temperature_unevenness - w_travel * motor_travel
    terms = dict(optical_efficiency=optical_efficiency, on_setpoint=on_setpoint,
                 temperature_unevenness=temperature_unevenness, motor_travel=motor_travel)
    return reward, terms