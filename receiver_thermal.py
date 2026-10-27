# Wrapper around Janna falling-particle receiver model
# One call per time step: incident flux on the curtain in, particle outlet state and losses out.
# The model is steady state, so each step is solved independently.

# SolTrace traces the rays → curtain_flux() bins the hits on the curtain plane into a 25 × 25 map of incident flux in kW/m² (row 0 = top of the curtain) 
# → step() hands that map to Janna’s model as flux_profile_input 
# → her model interpolates it onto its own 21 × 5 grid and uses it as the solar flux arriving at each spot of the curtain 
# → it returns what the particles got out of it.

import numpy as np
from external.falling_particle_receiver.falling_particle_receiver import Falling_Particle_Receiver

# G3P3 settings. 
G3P3_RECEIVER = dict(
    ap_height=1.32,
    ap_aspect_ratio=1.0,
    cavity_to_ap_height_ratio=1.1,      # must match cavity['height_ratio'] in build_field
    cavity_to_ap_width_ratio=1.1,       # must match cavity['width_ratio']
    curtain_depth_from_aperture=0.5,    # must match cavity['curtain_depth']
    is_aperture_at_bottom=True,
    rec_azimuth=19.414,
    tower_height=45.48,
    Tpin=555.0 + 273.15,
    Tpout_target=775.0 + 273.15,
    is_adjust_mflow=True,               # find the particle flow that gives Tpout_target, as the plant controller does

    is_flux_absolute=True,    flux_method=2,                      # flux map given as a 2D array
    model_type=3,
    radiation_formulation=1,
    ny=21, nx=5, nyrad=20, nxrad=5,     # coarser than Janna's defaults (41, 10, 40, 10) for speed
)

class ReceiverThermal:
    def __init__(self, params=None, min_power_kw=5.0):
        self.receiver = Falling_Particle_Receiver({**G3P3_RECEIVER, **(params or {})})
        self.min_power_kw = min_power_kw

    # curtain_flux: kW/m2, shape (n_height, n_width), row 0 = top of the curtain
    def step(self, curtain_flux, ambient_c=20.0, wind_speed=0.0, wind_direction=360.0):
        receiver = self.receiver
        flux = np.asarray(curtain_flux, dtype=float)
        incident_kw = float(flux.mean() * receiver.get_curtain_height() * receiver.get_curtain_width())
        if incident_kw < self.min_power_kw:
            return dict(converged=False, incident_kw=incident_kw, particle_kw=0.0, thermal_kw=0.0, efficiency=0.0,
                        mass_flow=0.0, particle_out_c=receiver.Tpin - 273.15, particle_max_c=receiver.Tpin - 273.15, wall_max_c=ambient_c,
                        aperture_ir_kw=0.0)
        receiver.flux_profile_input = flux
        receiver.Tamb = ambient_c + 273.15
        receiver.Tsky = receiver.Tamb - 20.0
        receiver.wspd = wind_speed
        receiver.wdir = wind_direction
        converged = receiver.simulate()
        results = receiver.results
        return dict(converged=bool(converged),
                    incident_kw=float(results['Qinc'] * 1e3),
                    particle_kw=float(results['Qp'] * 1e3),
                    thermal_kw=float(results['Qthermal'] * 1e3),
                    efficiency=float(results['efficiency']),
                    mass_flow=float(results['mflow']),
                    particle_out_c=float(results['Tpout'] - 273.15),
                    particle_max_c=float(results['Tpmax'] - 273.15),
                    wall_max_c=float(results['Twmax'] - 273.15),
                    aperture_ir_kw=float(results['QIR'] * 1e3))     # net infrared out through the aperture (QIR is in MW)
