import numpy as np
from external.falling_particle_receiver.falling_particle_receiver import Falling_Particle_Receiver

#--- G3P3 settings for Janna's receiver model
G3P3_RECEIVER = dict(
    ap_height = 1.32,                       # Aperture height (m)
    ap_aspect_ratio = 1.0,                  # Aperture width / height
    cavity_to_ap_height_ratio = 1.1,        # Must match cavity['height_ratio'] in build_field
    cavity_to_ap_width_ratio = 1.1,         # Must match cavity['width_ratio']
    curtain_depth_from_aperture = 0.5,      # Must match cavity['curtain_depth'] (m)
    is_aperture_at_bottom = True,
    rec_azimuth = 19.414,                   # Degrees
    tower_height = 45.48,                   # Aperture center height (m)
    Tpin = 555.0 + 273.15,                  # Particle inlet temperature (K)
    Tpout_target = 775.0 + 273.15,          # Particle outlet setpoint (K)
    is_adjust_mflow = True,                 # Find the particle flow that gives Tpout_target, as the plant controller does
    is_flux_absolute = True, flux_method = 2,   # Flux map given as a 2D array of absolute values
    model_type = 3,
    radiation_formulation = 1,
    ny = 21, nx = 5, nyrad = 20, nxrad = 5, # Curtain grid: coarser than Janna's defaults (41, 10, 40, 10) for speed; nx = 5 columns feed the unevenness term
)


#--- Wrapper around Janna's falling-particle receiver: one call per step, incident flux on the curtain in, particle outlet state and losses out.
#    The model is steady state, so each step is solved on its own. curtain_flux() bins the SolTrace hits into a 25 x 25 map (kW/m2, row 0 = top);
#    step() hands it over as flux_profile_input; the model interpolates it onto its own 21 x 5 grid.
class ReceiverThermal:
    def __init__(self, params = None, min_power_kw = 5.0):

        #--- The receiver
        self.receiver = Falling_Particle_Receiver({**G3P3_RECEIVER, **(params or {})})

        #--- Options
        self.min_power_kw = min_power_kw        # Below this incident power the model is not run: particles leave at the inlet temperature
        return

    #--- One step: curtain_flux in kW/m2, shape (n_height, n_width), row 0 = top of the curtain
    def step(self, curtain_flux, ambient_c = 20.0, wind_speed = 0.0, wind_direction = 360.0):
        receiver = self.receiver
        flux = np.asarray(curtain_flux, dtype = float)
        incident_kw = float(flux.mean() * receiver.get_curtain_height() * receiver.get_curtain_width())
        if incident_kw < self.min_power_kw:
            return dict(converged = False, incident_kw = incident_kw, particle_kw = 0.0, thermal_kw = 0.0, efficiency = 0.0,
                        mass_flow = 0.0, particle_out_c = receiver.Tpin - 273.15, particle_max_c = receiver.Tpin - 273.15, wall_max_c = ambient_c,
                        aperture_ir_kw = 0.0, aperture_eff_c = ambient_c,
                        particle_out_columns_c = np.full(receiver.nx, receiver.Tpin - 273.15))
        receiver.flux_profile_input = flux
        receiver.Tamb = ambient_c + 273.15
        receiver.Tsky = receiver.Tamb - 20.0
        receiver.wspd = wind_speed
        receiver.wdir = wind_direction
        converged = receiver.simulate()
        results = receiver.results
        ap_area = receiver.ap_height ** 2 * receiver.ap_aspect_ratio
        ir_out_w_m2 = max(results['QIR'], 0.0) * 1e6 / ap_area + 5.6704e-8 * receiver.Tsky ** 4   # Infrared leaving the cavity through the aperture; QIR is net of the sky's share
        aperture_eff_c = (ir_out_w_m2 / 5.6704e-8) ** 0.25 - 273.15                                 # Black body that radiates the same power: what the SNOUT sees
        return dict(converged = bool(converged),
                    incident_kw = float(results['Qinc'] * 1e3),
                    particle_kw = float(results['Qp'] * 1e3),
                    thermal_kw = float(results['Qthermal'] * 1e3),
                    efficiency = float(results['efficiency']),
                    mass_flow = float(results['mflow']),
                    particle_out_c = float(results['Tpout'] - 273.15),
                    particle_max_c = float(results['Tpmax'] - 273.15),
                    wall_max_c = float(results['Twmax'] - 273.15),
                    aperture_ir_kw = float(results['QIR'] * 1e3),                 # Net infrared out through the aperture (QIR is in MW)
                    aperture_eff_c = float(aperture_eff_c),
                    particle_out_columns_c = results['Tp'][-1, :] - 273.15)       # Bottom row of the curtain grid: outlet temperature per column