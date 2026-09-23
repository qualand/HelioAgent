# This is the physics 

#   build_field    -> "Build field, once"
#   receiver_axes  ->  helper for the Action box (which way is "up" on the plate)
#   set_aims       -> "Action space"
#   trace          -> "Ray trace"

import numpy as np
from copylot_to_soltrace import Heliostat_Field, simulate_soltrace, process_soltrace_results
from api.pysoltrace import Point

# Build the plant once. Same defaults as main.py
# SolarPILOT decides where the heliostats stand
# SolTrace gets a 3D scene built from that layout
# Returns the field object (list of heliostats) and PT (the SolTrace scene)\
    
def build_field():
    
    field = Heliostat_Field()
    field.rec_design_power = 12.0     # MWt
    field.tower_height = 50.0         # m
    field.rec_type = 2                # flat plate
    field.rec_height = 4.0            # m
    field.rec_width = 4.0             # m
    field.rec_elevation = -30.0       # deg, plate tilts down toward the field
    field.helio_height = 3.0          # m
    field.helio_width = 4.55          # m
    field.helio_n_cant_x = 2          # two mirror facets side by side
    field.helio_n_cant_y = 1
    field.helio_surf_err = 0.002      # rad
    field.helio_cant_method = 0
    field.helio_focus_method = 1
    field.n_focus_bands = 3
    field.aim_method = 3              # SolarPILOT's own aiming; we overwrite it later
    field.des_sim_ndays = 4           # fewer design days = faster layout
    field.des_sim_nhours = 2
    field.generate_field_via_copylot(display_results=False)
    PT = field.set_up_soltrace()
    
    return field, PT

# The plate is tilted, so "1 m on the plate" is not "1 m up in z"
# This returns 3 vectors in global (x east, y north, z up) coordinates
# An aim offset (dx, dv) in meters becomes center + dx*u_h + dv*u_v
# Both vectors are perpendicular to the plate normal, so the aim point always stays in the plane of the plate. 

def receiver_axes(field):
    p = field.results['sp_parameters']
    az = np.radians(p['receiver.0.rec_azimuth'])
    el = np.radians(p['receiver.0.rec_elevation'])
    center = np.array([0.0, 0.0, p['solarfield.0.tht']])
    u_h = np.array([np.cos(az), - np.sin(az), 0.0])
    u_v = np.array([-np.sin(el) * np.sin(az), -np.sin(el) * np.cos(az), np.cos(el)])
    return center, u_h, u_v

# The Action box. Write a new aim_point into every heliostat.

def set_aims(field, offsets, axes):
    center, u_h, u_v = axes
    for h, (dx, dv) in zip(field.heliostats, offsets):
        h.aim_point = Point(*(center + dx * u_h + dv * u_v))

# The Ray trace box. One sun position in, flux out. No decisions here. 
# 1. Point every mirror at its aim_point for this sun
# 2. Run SolTrace, single thread
# 3. Count rays into power and efficiencies
# 4. Bin the hits on the 4 m plate into a 25x25 flux map in kW/m2
# Elements[0] is the plate, elements [-1] is the 50 m back surface behind it

def trace(PT, field, sun_az, sun_el, nray = 1e5, dni = 950):
    for h in field.heliostats:
        h.update_geometry(PT, sun_az, sun_el)
    simulate_soltrace(PT, dni=dni, nray=nray, seed=123, nthreads=1)
    res = process_soltrace_results(PT, field.field_area)
    plate = PT.stages[-1].elements[0]
    flux = PT.bin_rays(plate, nx=25, ny=25) / 1e3
    return res, flux

############################################################################################################################
# Step 2

# An episode has to move through time, so the sun must come from a day and and hour instead
# The environment will keep a variable "hour" and add dt/60 to it every step

from util import sun_position

LATITUDE = 34.85 # Dagget, CA

def sun_at(day, hour):
    return sun_position(LATITUDE, day, hour)

# test: same aim points, sun advancing one minute per trace
if __name__ == "__main__":    
    field, PT = build_field()
    axes = receiver_axes(field)
    offsets = np.zeros((len(field.heliostats), 2))   # everyone at plate center
    set_aims(field, offsets, axes)

    day = 172                                        # June 21
    for hour in [8.0, 8.0 + 1/60, 8.0 + 2/60]:       # 8:00, 8:01, 8:02 solar time
        az, el = sun_at(day, hour)
        res, flux = trace(PT, field, az, el)
        print(f"hour {hour:7.4f}  az {az:7.2f}  el {el:6.2f}"
              f"  power {res['Absorbed power (kW)']:9.1f}  peak {flux.max():8.1f}")
    

