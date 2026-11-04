# This is the physics

#   build_field    -> "Build field, once"
#   receiver_axes  ->  helper for the Action box (which way is "up" on the aperture)
#   set_aims       -> "Action space"
#   trace          -> "Ray trace"

import numpy as np
import pandas as pd
from copylot_to_soltrace import Heliostat_Field, simulate_soltrace, process_soltrace_results
from api.pysoltrace import Point
from snout_soltrace import snout_results, panel_power_map
from snout_thermal import SnoutThermal
from receiver_soltrace import curtain_flux
from receiver_thermal import ReceiverThermal


# Build the plant once. G3P3 at the NSTTF: real heliostat layout and measured facet canting
# SolTrace gets a 3D scene built from that layout
# Returns the field object (list of heliostats) and PT (the SolTrace scene)

def build_field():
    field = Heliostat_Field()
    field.tower_height = 45.48          # G3P3 aperture center (NSTTF example)
    field.rec_type = 3
    field.rec_height = 1.32
    field.rec_width = 1.32
    field.rec_elevation = 0.0
    field.rec_azimuth = 19.414          # aperture aims at (0, 122) from the tower at (-40, 8.5)
    field.cavity = dict(height_ratio=1.1, width_ratio=1.1, curtain_depth=0.5, tower_face=10.0)
    field.snout = dict(depth=0.70, horiz=96.0, top=0.0, bot=34.0)
    field.helio_height = 6.3502
    field.helio_width = 6.8074          # real NSTTF heliostat: 25 facets of 1.2192 m with gaps
    field.helio_n_cant_x = 5
    field.helio_n_cant_y = 5
    field.helio_focus_method = 1
    field.n_focus_bands = 3
    field.helio_reflectivity = 0.885
    field.helio_soiling = 1.0
    field.helio_surf_err = 0.0012       # 1.2 mrad slope error
    field.helio_reflect_err = 0.00005   # 0.05 mrad specularity
    coords = pd.read_csv('nsttf_data/coordinates.csv', header=None, names=['X', 'Y', 'Z'], encoding='utf-8-sig')
    names = pd.read_csv('nsttf_data/ids.csv', header=None)[0].str.lstrip('*')
    field.set_layout(coords['X'].values + 40.0, coords['Y'].values - 8.5, coords['Z'].values)
    field.helio_facet_aperture = (1.2192, 1.2192)
    tables = [np.genfromtxt('nsttf_data/canting/' + name + '_Off_Axis_Canting_Details_TowerTop_Target.csv',
                            delimiter=',', skip_header=1) for name in names]
    field.helio_facet_data = [np.c_[t[:, 1], t[:, 2], t[:, 3] + 0.1778, t[:, 6:9]] for t in tables]
    PT = field.set_up_soltrace()
    field.thermal = ReceiverThermal()
    field.snout_thermal = SnoutThermal(field.results['sp_parameters'], field.snout, field.snout_panels)
    return field, PT

# The aperture is turned toward the field, so "1 m sideways on the aperture" is not "1 m along x"
# This returns 3 vectors in global (x east, y north, z up) coordinates
# An aim offset (dx, dv) in meters becomes center + dx*u_h + dv*u_v
# Both vectors are perpendicular to the aperture normal, so the aim point always stays in the aperture plane

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

# dx and dv on the aperture, transformed to a global xyz aim point

def offsets_to_xyz(offsets, axes): # aim point destination
    center, u_h, u_v = axes
    o = np.asarray(offsets, dtype=float)
    return center + o[:, [0]] * u_h + o[:, [1]] * u_v # center + dx sideways + dv up

# Write one xyz aim point into each heliostat; update_geometry() reads it on the next trace()

def set_aim_xyz(field, xyz): # aim point current position
    for h, p in zip(field.heliostats, xyz):
        h.aim_point = Point(*p)

# Discrete aim points: a 3 x 5 menu on the aperture, 15 cm inside its edges, with the aperture center as point 7

def aim_grid(field, n_cols = 3, n_rows = 5, margin = 0.15):
    p = field.results['sp_parameters']
    W, H = p['receiver.0.rec_width'], p['receiver.0.rec_height']    # 1.32 m, 1.32 m aperture
    xs = np.linspace(-W / 2 + margin, W / 2 - margin, n_cols)       # left to right
    vs = np.linspace(-H / 2 + margin, H / 2 - margin, n_rows)       # bottom to top
    grid = np.array([(x, v) for v in vs for x in xs])               # k = row * n_cols + col
    return grid

def center_point(grid):
    return int(np.argmin(np.linalg.norm(grid, axis=1)))             # the menu point closest to the aperture center

# Image-size rule: near mirrors (small images) take the outer points, far mirrors (big images) the center
def outer_to_center(grid):
    return np.argsort(-np.linalg.norm(grid, axis=1), kind='stable')   # menu points from the farthest from the center to the center

def spread_by_distance(positions, grid):
    distance = np.linalg.norm(positions[:, :2], axis=1)
    rank = np.argsort(np.argsort(distance))                      # 0 = nearest to the tower
    group = rank * len(grid) // len(distance)                    # equal-size groups, near to far
    return outer_to_center(grid)[group]

# Looks up each mirror's (dx, dv) in the menu, then hands the result to the continuous set_aims, which does the conversion to aim_point as before.
# Returns the offsets array so the environment can keep it as state.

def set_aims_discrete(field, choice, grid, axes):
    offsets = grid[np.asarray(choice, dtype=int)]
    set_aims(field, offsets, axes)
    return offsets

# The Ray trace box. One sun position in, flux and temperatures out. No decisions here.
# 1. Point every mirror at its aim_point for this sun (without Bill's facet self-check, which is slow; no .stinput file either)
# 2. Run SolTrace, single thread
# 3. Count rays into power and efficiencies; absorbed power and peak flux on each SNOUT panel
# 4. Bin the hits on the curtain into a 25x25 flux map in kW/m2 and hand it to the receiver thermal model
# 5. Bin the hits on each SNOUT panel into power per cell (on the SNOUT thermal mesh) and march the SNOUT wall temperatures one step

def trace(PT, field, sun_az, sun_el, nray=1e5, dni=950, dt_s=60.0, seed=123):
    for h in field.heliostats:
        h.update_geometry(PT, sun_az, sun_el, verify=False)
    simulate_soltrace(PT, dni=dni, nray=nray, seed=seed, nthreads=1)
    res = process_soltrace_results(PT, field.field_area)

    if field.snout is not None:
        res.update(snout_results(PT, field.snout_panels))

    flux = curtain_flux(PT, field.curtain, n_height=25, n_width=25)
    res.update(field.thermal.step(flux))

    if field.snout is not None:
        res.update(field.snout_thermal.step(snout_power_maps(PT, field), t_aperture_c=res['aperture_eff_c'], dt_s=dt_s))   # cavity seen through the aperture: black body radiating Janna's QIR
    return res, flux

# Incident ray power per bin on each SNOUT panel, W, binned on the thermal model's own mesh
def snout_power_maps(PT, field):
    return [panel_power_map(PT, panel, x_edges, y_edges) for panel, (x_edges, y_edges) in zip(field.snout_panels, field.snout_thermal.panel_edges)]

# March the SNOUT walls to steady state under the flux of the last trace (normal operation starts hot)
def warm_up_snout(field, PT, t_aperture_c, dt_s=60.0, max_minutes=240, tol_c=0.5):
    power_maps = snout_power_maps(PT, field)
    field.snout_thermal.reset()
    previous_c = -1.0
    for _ in range(max_minutes):
        result = field.snout_thermal.step(power_maps, t_aperture_c=t_aperture_c, dt_s=dt_s)
        if abs(result['SNOUT wall T max (C)'] - previous_c) < tol_c:
            break
        previous_c = result['SNOUT wall T max (C)']
    return result
############################################################################################################################
# Step 2

# An episode has to move through time, so the sun must come from a day and an hour instead
# The environment will keep a variable "hour" and add dt/60 to it every step

from util import sun_position, meinel_clearsky

LATITUDE = 34.96 # SANDIA

def sun_at(day, hour):
    return sun_position(LATITUDE, day, hour)

# First and last solar hour of the day with the sun at least min_el degrees up (same declination as util.sun_position)
def hours_above(day, min_el):
    declination = np.arcsin(0.39795 * np.cos(np.radians(0.98563 * (day - 173))))
    latitude = np.radians(LATITUDE)
    cos_hour_angle = (np.sin(np.radians(min_el)) - np.sin(declination) * np.sin(latitude)) / (np.cos(declination) * np.cos(latitude))
    half_day = np.degrees(np.arccos(np.clip(cos_hour_angle, -1.0, 1.0))) / 15.0
    return 12.0 - half_day, 12.0 + half_day

# ---- DNI. Choose here, nothing else asks ----
DNI_MODEL = 'clearsky'                   # 'clearsky' (Meinel model, default) or 'measured' (NSRDB 5-minute file, real clouds)
WEATHER_YEAR = 2021                      # only for 'measured'; 2018 to 2025 available, 2021 is closest to the 8-year mean annual DNI
WEATHER_FILE = 'NSRDB_alburquerque_weather/850562_35.08_-106.65_{year}.csv'   # NSRDB GOES 5-minute, PSM v4, nsrdb.nlr.gov data viewer

ALTITUDE_KM = 1.62                       # NSTTF, Albuquerque
LONGITUDE = -106.51                      # NSTTF, degrees east
STANDARD_MERIDIAN = -105.0               # Mountain Standard Time (UTC-7). NSRDB files are in standard time, no daylight saving

def dni_clearsky(day, hour, sun_el):
    return meinel_clearsky(day, np.radians(90.0 - sun_el), ALTITUDE_KM)

# The environment clock is solar time (12.00 = sun highest). The weather file clock is local standard time.
# solar time = standard time + 4 min per degree of longitude from the time-zone meridian + equation of time
def equation_of_time_min(day):
    B = 2 * np.pi * (day - 1) / 365
    return 229.18 * (0.000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B) - 0.014615 * np.cos(2 * B) - 0.04089 * np.sin(2 * B))

def solar_to_standard_hour(day, solar_hour):
    return solar_hour - (4 * (LONGITUDE - STANDARD_MERIDIAN) + equation_of_time_min(day)) / 60

_weather = {}
def load_weather(year=WEATHER_YEAR):
    if year not in _weather:                                        # read the file once
        table = pd.read_csv(WEATHER_FILE.format(year=year), skiprows=2)                                   # 2 metadata rows, then Year,Month,Day,Hour,Minute,Temperature,DNI,...
        day_of_year = pd.to_datetime(dict(year=2001, month=table.Month, day=table.Day)).dt.dayofyear     # 365-day calendar: NSRDB drops Feb 29, also in leap years
        table['minute_of_year'] = (day_of_year - 1) * 1440 + table.Hour * 60 + table.Minute
        _weather[year] = table
    return _weather[year]

def dni_measured(day, hour, sun_el):
    if sun_el <= 0:
        return 0.0
    table = load_weather(WEATHER_YEAR)
    minute_of_year = (day - 1) * 1440 + 60 * solar_to_standard_hour(day, hour)
    return float(np.interp(minute_of_year, table['minute_of_year'].to_numpy(), table['DNI'].to_numpy(dtype=float)))

if DNI_MODEL not in ('clearsky', 'measured'):
    raise ValueError(f"DNI_MODEL = '{DNI_MODEL}', choose 'clearsky' or 'measured'")
dni_at = dni_measured if DNI_MODEL == 'measured' else dni_clearsky    # (day, solar hour, sun_el) -> W/m2; the environment calls this
DNI_LABEL = f'measured DNI {WEATHER_YEAR}' if DNI_MODEL == 'measured' else 'clear-sky DNI'      # for plot titles
DNI_TAG = f'measured{WEATHER_YEAR}' if DNI_MODEL == 'measured' else 'clearsky'                   # for output file names

# test: same aim points, sun advancing one minute per trace
if __name__ == "__main__":
    field, PT = build_field()
    axes = receiver_axes(field)
    grid = aim_grid(field)
    n = len(field.heliostats)

    print("aim menu, k: (dx,dv) in meters")
    for k, (dx, dv) in enumerate(grid):
        print(f"  {k:2d}: ({dx:5.2f}, {dv:5.2f})")

    az, el = sun_at(172, 9.0)

    # A: every mirror on the center point
    set_aims_discrete(field, np.full(n, center_point(grid)), grid, axes)
    res_A, flux_A = trace(PT, field, az, el)

    # B: mirrors dealt around all 15 points in turn
    set_aims_discrete(field, np.arange(n) % len(grid), grid, axes)
    res_B, flux_B = trace(PT, field, az, el)

    for key in ['Absorbed power (kW)', 'Intercept efficiency (%)']:
        print(f"{key:28s}  A: {res_A[key]:10.2f}   B: {res_B[key]:10.2f}")
    print(f"{'peak flux on curtain (kW/m2)':28s}  A: {flux_A.max():10.2f}   B: {flux_B.max():10.2f}")