import numpy as np
from soltrace_api import SolTrace as PySolTrace, Point
from heliostat import heliostat
from snout_soltrace import add_snout
from receiver_soltrace import add_particle_receiver


#--- The NSTTF field, the G3P3 aperture with its SNOUT and cavity, and the SolTrace scene that holds them.
#    Inputs are attributes; build_field() in functions_4_environment.py gives the layout and the measured canting,
#    then calls set_up_soltrace(). The receiver, SNOUT and thermal modules read the aperture numbers from this object.
class HeliostatField:
    def __init__(self):
        #--- Aperture (x east, y north, z up, origin at the tower base)
        self.tower_height = 45.48               # Aperture center height (m)
        self.aperture_width = 1.32              # (m)
        self.aperture_height = 1.32             # (m)
        self.aperture_azimuth = 19.414          # Aperture normal, deg east of north: aims at (0, 122) from the tower at (-40, 8.5)
        self.aperture_elevation = 0.0           # Aperture normal, deg above horizontal
        self.cavity = dict(height_ratio = 1.1, width_ratio = 1.1, curtain_depth = 0.5, tower_face = 10.0)   # Curtain size / aperture size, curtain behind the aperture (m), black face around it (m)
        self.snout = dict(depth = 0.70, horiz = 96.0, top = 0.0, bot = 34.0)   # SNOUT depth (m), full side angle, top and bottom tilts (deg)
        self.wall_reflectivity = 0.8            # SNOUT panels and cavity walls, diffuse (Mills et al. 2024)
        #--- Heliostat: real NSTTF unit, 25 facets of 1.2192 m with gaps
        self.helio_width = 6.8074               # (m)
        self.helio_height = 6.3502              # (m)
        self.n_facets = (5, 5)                  # Facets across and up
        self.facet_gaps = (0.0, 0.0)            # Used only without measured facet data (m)
        self.facet_aperture = (1.2192, 1.2192)  # One facet (m)
        self.facet_data = None                  # One (25, 6) array per heliostat: facet origin x, y, z and canting normal nx, ny, nz, face-up frame
        self.facet_focal_length = 146.19        # Focal length of every measured facet (m)
        self.helio_reflectivity = 0.885         # Clean mirror
        self.helio_soiling = 1.0                # Soiling factor
        self.slope_error = 0.0012               # Surface slope error, each of x and y (rad)
        self.specularity_error = 0.00005        # Reflected beam error, each of x and y (rad)
        self.azimuth_error = 0.0                # Pointing errors (rad)
        self.elevation_error = 0.0
        #--- Atmosphere: DELSOL3 clear day, one heliostat optic per distance band so the trace carries the attenuation
        self.include_attenuation = True
        self.n_attenuation_groups = 60
        #--- Filled by set_layout() and set_up_soltrace()
        self.positions = None
        self.aim = None
        self.attenuation = None
        self.attenuation_group = None
        self.focal_lengths = None
        self.heliostats = []
        self.field_area = 0.0
        self.opt_snout = None
        self.snout_stage = None
        self.snout_panels = []
        self.receiver_stage = None
        self.curtain = None

    #--- Fixed layout: pivot positions (m), one attenuation factor per heliostat and its distance band
    def set_layout(self, x, y, z):
        self.positions = np.column_stack([x, y, z]).astype(float)
        self.aim = np.array([0.0, 0.0, self.tower_height])
        distance = np.linalg.norm(self.positions - self.aim, axis = 1)
        distance_km = distance / 1000.0
        if self.include_attenuation:
            self.attenuation = 1.0 - (0.006789 + 0.1046 * distance_km - 0.0170 * distance_km**2 + 0.002845 * distance_km**3)
        else:
            self.attenuation = np.ones(len(distance))
        n_groups = self.n_attenuation_groups if self.include_attenuation else 1
        edges = np.linspace(distance.min(), distance.max(), n_groups + 1)
        self.attenuation_group = np.clip(np.searchsorted(edges, distance, side = 'right') - 1, 0, n_groups - 1)
        self.focal_lengths = distance                   # Slant range, used only without measured facet data

    #--- Average attenuation of each distance band
    def group_attenuation(self):
        n_groups = self.attenuation_group.max() + 1
        count = np.bincount(self.attenuation_group, minlength = n_groups)
        total = np.bincount(self.attenuation_group, weights = self.attenuation, minlength = n_groups)
        return total / np.maximum(count, 1)

    #--- Sun, optics, heliostats, SNOUT, cavity and curtain into one SolTrace scene. Returns PT.
    def set_up_soltrace(self):
        PT = PySolTrace()
        sun = PT.add_sun()
        sun.position = Point(0.0, 0.0, 100.0)
        sun.shape = 'd'
        sun.sigma = 4.65
        angle = np.linspace(0, 4.65, 26)                                        # Limb-darkened sun shape
        intensity = np.maximum(1.0 - 0.5138 * (angle / 4.65)**4, 0.0)
        intensity[-1] = 0.0
        sun.user_intensity_table = [[angle[i], intensity[i]] for i in range(len(angle))]

        reflectivity = self.helio_reflectivity * self.helio_soiling
        slope_error_mrad = 1000 * np.sqrt(0.5 * (2 * self.slope_error**2 + self.azimuth_error**2 + self.elevation_error**2))
        spec_error_mrad = 1000 * self.specularity_error
        mirror_optics = []
        for group, attenuation in enumerate(self.group_attenuation()):
            optic = PT.add_optic('Heliostat %d' % group)
            for face in (optic.front, optic.back):                              # Back same as front so flipped facets still reflect
                face.reflectivity = reflectivity * attenuation
                face.slope_error = slope_error_mrad
                face.spec_error = spec_error_mrad
            mirror_optics.append(optic)

        stage = PT.add_stage()
        stage.name = 'Heliostat field'
        self.heliostats = []
        self.field_area = 0.0
        aim_point = Point(*self.aim)
        for index, position in enumerate(self.positions):
            if self.facet_data is not None:
                helio = heliostat(Point(*position), (self.helio_width, self.helio_height), self.n_facets, self.facet_gaps, self.facet_focal_length, aim_point,
                                  facet_data = self.facet_data[index], facet_aperture = self.facet_aperture)
            else:
                focal_length = self.focal_lengths[index]
                helio = heliostat(Point(*position), (self.helio_width, self.helio_height), self.n_facets, self.facet_gaps, focal_length, aim_point,
                                  onaxis_canting_distance = focal_length)
            helio.create_geometry(PT, mirror_optics[self.attenuation_group[index]])
            self.heliostats.append(helio)
            self.field_area += helio.heliostat_area

        opt_black = PT.add_optic('Black')
        opt_black.front.reflectivity = 0.0
        opt_black.back.reflectivity = 0.0
        self.opt_snout = PT.add_optic('SNOUT Surface')
        for face in (self.opt_snout.front, self.opt_snout.back):
            face.dist_type = 'f'
            face.reflectivity = self.wall_reflectivity
            face.slope_error = 0.00001
            face.spec_error = 1000.0
        if self.snout is not None:
            self.snout_stage, self.snout_panels = add_snout(PT, self, self.opt_snout)
        self.receiver_stage, self.curtain = add_particle_receiver(PT, self, opt_black, self.opt_snout)
        for helio in self.heliostats:
            helio.set_receiver(self.curtain['element'])
        return PT


#--- One trace. nray is the number of rays that hit something; the GPU runner ignores nthreads.
def simulate_soltrace(PT, dni, nray, seed = -1, nthreads = 1, as_power_tower = False):
    PT.num_ray_hits = int(nray)
    PT.max_rays_traced = int(nray) * 100
    PT.is_sunshape = True
    PT.is_surface_errors = True
    PT.dni = dni
    PT.run(seed, as_power_tower = as_power_tower, nthread = nthreads)
    return PT.raydata, PT.powerperray, PT.num_ray_hits


#--- Power on the curtain and the efficiency chain from the ray counts (same accounting as SolarPILOT's process_raytrace_simulation).
#    The curtain is element 0 of the last stage.
def process_soltrace_results(PT, field_area, flux_bins_x = 25, flux_bins_y = 25):
    rays = PT.raydata
    receiver_stage = len(PT.stages)
    on_receiver = rays[rays['stage'] == receiver_stage]
    results = {}
    absorbed_kw = len(on_receiver[on_receiver['element'] == -1]) * PT.powerperray / 1e3
    results['Absorbed power (kW)'] = absorbed_kw
    results['Field efficiency (%)'] = absorbed_kw * 100.0 / (field_area * PT.dni / 1e3)
    flux = PT.bin_rays(PT.stages[-1].elements[0], nx = flux_bins_x, ny = flux_bins_y)
    results['Max flux (kW/m^2)'] = flux.max() / 1e3
    results['Average flux (kW/m^2)'] = flux.mean() / 1e3
    results['Peak concentration ratio (-)'] = flux.max() / PT.dni
    results['Average concentration ratio (-)'] = flux.mean() / PT.dni

    on_field = rays[rays['stage'] == 1]
    rays_in = on_field['number'].nunique()                                       # Rays that hit the field
    hits_per_ray = on_field['number'].value_counts()
    several_hits = on_field[on_field['number'].isin(hits_per_ray[hits_per_ray > 1].index)]
    blocked = len(several_hits[several_hits['element'] < 0])                     # Absorbed by a second mirror after reflecting
    one_hit = on_field[on_field['number'].isin(hits_per_ray[hits_per_ray == 1].index)]
    absorbed_by_mirrors = len(one_hit[one_hit['element'] < 0])                   # Reflectivity and attenuation loss
    rays_out = rays_in - blocked - absorbed_by_mirrors
    receiver_in = len(on_receiver[on_receiver['element'].abs() == 1])
    receiver_absorbed = len(on_receiver[on_receiver['element'] == -1])
    sun_box_area = (PT.sunstats['xmax'] - PT.sunstats['xmin']) * (PT.sunstats['ymax'] - PT.sunstats['ymin'])
    results['Shadowing and cosine efficiency (%)'] = rays_in / PT.sunstats['nsunrays'] * sun_box_area / field_area * 100.0
    results['Blocking efficiency (%)'] = (1.0 - blocked / (rays_in - absorbed_by_mirrors)) * 100.0
    results['Reflection and attenuation efficiency (%)'] = (rays_in - absorbed_by_mirrors) / rays_in * 100.0
    results['Intercept efficiency (%)'] = receiver_in / rays_out * 100.0
    results['Absorption efficiency (%)'] = receiver_absorbed / receiver_in * 100.0
    results['Field efficiency w/o receiver (%)'] = results['Field efficiency (%)'] / results['Absorption efficiency (%)'] * 100.0
    return results
