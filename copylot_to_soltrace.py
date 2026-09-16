import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from vispy import scene
import pyvista as pv

import time
import copy
import pickle
from multiprocessing import Pool

from api.copylot import CoPylot
from api.pysoltrace import PySolTrace, Point
from heliostat import heliostat

plt.rcParams['font.size'] = 15.0
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5


class Heliostat_Field:

    def __init__(self):
        # Set solarpilot parameters
        self.weather_file =  "./weather_files/USA CA Daggett Barstow-daggett Ap (TMY3).csv"     # Weather file
        self.rec_design_power = 670.    # Design thermal power delivered from the solar field (MWt)
        self.tower_height = 195.0       # Tower optical height (m)

        # Layout setup
        self.des_sim_detail = 5         # 0 = Do not filter heliostats; 1 = Single simulation point; 3 = annual simulation; 5 = Representative profiles (default)
        self.hsort_method = 7           # 0 = Power to receiver; 1 = Total; 2 = Cosine; 3 = Attenuation; 4 = Intercept; 5 = Blocking; 6 = Shadowing; 7 = TOU-weighted power (default)
        self.des_sim_ndays = 4          # For limited annual simulation, the number of evenly spaced days to simulate
        self.des_sim_nhours = 2         # Simulation will run with the specified hourly frequency (1=every hour, 2=every other hour...)

        self.is_opt_zoning = False      # Enable optical layout zone method
        self.field_accept_max = 90.0    # Maximum solar field extent angle (deg)
        self.field_accept_min = -90.0   # Minimum solar field extent angle (deg)
        self.land_max_scaled_rad = 9.5  # Land max boundary (relative to tower height)
        self.land_min_scaled_rad = 0.75 # Land min boundary (relative to tower height)

        # Receiver
        self.rec_type = 2               # Receiver type  (0 = cylinderical, 2 = flat plate)
        self.rec_height = 21.6          # Receiver height (m)
        self.rec_width = 17.0           # Receiver width (m)
        self.rec_diameter = 17.65       # Receiver diameter (m) (Only used if rec_type == 0)
        self.rec_elevation = 0.0        # Receiver orientation elevation (deg)
        self.rec_azimuth = 0.0          # Receiver orientation azimuth (deg)
        self.rec_solar_abs = 0.94       # Solar absorptivity

        # Heliostat -> Ivanpah-like defaults
        self.helio_height = 12.2        # Heliostat height (m)
        self.helio_width = 12.2         # Heliostat width (m)
        self.helio_is_faceted = True    # Use multiple panels
        self.helio_n_cant_x = 2         # Number of canting facets in x (width) direction
        self.helio_n_cant_y = 8         # Number of canting facets in y (height) direction
        self.helio_x_gap = 0.0          # Gap between facets in x direction (m)
        self.helio_y_gap = 0.0          # Gap between facets in y direction (m)

        # Canting
        self.helio_cant_method = -1     # No canting=0; On-axis at slant=-1; On-axis, user-defined=1; NOT Supported: Off-axis, day and hour=3; User-defined vector=4

        # Optical errors
        self.helio_elev_err = 0.0       # [rad] Elevation pointing error
        self.helio_azi_err = 0.0        # [rad] Azimuth pointing error
        self.helio_surf_err = 0.00153   # [rad] Heliostat surface slope error in each x and y
        self.helio_reflect_err = 0.0002 # [rad] Reflected beam error

        self.helio_reflectivity = 0.95  # Average reflectivity (clean) of the mirrored surface
        self.helio_soiling = 0.95       # Average soiling factor

        # Attenuation
        self.include_attenuation = True  # Include attenuation? If True the default SolarPILOT "DELSOL3 clear day" model will be used, if False all atmospheric attenuation will be ignored
        self.n_atten_incr = 60           # Number of heliostat "groups" to use for adjusting reflectivity to account for attenuation

        # Focusing methods
        self.helio_focus_method = 1             # Flat=0; At slant=1; Group average=2; User-defined=3
        self.n_focus_bands = 0                  # helio_focus_method must be set to 1
        self.focus_bands_method = 'average'     # 'mid-point', 'average', 'maximum'
        # TODO: SolarPILOT field design will not reflect n_focus_bands

        #--- Month/day/Hour for flux profile calculation and image size priority aiming
        # SolarPILOT inputs for performance simulation
        self.flux_month = 6
        self.flux_day = 20
        self.flux_hour = 11.815
        self.flux_dni = 950   # DNI used during trace (W/m^2)

        self.flux_xres = 25   # Flux profile resolution in circumferential dimension (rec_type == 0). This also dictates the set of allowable aim points
        self.flux_yres = 25

        # Aim point method
        self.aim_method = 0            # Simple aim points=0; Sigma aiming=1; Probability shift=2; Image size priority=3; Keep existing=4; Freeze tracking=5
        self.flux_sigma_limit_x = 2.0  # Flux profile min. image offset from receiver edge
        self.flux_sigma_limit_y = 2.0  # Flux profile min. image offset from receiver edge

        #--- Results
        self.results = {}

        self.heliostats = []  # List of heliostat objects
        self.field_area = 0.0 # Total field area (m^2)

        return

    #--- Returns a dictionary with SolarPILOT variable names.  All parameters not specified use SolarPILOT default values
    def get_solarpilot_parameters(self):
        # TODO: need to confirm all parameters are passed correctly

        D = { # Climate parameters
             'ambient.0.weather_file': self.weather_file,                       # Weather file
             'ambient.0.sun_type': 1,                                           # Sun shape: Pillbox sun=2;Gaussian sun=4;Limb-darkened sun=1;Point sun=0;Buie CSR=5;User sun=3
             'ambient.0.atm_model': 0 if self.include_attenuation else 2,        # Atmospheric attenuation model: DELSOL3 clear day=0;DELSOL3 hazy day=1;User-defined=2'

             # Layout setup, design simulation data
             'solarfield.0.q_des': self.rec_design_power,           # Design thermal power delivered from the solar field (MWt)
             'solarfield.0.tht': self.tower_height,                 # Tower optical height (m)
             'solarfield.0.des_sim_detail': self.des_sim_detail,    # 0 = Do not filter heliostats; 1 = Single simulation point; 5 = Representative profiles (default)
             'solarfield.0.hsort_method': self.hsort_method,        # 0 = Power to receiver; 1 = Total; 2 = Cosine; 3 = Attenuation; 4 = Intercept; 5 = Blocking; 6 = Shadowing; 7 = TOU-weighted power (default)
             'solarfield.0.des_sim_ndays': self.des_sim_ndays,      # For limited annual simulation, the number of evenly spaced days to simulate
             'solarfield.0.des_sim_nhours': self.des_sim_nhours,    # Simulation will run with the specified hourly frequency (1=every hour, 2=every other hour...)


             'solarfield.0.is_opt_zoning': self.is_opt_zoning,      # Enables grouping of heliostats into zones for intercept factor calculation during layout only
             'solarfield.0.accept_max': self.field_accept_max,      # Maximum solar field extent angle (deg)
             'solarfield.0.accept_min': self.field_accept_min,      # Minimum solar field extent angle (deg)
             'land.0.max_scaled_rad': self.land_max_scaled_rad,     # Maximum radius (in units of tower height) for placing heliostats
             'land.0.min_scaled_rad': self.land_min_scaled_rad,     # Minimum radius (in units of tower height) for positioning of the heliostats

             # Heliostat design
             'heliostat.0.is_faceted': True if self.helio_n_cant_x > 1 or self.helio_n_cant_y > 1 else False,      # Use multiple panels
             'heliostat.0.height': self.helio_height,               # Heliostat height (m)
             'heliostat.0.width': self.helio_width,                 # Heliostat width (m)
             'heliostat.0.n_cant_x': self.helio_n_cant_x,           # Number of canting facets in x (width) direction
             'heliostat.0.n_cant_y': self.helio_n_cant_y,           # Number of canting facets in y (height) direction
             'heliostat.0.x_gap': self.helio_x_gap,                 # Gap between facets in x direction (m)
             'heliostat.0.y_gap': self.helio_y_gap,                 # Gap between facets in y direction (m)
             #'heliostat.0.x_focal_length': self.helio_focal_length, # Heliostat focal length (m) in x direction     #TODO: update these...
             #'heliostat.0.y_focal_length': self.helio_focal_length, # Heliostat focal length (m) in y direction

             'heliostat.0.err_azimuth':   self.helio_azi_err,       # Standard deviation of the normal error dist. of the azimuth angle
             'heliostat.0.err_elevation': self.helio_elev_err,      # Standard deviation of the normal error dist. of the elevation angle
             'heliostat.0.err_reflect_x': self.helio_reflect_err,   # error in reflected vector (horiz.) caused by atmospheric refraction, tower sway, etc.
             'heliostat.0.err_reflect_y': self.helio_reflect_err,   # error in reflected vector (vert.) caused by atmospheric refraction, tower sway, etc.
             'heliostat.0.err_surface_x': self.helio_surf_err,      # Heliostat surface slope error in x (rad)
             'heliostat.0.err_surface_y': self.helio_surf_err,      # Helisotat surface slope error in y (rad)
             'heliostat.0.reflectivity': self.helio_reflectivity,   # Average reflectivity (clean) of the mirrored surface
             'heliostat.0.soiling': self.helio_soiling,             # Average soiling factor
             'heliostat.0.reflect_ratio': 1.0,                      # Ratio of mirror area to total area of the heliostat defined by wm x hm
             'heliostat.0.focus_method': self.helio_focus_method,   # 0 = Flat, 1 = At slant (each heliostat has a focal length dictated by its distance to the receiver)
             # TODO: should we set up solarpilot to use focal bands?

             # Receiver design
             'receiver.0.rec_type': self.rec_type,           # Receiver type: 0 = External cylindrical, 1 = Cavity, 2 = Flat plate, 3 = Falling Particle
             'receiver.0.rec_height': self.rec_height,       # Receiver height (m)
             'receiver.0.rec_diameter': self.rec_diameter,   # Receiver diameter (m)
             'receiver.0.rec_width': self.rec_width,         # Receiver width (m)
             'receiver.0.rec_azimuth': self.rec_azimuth,     # Receiver azimuth (deg)
             'receiver.0.rec_elevation': self.rec_elevation, # Receiver elevation (deg)
             'receiver.0.absorptance': self.rec_solar_abs,   # Receiver solar absorptivity
             'receiver.0.therm_loss_base': 0.0,              # NOTE: Turning off thermal losses for now
             'receiver.0.piping_loss_coef': 0.0,             # NOTE: Turning off piping losses as well
             # TODO: update receiver acceptance angles?

             # Flux simulation
             'fluxsim.0.flux_month':self.flux_month,                # Month for flux profile
             'fluxsim.0.flux_day': self.flux_day,                   # Day for flux profile
             'fluxsim.0.flux_hour':self.flux_hour,                  # Time of day for flux profile
             'fluxsim.0.flux_dni':self.flux_dni,                    # DNI for flux profile

             'fluxsim.0.flux_solar_az': 180.0,                     # Solar azimuth for flux profile
             'fluxsim.0.flux_solar_el': 55.0,                      # Solar elevation for flux profile

             'fluxsim.0.aim_method':self.aim_method,                # Simple aim points=0;Sigma aiming=1;Probability shift=2;Image size priority=3;Keep existing=4;Freeze tracking=5
             'fluxsim.0.sigma_limit_x':self.flux_sigma_limit_x,     # Min image offset
             'fluxsim.0.sigma_limit_y':self.flux_sigma_limit_y,
             'fluxsim.0.x_res': self.flux_xres,        # Flux profile resolution in vertical dimension (note this also dictates the set of allowable aim points)
             'fluxsim.0.y_res': self.flux_yres
             }

        if not self.include_attenuation:
            D['ambient.0.atm_coefs'] = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]  # Atmospheric attenuation coefficients
        return D

    #---- Run SolarPILOT using current parameters. All parameters not specified will use SolarPILOT defaults
    def generate_field_via_copylot(self, display_results = False):
        cp = CoPylot()
        data = cp.data_create()

        R = {}
        R['sp_parameters'] = self.get_solarpilot_parameters()  # Create dictionary based on current parameters

        if (not cp.set_data(data, R['sp_parameters'])):
            raise Exception("Failed to set some variables within SolarPILOT.")

        #--- Generate heliostat field layout (default SolarPILOT settings use summer solstice solar noon as design point, simple aim points)
        if display_results:
            print('Generating heliostat field layout via SolarPILOT...')
        cp.generate_layout(data)  # Generate field layout - NOTE: This can be very slow and memory intensive for large number of heliostats
        # TODO: We would need to update focal lengths here if using focal bands... and regenerate layout

        if display_results:
            print('Simulating heliostat field via SolarPILOT...')

        #--- Simulate heliostat field performance to get flux map and aimpoints
        #NOTE: could skip this if not using SolarPILOT aiming...
        cp.simulate(data)                                   # Simulate flux
        R['sp_incident_flux'] = np.array(cp.get_fluxmap(data))       # NOTE: flux maps are incident flux maps,  not absorbed
        R['sp_summary'] = cp.summary_results(data)
        R['sp_detailed_field'] = cp.detail_results(data)

        if R['sp_detailed_field'] is None:
            raise Exception("SolarPILOT failed to generate heliostat field!")

        # Get simulation data from weather file
        R['sp_sim_data'] = pd.DataFrame(cp.generate_weather_simulation_data(data, self.des_sim_detail))
        R['sp_sim_data'].columns = ['month', 'day', 'hour', 'dni', 'temp', 'wind', 'step_weight', 'azimuth', 'zenith']
        R['sp_sim_data'] = R['sp_sim_data'][(R['sp_sim_data']['dni'] > 0.0) & (R['sp_sim_data']['zenith'] < 80.0)]          # Filtering out zero DNI and low solar angles
        R['sp_sim_data'].reset_index(inplace=True)

        if display_results:
            print('Layout results from SolarPILOT:')
            cp.summary_results(data, False)
            print(' ')

        self.results = R

        # Attenuation and focal lengths information for SolTrace setup
        self.results['st_field_info'] = dict()
        self.results['st_field_info'].update(self.create_attenuation_groups())
        self.results['st_field_info'].update(self.create_focal_lengths(numFocalBands=self.n_focus_bands))

        cp.data_free(data)

    #--- create attenuation groups
    def create_attenuation_groups(self):
        # Generate average attenuation in increments (used to include attenuation in SolTrace)
        sp_field = self.results['sp_detailed_field']
        nhel = len(sp_field['x_aimpoint'].values)
        dist = ((sp_field['x_aimpoint'].values-sp_field['x_location'].values)**2
                + (sp_field['y_aimpoint'].values-sp_field['y_location'].values)**2
                + (sp_field['z_aimpoint'].values-sp_field['z_location'].values)**2)**0.5
        distpts = np.linspace(dist.min(), dist.max(), self.n_atten_incr+1, endpoint = True)
        atten_avg = np.zeros(self.n_atten_incr)
        atten_id = np.zeros(nhel, dtype = int)
        for j in range(self.n_atten_incr):
            inds = np.where(np.logical_and(dist>= distpts[j], dist<distpts[j+1]))[0]
            atten_avg[j] = sp_field['attenuation'].values[inds].mean() if len(inds)>0 else 0.0
            atten_id[inds] = j

        A = dict()
        A['attenuation_avg_pts'] = atten_avg.tolist()
        A['attenuation_dist_pts'] = distpts.tolist()
        A['attenuation_id'] = atten_id.tolist()
        return A

    #--- create focal lengths
    def create_focal_lengths(self, numFocalBands: int = -2):
        """
            Creates focal lengths for each heliostat based on distance to receiver and number of focal bands.

            Args:
                numFocalBands (int): Number of focal bands to create. If -2 (default), use existing self.n_focus_bands.
                                     If -1, perfect focus for all heliostats. If 0, flat heliostats.

            Returns:
                dict: Dictionary with 'helio_focal_length' and 'focus_method' for soltrace geometry.
        """
        sp_field = self.results['sp_detailed_field']
        sp_params = self.results['sp_parameters']

        if numFocalBands >= -1:
            self.n_focus_bands = numFocalBands
            self.helio_focus_method = 1     # Use heliostat focal lengths calculated below
            if numFocalBands == -1:         # Perfect focus case
                self.n_focus_bands = 0
            if numFocalBands == 0:          # Flat case
                self.helio_focus_method = 0
        elif numFocalBands != -2:           # Default value
            raise Exception("Failed to set number of field focal bands.")

        # Heliostat positions
        pos = np.transpose(np.array([sp_field[k] for k in ['x_location', 'y_location', 'z_location']]))

        # Focal length (based on simple aim point)
        ax = 0
        ay = 0
        if sp_params['receiver.0.rec_type'] == 0:
            view_az = np.arctan2(pos[:,0], pos[:,1])
            ax = 0.5*sp_params['receiver.0.rec_diameter']*np.sin(view_az)
            ay = 0.5*sp_params['receiver.0.rec_diameter']*np.cos(view_az)

        # Perfect focused for all heliostats
        res = {}
        res['helio_focal_length'] = (np.sqrt((ax-pos[:,0])**2 + (ay-pos[:,1])**2 + (sp_params['solarfield.0.tht']-pos[:,2])**2)).tolist()
        res['total_field_rows'] = pd.DataFrame(res['helio_focal_length']).nunique()[0]

        # Create focal bands
        if self.n_focus_bands > 0:
            min_fl = min(res['helio_focal_length'])
            max_fl = max(res['helio_focal_length'])
            delta_fl = (max_fl - min_fl) / self.n_focus_bands
            focal_lims = [i*delta_fl + min_fl for i in range(self.n_focus_bands+1)]

            focal_lengths = list()
            if self.focus_bands_method == 'mid-point':
                # Mid-point of bands
                focal_lengths = [(focal_lims[i] + focal_lims[i+1])/2 for i in range(len(focal_lims) - 1)] # Using the mid-point
                # buffers on limits
                focal_lims[0] -= 5
                focal_lims[-1] += 5
            elif self.focus_bands_method == 'average':
                # Average focal length within bands
                # buffers on limits
                focal_lims[0] -= 5
                focal_lims[-1] += 5
                for i in range(len(focal_lims) - 1):
                    avg_fl = 0.0
                    hel_count = 0
                    for hel_fl in res['helio_focal_length']:
                        if (hel_fl >= focal_lims[i]) and (hel_fl < focal_lims[i+1]):
                            avg_fl += hel_fl
                            hel_count += 1
                    focal_lengths.append(avg_fl/hel_count)
                    if avg_fl == 0.0:
                        raise Exception('No heliostats found in focal length band {} to {} m.'.format(focal_lims[i], focal_lims[i+1]))
            elif self.focus_bands_method == 'maximum':
                # Maximum focal length within bands
                # buffers on limits
                focal_lims[0] -= 5
                focal_lims[-1] += 5
                for i in range(len(focal_lims) - 1):
                    max_fl_band = 0.0
                    for hel_fl in res['helio_focal_length']:
                        if (hel_fl >= focal_lims[i]) and (hel_fl < focal_lims[i+1]):
                            if hel_fl > max_fl_band:
                                max_fl_band = hel_fl
                    focal_lengths.append(max_fl_band)
                    if max_fl_band == 0.0:
                        raise Exception('No heliostats found in focal length band {} to {} m.'.format(focal_lims[i], focal_lims[i+1]))
            else:
                raise Exception('focus_bands_method {} is not supported.'.format(self.focus_bands_method))

            # Assign focal lengths to heliostats based on bands
            for idx, hel_fl in enumerate(res['helio_focal_length']):
                for i in range(len(focal_lims)-1):
                    if (hel_fl >= focal_lims[i]) and (hel_fl < focal_lims[i+1]):
                        res['helio_focal_length'][idx] = focal_lengths[i]
        return res

    #--- Set up heliostat field and receiver in for SolTrace.
    def set_up_soltrace(self):
        PT = PySolTrace()

        sp_params = self.results['sp_parameters']
        sp_field = self.results['sp_detailed_field']
        #--- Create sun (using limb-darkened sun-shape)
        sun = PT.add_sun()
        sun.position = Point(0.0, 0.0, 100.0)
        sun.shape = 'd'
        sun.sigma = 4.65

        # Limb-darkened sun-shape
        pts = np.linspace(0, 4.65, 26)
        intensity = np.maximum(1.0 - 0.5138*((pts/4.65)**4), 0.0)
        intensity[-1] = 0.0
        sun.user_intensity_table = [[pts[i], intensity[i]] for i in range(len(pts))]

        #--- Create heliostat optical property sets
        refl = sp_params['heliostat.0.reflectivity'] * sp_params['heliostat.0.soiling'] * sp_params['heliostat.0.reflect_ratio']
        errslope = 1000*(0.5*(sp_params['heliostat.0.err_surface_x']**2
                              + sp_params['heliostat.0.err_surface_y']**2
                              + sp_params['heliostat.0.err_azimuth']**2
                              + sp_params['heliostat.0.err_elevation']**2))**0.5
        errspec = 1000*(0.5*(sp_params['heliostat.0.err_reflect_x']**2
                             + sp_params['heliostat.0.err_reflect_y']**2))**0.5

        if not self.include_attenuation:
            opth = PT.add_optic("Heliostat")
            opth.front.reflectivity = refl
            opth.front.slope_error = errslope
            opth.front.spec_error = errspec
            opth.back.reflectivity = refl       # 0.0   NOTE: Setting back and front the same so that flipping surfaces works correctly
            opth.back.slope_error = errslope    # 100.
            opth.back.spec_error = errspec      # 0.0
        else:
            st_field_info = self.results['st_field_info']
            nhelopt = len(st_field_info['attenuation_avg_pts'])
            opth = [None for j in range(nhelopt)]
            for j in range(nhelopt):
                opth[j] = PT.add_optic("Heliostat %d"%j)
                opth[j].front.reflectivity = refl*st_field_info['attenuation_avg_pts'][j]
                opth[j].front.slope_error  = errslope
                opth[j].front.spec_error = errspec
                opth[j].back.reflectivity = refl*st_field_info['attenuation_avg_pts'][j]    # 0.0
                opth[j].back.slope_error = errslope                                         # 100.
                opth[j].back.spec_error = errspec                                           # 0.0

        #--- Create heliostat field stage (Stage 0)
        st = PT.add_stage()
        st.name = 'Heliostat field'
        self.field_area = 0.0
        self.heliostats = list()
        for i in range(len(sp_field['x_location'])):
            position = Point(sp_field['x_location'][i],
                             sp_field['y_location'][i],
                             sp_field['z_location'][i])
            aperture_size = (sp_params['heliostat.0.width'],
                             sp_params['heliostat.0.height'])
            number_panels = (int(sp_params['heliostat.0.n_cant_x']),
                             int(sp_params['heliostat.0.n_cant_y']))
            gaps = (sp_params['heliostat.0.x_gap'],
                    sp_params['heliostat.0.y_gap'])
            aim_point = Point(sp_field['x_aimpoint'][i],
                              sp_field['y_aimpoint'][i],
                              sp_field['z_aimpoint'][i])

            focal_length = st_field_info['helio_focal_length'][i]   # Assuming same focal length in both x and y directions
            ## TODO: canting methods?

            helio = heliostat(position, aperture_size, number_panels, gaps, focal_length, aim_point)
            mirror_optic = opth[st_field_info['attenuation_id'][i]] if self.include_attenuation else opth
            helio.create_geometry(PT, mirror_optic)
            self.heliostats.append(helio)
            self.field_area += helio.heliostat_area

        #--- Add optical property sets for recevier
        def set_back_optics_to_front(opt):
            for k in vars(opt.front).keys():
                setattr(opt.back, k, getattr(opt.front,k))
            return

        # Generic black surface properties
        opt_black = PT.add_optic("Black")
        opt_black.front.reflectivity = 0.
        set_back_optics_to_front(opt_black)

        # Receiver surface properties
        opt_rec = PT.add_optic("Receiver")
        opt_rec.front.reflectivity = 1.0 - sp_params['receiver.0.absorptance']
        opt_rec.front.dist_type = 'f'
        set_back_optics_to_front(opt_rec)

        #--- Add Receiver
        st_rec = PT.add_stage()
        st_rec.name = 'Receiver'
        st_rec.is_virtual = False
        st_rec.is_multihit = False
        st_rec.is_tracethrough = False
        st_rec.position = Point(0.0, 0.0, sp_params['solarfield.0.tht'])
        st_rec.aim = Point(0.0, 0.0, sp_params['solarfield.0.tht'] + 1)
        st_rec.zrot = 0.0

        rec_el = st_rec.add_element()
        if sp_params['receiver.0.rec_type'] == 0:          # Cylindrical
            rec_el.position = Point(0.0, -0.5*sp_params['receiver.0.rec_diameter'], 0.0)
            rec_el.aim = Point(0.0, 0.0, 0.0)
            rec_el.zrot = 0.0
            rec_el.aperture_singleax_curve(0.0, 0.0, sp_params['receiver.0.rec_height'])
            rec_el.optic = opt_rec
            rec_el.surface_cylindrical(0.5*sp_params['receiver.0.rec_diameter'])

            # Bottom cap
            cap_el = st_rec.add_element()
            cap_el.position = Point(0.0, 0.0, -0.5*sp_params['receiver.0.rec_height'])
            cap_el.aim = Point(0.0, 0.0, 0.0)
            cap_el.zrot = 0.0
            cap_el.aperture_circle(sp_params['receiver.0.rec_diameter'])
            cap_el.optic = opt_black
            cap_el.surface_flat()

        elif sp_params['receiver.0.rec_type'] == 2:        # Flat plate
            # Receiver normal vector
            az = sp_params['receiver.0.rec_azimuth'] * np.pi / 180
            el = sp_params['receiver.0.rec_elevation'] * np.pi / 180
            rec_nv = np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])

            # Z-rotation
            alpha = np.arctan2(rec_nv[0], rec_nv[2])
            beta = np.arcsin(rec_nv[1])
            zrot = np.arctan(np.tan(alpha)/np.sin(beta))

            rec_el.position = Point(0.0, 0.0, 0.0)
            rec_el.aim = Point(rec_nv[0] * 1000, rec_nv[1] * 1000, rec_nv[2] * 1000)
            rec_el.zrot = zrot*180/np.pi
            rec_el.aperture_rectangle(sp_params['receiver.0.rec_width'], sp_params['receiver.0.rec_height'])
            rec_el.optic = opt_rec
            rec_el.surface_flat()

            back_surface = st_rec.add_element()
            back_surface.position = Point(0.0, -1.0, 0.0)
            back_surface.aim = Point(rec_nv[0] * 1000, rec_nv[1] * 1000, rec_nv[2] * 1000)
            back_surface.zrot = zrot*180/np.pi
            back_surface.aperture_rectangle(50.0, 50.0)
            back_surface.optic = opt_rec
            back_surface.surface_flat()


        # Set receiver to heliostats
        for h in self.heliostats:
            h.set_receiver(rec_el)

        return PT

#--- Plot field layout based on metric within field dictionary
def plot_field(field: dict, metric: str = 'efficiency', include_color: bool = True, savename = None):
    x = field['x_location']
    y = field['y_location']
    eff = field[metric]
    fig = plt.figure(figsize = (8.0, 5.0))
    if include_color:
        plt.scatter(x, y, marker = 's', s=5.0, c = eff, cmap = 'Blues', vmin = 0.0, vmax = eff.max())
    else:
        plt.scatter(x, y, marker = 's', s=5.0, c = 'grey')
    plt.colorbar()
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename+'.png', dpi = 500)
    else:
        plt.show()
    return

def plot_attenuation(field):
    # Plot attenuation vs. distance to aimpoint
    dist = ((field['x_aimpoint'].values-field['x_location'].values)**2 + (field['y_aimpoint'].values-field['y_location'].values)**2 + (field['z_aimpoint'].values-field['z_location'].values)**2)**0.5
    fig = plt.figure(figsize = (4.0, 3.5))
    plt.plot(dist, field['attenuation'].values, ls = 'None', marker = '.', ms = 2.0)
    plt.xlabel('Distance to tower (m)')
    plt.ylabel('Attenuation efficiency')
    plt.show()

#--- Plot field, flux, and attenuation results
def plot_results(field:dict, flux, include_attenuation:bool = False):
    # Plot layout
    plot_field(field)

    # Plot flux distribution image
    fig = plt.figure(figsize = (4.0, 3.5))
    im = plt.imshow(flux)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

    if include_attenuation:
        plot_attenuation(field)
    return




def simulate_soltrace(PT: PySolTrace, dni, nray, seed = -1, nthreads = 1, as_power_tower = False, no_callback = True, use_embree=True):
    PT.num_ray_hits = nray
    PT.max_rays_traced = nray*100
    PT.is_sunshape = True
    PT.is_surface_errors = True
    PT.dni = dni

    PT.run(seed, as_power_tower = as_power_tower, nthread = nthreads, no_callback=no_callback, use_embree=use_embree)

    df = PT.raydata
    ppr = PT.powerperray
    nhit = PT.num_ray_hits

    return df, ppr, nhit

def simulate_time_step(PT:PySolTrace, field: Heliostat_Field, sim_data, nray = 1.e5, seed = 123, nthreads = 1, no_callback = True):
    # Update heliostat geometry based on sun position
    for h in field.heliostats:
        h.update_geometry(PT, sim_data.azimuth, 90.0 - sim_data.zenith)
    #PT.write_soltrace_input_file('test_dp.stinput')
    df, ppr, nhit = simulate_soltrace(PT, dni = sim_data.dni, nray = nray , seed = seed, nthreads = nthreads, no_callback=no_callback)
    results = process_soltrace_results(PT, field.field_area)
    return results


def simulate_time_step_with_FL_correction(PT:PySolTrace, field: Heliostat_Field, sim_data, nray = 1.e5, seed = 123, nthreads = 1, no_callback = True):
    # Update heliostat geometry based on sun position
    for h in field.heliostats:
        temp_focal_length = h.calculate_temperature_focal_length(sim_data.temp)
        h.update_focal_length(temp_focal_length)
        h.update_geometry(PT, sim_data.azimuth, 90.0 - sim_data.zenith)

    #PT.write_soltrace_input_file('test_dp.stinput')
    df, ppr, nhit = simulate_soltrace(PT, dni = sim_data.dni, nray = nray , seed = seed, nthreads = nthreads, no_callback=no_callback)
    results = process_soltrace_results(PT, field.field_area)
    return results


def process_soltrace_results(PT: PySolTrace, fieldArea, fluxBinNx:int = 25, fluxBinNy:int = 25):
    """
    Returns a dictionary of results with the following keys:
        'Absorbed power (kW)'
        'Field efficiency (%)'
        'Max flux (kW/m^2)'
        'Average flux (kW/m^2)'
        'Peak concentration ratio (-)'
        'Average concentration ratio (-)'
        'Shadowing and cosine efficiency (%)'
        'Blocking efficiency (%)'
        'Reflection and attenuation efficiency (%)'
        'Intercept efficiency (%)'
        'Absorption efficiency (%)'
        'Field efficiency w/o receiver (%)'
    """
    rayData = PT.raydata

    results = {}
    rec_stage = rayData[rayData['stage'] == 2]
    # absorbed by the receiver
    abs_power = len(rec_stage[rec_stage['element'] == -1]) * PT.powerperray / 1.e3
    results['Absorbed power (kW)'] = abs_power
    results['Field efficiency (%)'] = abs_power * 100. / (fieldArea * PT.dni / 1.e3)

    flux_st = PT.bin_rays(PT.stages[-1].elements[-1], nx=fluxBinNx, ny=fluxBinNy)
    results['Max flux (kW/m^2)'] = flux_st.max() / 1.e3
    results['Average flux (kW/m^2)'] = flux_st.mean() / 1.e3

    results['Peak concentration ratio (-)'] = flux_st.max() / PT.dni
    results['Average concentration ratio (-)'] = flux_st.mean() / PT.dni

    # Process ray data - copied from SolarPILOT (interopt.cpp -> process_raytrace_simulation)
    # prevRay = 0
    # nhin = 0        # Rays that hit the heliostat field
    # nhblock = 0     # Rays blocked by heliostats
    # nhabs = 0       # Rays absorbed by heliostat (reflectivity loss)
    # nhout = 0       # Rays that exit the heliostat field
    # nrin = 0        # Rays that hit the receiver
    # nrabs = 0       # Rays absorbed by the receiver
    # for _, ray in rayData.iterrows():
    #     if ray['stage'] == 1: # Heliostat field stage
    #         if (ray['element'] > 0) and (ray['number'] != prevRay): # Reflected, no second reflection allowed
    #             nhin += 1
    #         elif ray['element'] < 0: # Absorbed
    #             if ray['number'] == prevRay:
    #                 nhblock += 1
    #             else:
    #                 nhabs += 1
    #                 nhin += 1
    #     elif ray['stage'] == 2: # Receiver stage
    #         if ray['element'] != 0:
    #             if (ray['element'] == 1) or (ray['element'] == -1):
    #                 nrin += 1   # Reflected or absorbed
    #         if (ray['element'] == -1):
    #             nrabs += 1
    #     prevRay = ray['number']
    # nhout = nhin - nhblock - nhabs

    # Count rays using dataframe operations
    field_stage = rayData[rayData['stage'] == 1]
    nhin = field_stage['number'].nunique()      # Rays that hit the heliostat field

    ray_counts = field_stage['number'].value_counts()
    multi_hits = ray_counts[ray_counts > 1].index
    fieldMultiHits = field_stage[field_stage['number'].isin(multi_hits)]
    nhblock = len(fieldMultiHits[fieldMultiHits['element'] < 0])    # Rays blocked by heliostats after initial reflection

    single_hits = ray_counts[ray_counts == 1].index
    fieldSingleHits = field_stage[field_stage['number'].isin(single_hits)]
    nhabs = len(fieldSingleHits[fieldSingleHits['element'] < 0])    # Rays absorbed by heliostats

    rec_stage = rayData[rayData['stage'] == 2]
    nrin = len(rec_stage[(rec_stage['element'] == 1) | (rec_stage['element'] == -1)])   # Rays that hit the receiver
    nrabs = len(rec_stage[rec_stage['element'] == -1])              # Rays absorbed by the receiver
    nhout = nhin - nhblock - nhabs                                  # Rays that leave the heliostat field

    # NOTE: Ray counts must be multipled by the reflect_ratio value if not equal to 1

    # Calculate efficiency metrics
    Abox = (PT.sunstats['xmax'] - PT.sunstats['xmin']) * (PT.sunstats['ymax'] - PT.sunstats['ymin'])
    nsunrays = PT.sunstats['nsunrays']

    results['Shadowing and cosine efficiency (%)'] = nhin / nsunrays * Abox / fieldArea * 100.
    results['Blocking efficiency (%)'] = (1. - (nhblock / (nhin - nhabs))) * 100.
    results['Reflection and attenuation efficiency (%)'] = (nhin - nhabs) / nhin * 100.
    results['Intercept efficiency (%)'] = nrin / nhout * 100.
    results['Absorption efficiency (%)'] = nrabs / nrin * 100.
    results['Field efficiency w/o receiver (%)'] = (results['Field efficiency (%)'] / 100.) / (results['Absorption efficiency (%)'] / 100.) * 100.

    return results

#=======================================================================================================================

# TODO: Update example to use new class structure
#--- Example calculations
if __name__ == "__main__":
    # Controls
    run_design_point = True
    run_convergence = False
    run_simulation = False

    # Creating initial solar field
    field = Heliostat_Field()
    field.rec_design_power = 50.       # [MWt] Receiver design power
    field.aim_method = 3               # Simple aim points=0; Sigma aiming=1; Probability shift=2; Image size priority=3; Keep existing=4; Freeze tracking=5
    field.include_attenuation = True   # Include atmospheric attenuation in SolTrace simulation

    # Receiver parameters
    field.rec_height = 10.0
    field.rec_width = 10.0
    field.rec_azimuth = 0
    field.rec_elevation = 0.0 #-30

    # Heliostat parameters
    field.helio_focus_method = 1        # 0 = Flat, 1 = At slant (each heliostat has a focal length dictated by its distance to the receiver)
    field.helio_size = 6.0

    R = field.create_field_via_copylot(sim_data_method = 5,   # 3 = Annual simulation, 5 = Representative profiles      # NOTE: Annual simulations take ~ 15 minutes at nrays = 1.e5
                                       display_results = True)
    plot_results(R['sp_detailed_field'], R['sp_incident_flux'], field.include_attenuation)   # Plot field, flux, and attenuation grouping
    PT = field.set_up_soltrace(R['st_field_info'])                                  # Create SolTrace geometry

    # Design point conditions
    if run_design_point:
        # PT.write_soltrace_input_file('test_dp.stinput')
        df, ppr, nhit = simulate_soltrace(PT, dni = field.flux_dni, nray = 1.e6 , seed = 123, nthreads = 12)   # nray = 2e7
        # Plot the flux map on the last element of the last stage (the receiver in this case)
        PT.plot_flux(PT.stages[-1].elements[-1], nx=25, ny=25, levels=15)

        TotalHeliostats = sum(R['st_field_info']['enabled'])
        fieldArea = TotalHeliostats * field.helio_size**2
        results = process_soltrace_results(PT, fieldArea)

        # Print out results
        print("Number of Heliostats: {:,d}".format(TotalHeliostats))
        print("SolTrace power absorbed by the Receiver: {:,.2f} kW".format(results['Absorbed power (kW)']))

        sp_abs_power = R['sp_summary']['All receivers']['Power absorbed by the receiver']
        print("SolarPILOT power absorbed by the Receiver: {:,.2f} kW".format(sp_abs_power))
        rel_error = (results['Absorbed power (kW)'] - sp_abs_power)/sp_abs_power        # TODO: Should we be comparing to target power?
        print("Relative Error: {:,.2f} %".format(rel_error*100.))

        print("SolTrace Results:")
        print("Heliostat Field Efficiency: {:.2f} %".format(results['Field efficiency (%)']))
        print("Maximum Flux: {:.2f} kW/m^2".format(results['Max flux (kW/m^2)']))
        print("Mean Flux: {:.2f} kW/m^2".format(results['Average flux (kW/m^2)']))
        print("Peak Concentration Ratio: {:,.2f}".format(results['Peak concentration ratio (-)']))
        print("Average Concentration Ratio: {:,.2f}".format(results['Average concentration ratio (-)']))

        receiverArea = field.rec_height * field.rec_width
        GeometricConcentrationRaito = fieldArea / receiverArea
        print("Geometric Concentration Ratio: {:,.2f}".format(GeometricConcentrationRaito))

        # Plot ray data
        # TODO: add missed rays?
        plot3dMethod = 2   # 1 = VisPy, 2 = PyVista
        fraction_points_to_plot = 0.5
        fraction_rays_to_plot = 0.0005
        if True:
            rayData = PT.raydata
            # rayData.to_csv('ray_data.csv')
            rayDataSize = len(rayData)

            sceneLength = rayData['loc_y'].max() - rayData['loc_y'].min()

            rayData = rayData[rayData['element'] != 0]  # Remove rays that miss all elements
            rayData.loc[:, 'loc_z'] += field.helio_size / 2.0 + 1.5   # move ground to z = 0
            rayData.loc[:, 'loc_y'] -= sceneLength / 2   # move scene center to origin
            rayData.loc[rayData['stage'] == 2, 'loc_z'] += field.tower_height   # Adjust for tower height

            xyz = rayData[['loc_x', 'loc_y', 'loc_z']].sample(n=int(rayDataSize * fraction_points_to_plot)).to_numpy()

            maxrays = rayData['number'].max()
            sampled_rays = random.sample(range(0, maxrays), int(maxrays * fraction_rays_to_plot))
            ray_segments = rayData[rayData['number'].isin(sampled_rays)]

            rays = list()
            for ray in sampled_rays:
                ray_info = ray_segments[ray_segments['number'] == ray]
                ray_path = list()
                end = ray_info[['loc_x', 'loc_y', 'loc_z']].iloc[0].to_numpy()
                direction = ray_info[['cos_x', 'cos_y', 'cos_z']].iloc[0].to_numpy()
                start = end - direction * 100.0  # Start point some distance before first hit
                ray_path.append((start, end))
                for interaction in range(len(ray_info)):
                    start = end
                    end = ray_info[['loc_x', 'loc_y', 'loc_z']].iloc[interaction].to_numpy()
                    ray_path.append((start, end))

                rays.append(np.array(ray_path))

            if plot3dMethod == 1:
                print("Launching VisPy ray plot...")
                floor = scene.visuals.Plane(width=rayData['loc_x'].max()*2 + 10,
                                            height=rayData['loc_y'].max()*2 + 10,
                                            direction='+z',
                                            color=(0.5, 0.5, 0.5, 1))
                canvas = scene.SceneCanvas(keys='interactive', show=True)
                view = canvas.central_widget.add_view()
                scatter = scene.visuals.Markers()
                scatter.set_data(xyz, edge_color='white', face_color='white', size=1)
                view.add(scatter)
                view.add(floor)

                for ray in rays:
                    starts = ray[:,0]
                    ends = ray[:,1]
                    segments = np.stack((starts, ends), axis=1).reshape(-1, 3)
                    lines = scene.visuals.Line(pos=segments, connect='segments', color='yellow')
                    view.add(lines)


                view.camera = 'turntable'
                canvas.app.run()
            elif plot3dMethod == 2:
                print("Launching PyVista ray plot...")
                print(pv.Report())
                pv.set_plot_theme('dark')
                cloud = pv.PolyData(xyz)
                plotter = pv.Plotter()
                plotter.add_points(cloud, render_points_as_spheres=False, point_size=2)
                plotter.add_floor(face='-z', color='brown', opacity=0.5)

                for ray in rays:
                    starts = ray[:,0]
                    ends = ray[:,1]
                    segments = np.stack((starts, ends), axis=1).reshape(-1, 3)
                    line = pv.Line(segments[0], segments[1])
                    for i in range(1, len(segments)//2):
                        line += pv.Line(segments[i*2], segments[i*2+1])
                    plotter.add_mesh(line, color='yellow')

                plotter.show()

    # Number of Rays convergence testing
    if run_convergence:
        # TODO: This could be parallelized for each seed value
        # Need to create and dataframe structure and update plotting to show range of results
        seeds = [random.randrange(1,500, 1) for i in range(5)]
        #nrays = [1.e4, 2.5e4, 5.e4, 7.5e4, 1.e5, 2.5e5, 5.e5, 7.5e5, 1.e6, 2.5e6, 5.e6, 7.5e6, 1.e7]    # NOTE: greater than 1.e7 runs into memory issues
        #nrays = [1.e4, 5.e4, 1.e5, 5.e5, 1.e6, 5.e6, 1.e7]
        nrays = [1.e4, 1.e5, 1.e6, 1.e7]

        fieldArea = sum(R['st_field_info']['enabled'])  * field.helio_size**2
        seed_results = dict()
        for seed in seeds:
            # Using dataframes instead of lists
            # output = pd.DataFrame()
            # df_dictionary = pd.DataFrame([dictionary])
            # output = pd.concat([output, df_dictionary], ignore_index=True)
            # print(output.head())

            listRes = list()
            for nray in nrays:
                df, ppr, nhit = simulate_soltrace(PT, dni = field.flux_dni, nray = nray , seed = seed, nthreads = 10)
                listRes.append(process_soltrace_results(PT, fieldArea))

            # Merge results into a dictionary
            dictRes = {}
            for key in listRes[0].keys():
                dictRes[key] = [res[key] for res in listRes]

            seed_results[seed] = dictRes

        first_key = True
        for _, resDict in seed_results.items():
            if first_key:
                maxValues = copy.deepcopy(resDict)
                minValues = copy.deepcopy(resDict)
                avgValues = copy.deepcopy(resDict)
                first_key = False
                continue

            for key, metricList in resDict.items():
                for idx, value in enumerate(metricList):
                    maxValues[key][idx] = max(maxValues[key][idx], value)
                    minValues[key][idx] = min(minValues[key][idx], value)
                    avgValues[key][idx] += value

        for key, metricList in avgValues.items():
            for idx, value in enumerate(metricList):
                avgValues[key][idx] /= len(seed_results)

        seed_results['max'] = maxValues
        seed_results['min'] = minValues
        seed_results['avg'] = avgValues

        fig_names = {'Absorbed power (kW)': "abs_power_converge.png",
                   'Field efficiency (%)': "field_eff_converge.png",
                   'Max flux (kW/m^2)': "max_flux_converge.png",
                   'Average flux (kW/m^2)': "mean_flux_converge.png",
                   'Peak concentration ratio (-)': "peak_CR_converge.png",
                   'Average concentration ratio (-)': "avg_CR_converge.png"}

        for key, _ in seed_results['avg'].items():
            plt.figure()
            # Plot results from each seed
            for seed in seed_results.keys():
                if seed not in ['min', 'max', 'avg']:
                    plt.plot(nrays, seed_results[seed][key], '--b', linewidth = 1.0, alpha = 0.75)
            plt.fill_between(nrays, seed_results['min'][key], seed_results['max'][key], alpha = 0.5)
            plt.plot(nrays, seed_results['avg'][key], 'k', linewidth = 3.0)
            plt.plot(nrays, seed_results['min'][key], '--r', linewidth = 2.0)
            plt.plot(nrays, seed_results['max'][key], '--r', linewidth = 2.0)
            plt.xscale('log')
            plt.xlabel("Number of rays")
            plt.ylabel(key)
            plt.tight_layout()
            plt.savefig('convergence_figures/' + fig_names[key], dpi=300)

    # Time-series simulation
    if run_simulation:
        run_sim_in_parallel = False

        tstart = time.time()
        R['sim_data']['abs_power'] = np.nan         # [kW] Power absorbed
        R['sim_data']['max_flux'] = np.nan          # [kW/m^2] maximum flux on receiver
        R['sim_data']['mean_flux'] = np.nan         # [kW/m^2] mean flux on receiver
        R['sim_data']['field_eff'] = np.nan         # [%] Field Efficiency
        field_area = sum(R['st_field_info']['enabled']) * field.helio_size**2

        # Series
        if not run_sim_in_parallel:
            for idx, row in R['sim_data'].iterrows():
                (abs_power, field_eff, max_flux, mean_flux) = simulate_time_step(PT, R['sp_detailed_field'], field_area, row)
                R['sim_data'].at[idx, 'abs_power'] = abs_power
                R['sim_data'].at[idx, 'field_eff'] = field_eff
                R['sim_data'].at[idx, 'max_flux'] = max_flux
                R['sim_data'].at[idx, 'mean_flux'] = mean_flux
        else:
            # Parallelization
            with Pool() as pool:
                results = pool.starmap(simulate_time_step, [(PT, R['sp_detailed_field'], field_area, row) for _, row in R['sim_data'].iterrows()])
            abs_power, field_eff, max_flux, mean_flux = zip(*results)
            R['sim_data']['abs_power'] = abs_power      # [kW] Power absorbed
            R['sim_data']['field_eff'] = field_eff      # [%] Field Efficiency
            R['sim_data']['max_flux'] = max_flux        # [kW/m^2] maximum flux on receiver
            R['sim_data']['mean_flux'] = mean_flux      # [kW/m^2] mean flux on receiver

        print("\nSimulation complete. Total simulation time {:.2f} seconds.".format(time.time()-tstart))
        print("sum of absorbed energy: {:,.2f} kWh".format(sum(R['sim_data']['abs_power'])))


        assert abs(sum(R['sim_data']['abs_power']) - 1420033.8) < 1.0
            #PT.plot_flux(PT.stages[-1].elements[-1], nx=15, ny=15, levels=15)
