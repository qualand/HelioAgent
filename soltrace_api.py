import ctypes
import random
import numpy as np
import pandas as pd
from pysoltrace import Point, dot_h, api as STAPIv2, math_utils, soltrace_structs as structs
from pysoltrace.legacy import legacy

RUNNER = 'optix'                                                      # The engine every trace uses: 'optix' (GPU), 'embree' or 'native' (CPU). Change it here only.
RUNNERS = dict(native = 0, optix = 1, embree = 2)
ELEMENT_NULL = np.uint64(2**64 - 5)                                  # The engine's "no element" marker (-5) seen as unsigned


#--- The agent's one door to SolTrace. Same object model as the old api/pysoltrace.py (optics, sun, stages, elements, Point,
#    util_* functions) and the same ray table convention: element = +(local id + 1) reflected, -(local id + 1) absorbed,
#    stage = stage id + 1, one row per hit. The engine underneath is stapi_v2 (SolTrace 4), on the CPU or on the GPU.
class SolTrace(legacy):
    def __init__(self, runner = RUNNER, **kwargs):
        kwargs.setdefault('optics', [])                               # The legacy class shares one default list between instances
        kwargs.setdefault('stages', [])
        super().__init__(testing = True, **kwargs)                   # testing silences the engine's message callback
        self.runner = RUNNERS[runner]
        self.stapi = None
        self.local_id = None
        self.stage_number = None

    def run(self, seed = -1, as_power_tower = False, nthread = 1, thread_id = 0, no_callback = False, **unused):
        if self.stapi is not None:
            self.stapi.free()
        self.stapi = STAPIv2(testing = True)
        if seed < 0:
            seed = random.randint(1, int(1e9))
        self._create(as_power_tower, seed)
        self._trace(seed, nthread)
        self._collect()
        return 1

    def free(self):
        if self.stapi is not None:
            self.stapi.free()
            self.stapi = None

    #--- Scene into the engine. Stages are gone in SolTrace 4: every element goes into one list and is traced in one pass,
    #    so the element id the engine returns is global. local_id and stage_number translate it back.
    def _create(self, as_power_tower, seed):
        stapi = self.stapi
        stapi.legacy.sim_params(int(self.num_ray_hits), int(self.max_rays_traced), as_power_tower)
        stapi.parameters.errors(self.is_sunshape, self.is_surface_errors)
        pdll, pcxt, check = stapi.sneak()
        pdll.st_sim_seed.argtypes = [ctypes.c_void_p, ctypes.c_double]
        check(pdll.st_sim_seed(pcxt, float(seed)))                    # The GPU runner reads this seed; the CPU runner reads the ones given at setup
        self.sun.Create(stapi, _do = True)
        for optic in self.optics:
            optic.Create(stapi, _do = True)
        local_id, stage_number = [-1], [0]
        for element in self._elements:
            local_id.append(element.id)
            stage_number.append(0)
            self._add_element(element, None)
        for stage in self.stages:
            unstager = math_utils.get_unstager(stage.position, stage.aim, stage.zrot)
            for element in stage.elements:
                global_id = self._add_element(element, unstager)
                assert global_id == len(local_id), 'element ids are expected to be consecutive and 1-based'
                local_id.append(element.id)
                stage_number.append(stage.id + 1)
        self.local_id = np.array(local_id, dtype = np.int64)
        self.stage_number = np.array(stage_number, dtype = np.int64)

    def _add_element(self, element, unstager):
        position = unstager(element.position) if unstager else element.position
        aim = unstager(element.aim) if unstager else element.aim
        element_args = structs.element(list(position), list(aim), element.zrot, element.enabled, element.virtual)
        geometry = structs.geometry(list(element.aperture_params), surface_params_for_engine(element), element.aperture.encode(), element.surface.encode())
        return self.stapi.data.element.add(element_args, element.optic.id, geometry)

    def _trace(self, seed, nthread):
        if self.runner == RUNNERS['optix']:
            self.stapi.runner.setup(self.runner)
        else:
            seeds = [seed + 608 * i for i in range(nthread)]
            self.stapi.runner.setup(self.runner, nthread, seeds)
        self.stapi.runner.run()
        self.stapi.runner.report(dot_h.st_runner_statistics.RAY_RECORDS)

    #--- Ray table out of the engine straight into numpy, then back to the old convention
    def _collect(self):
        stapi = self.stapi
        pdll, pcxt, check = stapi.sneak()
        n = stapi.result.num()
        names = ('loc_x', 'loc_y', 'loc_z', 'cos_x', 'cos_y', 'cos_z')
        columns = {name: np.empty(n, dtype = np.float64) for name in names}
        element = np.empty(n, dtype = np.uint64)
        stage = np.empty(n, dtype = np.uint64)
        number = np.empty(n, dtype = np.uint64)
        as_double = lambda array: array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        as_uint64 = lambda array: array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        args = dot_h.args_results_data(*[as_double(columns[name]) for name in names], as_uint64(element), as_uint64(stage), as_uint64(number))
        check(pdll.st_get_results_data(pcxt, ctypes.byref(args)))
        number = number.astype(np.int64)
        if np.any(number[1:] < number[:-1]):
            order = np.argsort(number, kind = 'stable')               # Rows of one ray must stay together and in order
            element, number = element[order], number[order]
            columns = {name: values[order] for name, values in columns.items()}
        first_of_ray = np.r_[True, number[1:] != number[:-1]]
        last_of_ray = np.r_[number[1:] != number[:-1], True]
        hit = element != ELEMENT_NULL                                 # Drops the CREATE row on the sun plane and the EXIT row of rays that left
        absorbed = hit & last_of_ray                                  # A ray that reflected or passed through always has a later row
        for name in ('cos_x', 'cos_y', 'cos_z'):                      # The engine stores the direction leaving each hit; the old table stored
            columns[name] = np.where(first_of_ray, np.nan, np.r_[np.nan, columns[name][:-1]])   # the direction arriving, which is the previous row's
        global_id = element[hit].astype(np.int64)
        signed = np.where(absorbed[hit], -(self.local_id[global_id] + 1), self.local_id[global_id] + 1)
        self.raydata = pd.DataFrame({**{name: values[hit] for name, values in columns.items()},
                                     'element': signed, 'stage': self.stage_number[global_id], 'number': number[hit]})
        width, height, area, nsunrays = stapi.result.sun_stats()
        if width == 0.0 or height == 0.0:
            width = height = np.sqrt(area)
        self.sunstats = dict(xmin = 0.0, xmax = width, ymin = 0.0, ymax = height, width = width, height = height,
                             area = area, nsunrays = nsunrays, position = self.sun.position)
        self.powerperray = area * self.dni / nsunrays

    def get_ray_dataframe(self):
        return self.raydata

    def get_sun_stats(self):
        return self.sunstats

    #--- Flux map (W/m2) on a flat element, indexed [x bin, y bin] in the element's own frame, as the old bin_rays did.
    #    process_soltrace_results reads its max and mean.
    def bin_rays(self, element, nx = 25, ny = 25, absorbed_only = True):
        assert element.surface == 'f', 'bin_rays is written for flat elements'
        rays = self.raydata
        on_element = (rays['stage'] == element.stage_id + 1) & ((rays['element'] == -(element.id + 1)) if absorbed_only else (rays['element'].abs() == element.id + 1))
        hits = rays.loc[on_element, ['loc_x', 'loc_y', 'loc_z']].to_numpy()
        euler = math_utils.euler_angles(np.array(element.position.as_list()), np.array(element.aim.as_list()), element.zrot)
        to_local = math_utils.euler_transforms(euler)['rreftoloc']
        local = (hits - np.array(element.position.as_list())) @ to_local.T
        width, height = element.aperture_params[0:2]
        x_edges = np.linspace(-width / 2, width / 2, nx + 1)
        y_edges = np.linspace(-height / 2, height / 2, ny + 1)
        counts, _, _ = np.histogram2d(local[:, 0], local[:, 1], bins = [x_edges, y_edges])
        return counts * self.powerperray / ((width / nx) * (height / ny))


#--- The engine takes vertex curvatures c = 1 / (2 f) for a parabola; the legacy wrapper still stores the focal lengths
#    that surface_parabolic(focal_len_x, focal_len_y) receives. Converted here and nowhere else.
def surface_params_for_engine(element):
    params = list(element.surface_params)
    if element.surface == 'p':
        params[0] = 0.0 if params[0] == 0.0 else 0.5 / params[0]
        params[1] = 0.0 if params[1] == 0.0 else 0.5 / params[1]
    return params


def installed_runners():
    return STAPIv2.get_installed_runners()
