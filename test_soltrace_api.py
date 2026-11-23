import os
import tempfile
import numpy as np
import pandas as pd
from soltrace_api import SolTrace, Point, installed_runners


#--- Three mirrors, a curtain inside a cavity and two quadrilateral shield panels in front: the real scene in miniature
def build_scene(runner):
    PT = SolTrace(runner = runner)
    mirror = PT.add_optic('mirror')
    mirror.front.reflectivity = 0.885
    mirror.front.slope_error = 1.2
    mirror.front.spec_error = 0.05
    wall = PT.add_optic('wall')
    wall.front.reflectivity = 0.8
    wall.front.dist_type = 'g'                                   # 'f' (diffuse) runs on the GPU only; the SolTrace 4 CPU runner rejects it
    curtain_optic = PT.add_optic('curtain')
    curtain_optic.front.reflectivity = 0.15
    sun = PT.add_sun()
    sun.position = Point(0.3, -0.8, 100.0)
    target = Point(0.0, 0.0, 12.0)
    field = PT.add_stage()
    for x, y in [(-8.0, -20.0), (0.0, -24.0), (8.0, -20.0)]:
        facet = field.add_element()
        facet.optic = mirror
        facet.position = Point(x, y, 0.0)
        aim_direction = ((target - facet.position).unitize() + sun.position.unitize()) / 2.0
        facet.aim = facet.position + aim_direction * 100.0
        facet.zrot = PT.util_calc_zrot_azel(aim_direction)
        facet.aperture_rectangle(2.0, 2.0)
        facet.surface_parabolic(25.0, 25.0)
    shield = PT.add_stage()
    shield.is_tracethrough = True
    for sign in (-1.0, 1.0):
        panel = shield.add_element()
        panel.optic = wall
        panel.position = Point(sign * 1.0, -1.0, 12.0)
        panel.aim = Point(sign * 1.0 - sign * 10.0, -1.0, 12.0)
        panel.zrot = 0.0
        panel.aperture_quadrilateral(-0.5, -1.0, 0.5, -1.0, 0.7, 1.0, -0.7, 1.0)
        panel.surface_flat()
    receiver = PT.add_stage()
    receiver.is_multihit = True
    curtain = receiver.add_element()
    curtain.optic = curtain_optic
    curtain.position = Point(0.0, 0.5, 12.0)
    curtain.aim = Point(0.0, -100.0, 12.0)
    curtain.zrot = 0.0
    curtain.aperture_rectangle(2.0, 2.0)
    curtain.surface_flat()
    PT.num_ray_hits = 20000
    PT.max_rays_traced = PT.num_ray_hits * 100
    PT.is_sunshape = True
    PT.is_surface_errors = True
    PT.dni = 950.0
    return PT, field, shield, receiver, curtain


#--- The engine writes the event of every row (ABSORB, REFLECT, ...) to CSV; our sign convention must agree with it row by row
def check_against_engine_events(PT):
    path = os.path.join(tempfile.gettempdir(), 'soltrace_api_check.csv')
    PT.stapi.result.csv.dump(path)
    engine = pd.read_csv(path)
    hits = engine[~engine['Event'].isin(['CREATE', 'EXIT'])].reset_index(drop = True)
    ours = PT.raydata
    assert len(ours) == len(hits), (len(ours), len(hits))
    assert np.allclose(ours['loc_x'].to_numpy(), hits['Pos X'].to_numpy())
    assert np.array_equal(ours['number'].to_numpy(), hits['Ray Number'].to_numpy())
    absorbed_ours = ours['element'].to_numpy() < 0
    absorbed_engine = (hits['Event'] == 'ABSORB').to_numpy()
    assert np.array_equal(absorbed_ours, absorbed_engine), 'absorbed flag differs from the engine events'
    print('   rows %d, absorbed %d, reflected %d: matches the engine event log' % (len(ours), absorbed_ours.sum(), (~absorbed_ours).sum()))


def run_checks(runner):
    print('runner', runner)
    PT, field, shield, receiver, curtain = build_scene(runner)
    PT.run(123, True, 1)
    rays = PT.raydata
    assert list(rays.columns) == ['loc_x', 'loc_y', 'loc_z', 'cos_x', 'cos_y', 'cos_z', 'element', 'stage', 'number']
    assert set(rays['stage'].unique()) <= {1, 2, 3}
    assert rays['element'].abs().max() <= 3 and rays['element'].min() >= -3
    on_curtain = rays[(rays['stage'] == receiver.id + 1) & (rays['element'].abs() == curtain.id + 1)]
    absorbed = on_curtain[on_curtain['element'] < 0]
    reflected_from_field = rays[(rays['stage'] == 1) & (rays['element'] > 0)]
    print('   sun box %.1f m2, %d sun rays, %.3f W per ray' % (PT.sunstats['area'], PT.sunstats['nsunrays'], PT.powerperray))
    print('   field reflections %d, curtain hits %d, absorbed %d (%.1f%% of hits, optic says 85%%)' % (
        len(reflected_from_field), len(on_curtain), len(absorbed), 100.0 * len(absorbed) / max(len(on_curtain), 1)))
    print('   curtain power %.2f kW' % (len(absorbed) * PT.powerperray / 1e3))
    check_against_engine_events(PT)
    PT.run(123, True, 1)
    assert len(PT.raydata) == len(rays), 'same seed should give the same trace'
    PT.free()
    return rays


if __name__ == '__main__':
    runners = installed_runners()
    print('installed runners', runners)
    reference = run_checks('native')
    if runners.get('OPTIX'):
        gpu = run_checks('optix')
        by_element = ['stage', 'element']
        print('CPU vs GPU hit counts per element:')
        print(pd.concat([reference.groupby(by_element).size().rename('cpu'), gpu.groupby(by_element).size().rename('gpu')], axis = 1).fillna(0).astype(int))
