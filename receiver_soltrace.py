# Falling-particle receiver for Soltrace -> a black counting plane where the particle curtain hangs,
# behind the aperture, and a black tower face around the aperture so only rays through the aperture reach the curtain

import numpy as np
from api.pysoltrace import Point

def receiver_frame(sp_params):
    azimuth = np.radians(sp_params['receiver.0.rec_azimuth'])
    aperture_center = np.array([0.0, 0.0, sp_params['solarfield.0.tht']])
    unit_sideways = np.array([np.cos(azimuth), -np.sin(azimuth), 0.0])
    unit_up = np.array([0.0, 0.0, 1.0])
    unit_normal = np.array([np.sin(azimuth), np.cos(azimuth), 0.0])
    return aperture_center, unit_sideways, unit_up, unit_normal

# Flat rectangle with its local x sideways and local y vertical, so PT.bin_rays() works on it
def add_rectangle(PT, stage, center, width, height, unit_normal, optic, comment):
    element = stage.add_element()
    element.position = Point(*center.tolist())
    element.aim = Point(*(center + unit_normal).tolist())
    element.zrot = PT.util_calc_zrot_azel(unit_normal.tolist())
    element.aperture_rectangle(width, height)
    element.surface_flat()
    element.optic = optic
    element.comment = comment
    return element

def add_particle_receiver(PT, sp_params, cavity, optic):
    aperture_center, unit_sideways, unit_up, unit_normal = receiver_frame(sp_params)
    aperture_width = sp_params['receiver.0.rec_width']
    aperture_height = sp_params['receiver.0.rec_height']
    curtain_width = cavity['width_ratio'] * aperture_width
    curtain_height = cavity['height_ratio'] * aperture_height
    half_width, half_height = aperture_width / 2, aperture_height / 2
    face = cavity.get('tower_face', 10)
    
    def point(sideways, up, out):
        return aperture_center + sideways * unit_sideways + up * unit_up + out * unit_normal
    
    stage = PT.add_stage()
    stage.name = 'Falling-particle RECEIVER'
    stage.is_multihit = False
    
    # Element 0: particle curtain. Bottom of the curtain level with the bottom of the aperture
    curtain_center = point(0.0, -half_height + curtain_height / 2, -cavity['curtain_depth'])
    curtain_element = add_rectangle(PT, stage, curtain_center, curtain_width, curtain_height, unit_normal, optic, 'Particle curtain')
    
    # Tower face: four black strips in the aperture plane, leaving the aperture open
    strips = {
        'above' : (0.0, (half_height + face) / 2, 2 * face, face - half_height),
        'below' : (0.0, -(half_height + face) / 2, 2 * face, face - half_height),
        'east' : ((half_width + face) / 2, 0.0, face - half_width, aperture_height),
        'west' : (-(half_width + face) / 2, 0.0, face - half_width, aperture_height),
        
    }
    
    for name, (sideways, up, width, height) in strips.items():
        add_rectangle(PT, stage, point(sideways, up, 0.0), width, height, unit_normal, optic, 'Tower face ' + name)
        
    curtain = dict(element=curtain_element, center=curtain_center, unit_sideways=unit_sideways, unit_up=unit_up,
                   width=curtain_width, height=curtain_height)
    return stage, curtain

# Incident flux on the curtain [kW/m2], shape (n_height, n_width), row 0 = top of the curtain, 
# column 0 = west side seen from the field. This is the input the thermal model expects.

def curtain_flux(PT, curtain, n_height=25, n_width=25):
    rays = PT.raydata
    element = curtain['element']
    hits = rays[(rays['stage'] == element.stage_id + 1) & (rays['element'].abs() == element.id + 1)]
    offsets = hits[['loc_x', 'loc_y', 'loc_z' ]].to_numpy() - curtain['center']
    up = offsets @ curtain['unit_sideways']
    sideways = offsets @ curtain['unit_sideways']
    up_edges = np.linspace(-curtain['height'] / 2, curtain['height'] / 2, n_height + 1)
    side_edges = np.linspace(-curtain['width'] / 2, curtain['width'] / 2, n_width +1)
    counts, _, _ = np.histogram2d(up, sideways, bins=[up_edges, side_edges])
    cell_area = (up_edges[1] - up_edges[0]) * (side_edges[1] - side_edges[0])
    flux = counts * PT.powerperray / cell_area / 1e3
    return flux [::-1]
                 

        