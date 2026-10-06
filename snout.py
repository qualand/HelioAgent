# Four flat opaque panels in on trace-through stage between the field and the receiver.
# Rays that hit a panel are lost; rays that miss all panels continue to the receiver. 
#   depth [m]        aperture plane to SNOUT front plane
#   horiz [deg]      full angle between the east and west panels
#   top, bot [deg]   tilt of the top and bottom panels from the aperture normal, + = down toward the field

import numpy as np
from api.pysoltrace import Point

# Returns the four global corners of each panel and one point inside the tunnel. 
def snout_corners(sp_params, snout):
    aperture_width = sp_params['receiver.0.rec_width']
    aperture_height = sp_params['receiver.0.rec_height']
    depth = snout['depth']
    azimuth = np.radians(sp_params['receiver.0.rec_azimuth'])
    elevation = np.radians(sp_params['receiver.0.rec_elevation'])
    
    # Aperture frame
    aperture_center = np.array([0.0, 0.0, sp_params['solarfield.0.tht']])
    unit_sideways = np.array([np.cos(azimuth), -np.sin(azimuth), 0.0])
    unit_up = np.array([-np.sin(elevation) * np.sin(azimuth), -np.sin(elevation) * np.cos(azimuth), np.cos(elevation)])
    unit_normal = np.array([np.cos(elevation) * np.sin(azimuth), np.cos(elevation) * np.cos(azimuth), np.sin(elevation)])
   
    # Front plane dimensions
    side_flare = depth * np.tan(np.radians(snout['horiz']) / 2)
    top_front_height = aperture_height / 2 - depth * np.tan(np.radians(snout['top']))
    bottom_front_height = -aperture_height / 2 - depth * np.tan(np.radians(snout['bot']))
    
    # Global point at the give distances sideways, up and out toward the field from the aperture center
    def corner(sideways, up, out):
        return aperture_center + sideways * unit_sideways + up * unit_up + out * unit_normal
    
    half_width, half_height = aperture_width / 2, aperture_height / 2
    
    # Each panel: two corners on the aperture edge (out = 0), two on the front plane (out = depth)
    panel_corners = {
        'top': [corner(-half_width, half_height, 0),
                corner(half_width, half_height, 0),
                corner(half_width + side_flare, top_front_height, depth),
                corner(-half_width - side_flare, top_front_height, depth)],
        'bottom': [corner(-half_width, -half_height, 0),
                   corner(half_width, -half_height, 0),
                   corner(half_width + side_flare, bottom_front_height, depth),
                   corner(-half_width - side_flare, bottom_front_height, depth)],
        'east': [corner(half_width, -half_height, 0),
                 corner(half_width, half_height, 0),
                 corner(half_width + side_flare, top_front_height, depth),
                 corner(half_width + side_flare, bottom_front_height, depth)],
        'west': [corner(-half_width, -half_height, 0),
                 corner(-half_width, half_height, 0),
                 corner(-half_width - side_flare, top_front_height, depth),
                 corner(-half_width - side_flare, bottom_front_height, depth)],
    }
    
    # Point on the tunnel axis, halfway between the aperture and the flot plane.
    # Used to orient each panel's normal so it points into the tunnel.
    tunnel_axis_point = aperture_center + 0.5 * depth * unit_normal
    return panel_corners, tunnel_axis_point

# Rotation matrix from global (SolTrace "refence") to the element's local frame
def ref_to_local(aim_direction):
    alpha = np.arctan2(aim_direction[0], aim_direction[2])
    beta = np.arcsin(aim_direction[1])
    return np.array([[np.cos(alpha), 0.0, -np.sin(alpha)],
                     [-np.sin(alpha) * np.sin(beta), np.cos(beta), -np.cos(alpha) * np.sin(beta)],
                     [np.sin(alpha) * np.cos(beta), np.sin(beta), np.cos(alpha) * np.cos(beta)]])

# Add the SNOUT stage with its four panels to the PySolTrace scene PT
def add_snout(PT, sp_params, snout, optic):
    snout_stage = PT.add_stage()
    snout_stage.name = 'SNOUT'
    snout_stage.is_tracethrough = True # rays that miss every panel go on to the next stage (the receiver)
                                    # stage position/aim keep their defaults, so panels are given in global coordinates
                                    
    panel_corners, tunnel_axis_point = snout_corners(sp_params, snout)
    panels = []     # one dict per panel, kept for binning the flux on it later
    for panel_name, corners in panel_corners.items():
        corners = np.array(corners)
        
        # Element origin at the panel centroid
        centroid = corners.mean(axis=0)
    
        # Panel normal from the two edges that share corner 0, flipped so it points into the tunnel.
        # SolTrace uses the aim vector (centroid -> centroid + normal) as the element's local z axis.
        panel_normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
        panel_normal /= np.linalg.norm(panel_normal)
        if panel_normal @ (tunnel_axis_point - centroid) < 0:
            panel_normal = -panel_normal

        # Corners in the element's local x-y plane (local z is zero on a flat panel).
        # SolTrace's quadrilateral aperture ('q') takes the four corners in these local coordinates.
        rotation = ref_to_local(panel_normal)
        local_xy = (corners - centroid) @ rotation[:2].T

        # Make the corner order counter-clockwise in local x-y (shoelace sign), as copylot.py lists them
        shoelace = np.sum(local_xy[:, 0] * np.roll(local_xy[:, 1], -1) - np.roll(local_xy[:, 0], -1) * local_xy[:, 1])
        if shoelace < 0:
            local_xy = local_xy[::-1]

        panel = snout_stage.add_element()
        panel.position = Point(*centroid.tolist())                  # global, because the stage frame is the global frame
        panel.aim = Point(*(centroid + panel_normal).tolist())      # any point along the normal works as the aim
        panel.zrot = 0.0                                            # ref_to_local() above assumes zrot = 0
        panel.aperture_quadrilateral(*local_xy.ravel().tolist())
        panel.surface_flat()
        panel.optic = optic
        panel.comment = 'SNOUT ' + panel_name
        panels.append(dict(name=panel_name, element=panel, centroid=centroid, rotation=rotation, local_xy=local_xy))
    return snout_stage, panels


# True for the points (local x-y) that lie inside a convex quadrilateral given counter-clockwise
def inside_quadrilateral(corners_xy, points_xy):
    inside = np.ones(len(points_xy), dtype=bool)
    for start_index in range(4):
        start, end = corners_xy[start_index], corners_xy[(start_index + 1) % 4]
        cross = (end[0] - start[0]) * (points_xy[:, 1] - start[1]) - (end[1] - start[1]) * (points_xy[:, 0] - start[0])
        inside &= cross >= -1e-12
    return inside


# Incident flux map [kW/m2] on one SNOUT panel from the last trace, on an nbins x nbins grid.
# Same idea as PT.bin_rays(): count the ray hits in each cell and divide by the cell area.
# The panel is a trapezoid, so cells on its slanted edges are only partly inside the panel.
# The area of each cell inside the panel comes from a fine sub-grid; cells less than a quarter
# inside are set to zero so a few rays on a sliver of a cell cannot fake a hot spot.
def panel_flux_map(PT, panel, nbins=10, subsamples=8):
    local_xy = panel['local_xy']
    x_edges = np.linspace(local_xy[:, 0].min(), local_xy[:, 0].max(), nbins + 1)
    y_edges = np.linspace(local_xy[:, 1].min(), local_xy[:, 1].max(), nbins + 1)
    cell_width, cell_height = x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]

    # Fraction of each cell that lies inside the panel
    sub_x = x_edges[0] + (np.arange(nbins * subsamples) + 0.5) * cell_width / subsamples
    sub_y = y_edges[0] + (np.arange(nbins * subsamples) + 0.5) * cell_height / subsamples
    grid_x, grid_y = np.meshgrid(sub_x, sub_y)
    inside = inside_quadrilateral(local_xy, np.c_[grid_x.ravel(), grid_y.ravel()]).reshape(grid_x.shape)
    fraction_inside = inside.reshape(nbins, subsamples, nbins, subsamples).mean(axis=(1, 3))
    cell_area = fraction_inside * cell_width * cell_height

    # Ray hits on this panel, moved into the panel's local x-y frame (ray positions are in stage
    # coordinates, and the SNOUT stage frame is the global frame)
    rays = PT.raydata
    element = panel['element']
    hits = rays[(rays['stage'] == element.stage_id + 1) & (rays['element'].abs() == element.id + 1)]
    hit_xy = (hits[['loc_x', 'loc_y', 'loc_z']].to_numpy() - panel['centroid']) @ panel['rotation'][:2].T
    counts, _, _ = np.histogram2d(hit_xy[:, 1], hit_xy[:, 0], bins=[y_edges, x_edges])

    flux = np.where(fraction_inside > 0.25, counts * PT.powerperray / np.maximum(cell_area, 1e-12), 0.0)
    return flux / 1e3


# Absorbed power [kW] and peak incident flux [kW/m2] on each panel from the last trace
def snout_results(PT, panels):
    rays = PT.raydata
    results = {}
    for panel in panels:
        element = panel['element']
        absorbed = rays[(rays['stage'] == element.stage_id + 1) & (rays['element'] == -(element.id + 1))]
        results['SNOUT ' + panel['name'] + ' absorbed (kW)'] = len(absorbed) * PT.powerperray / 1e3
        results['SNOUT ' + panel['name'] + ' peak flux (kW/m^2)'] = panel_flux_map(PT, panel).max()
    return results