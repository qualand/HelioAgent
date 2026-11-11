import numpy as np
from api.pysoltrace import Point


#--- The four corners of each SNOUT panel in global coordinates, and a point on the tunnel axis to orient the normals
#    snout: depth (m) from the aperture plane to the front plane; horiz (deg) full angle between the east and west panels;
#    top, bot (deg) tilt of the top and bottom panels from the aperture normal, + = down toward the field
def snout_corners(sp_params, snout):
    aperture_width = sp_params['receiver.0.rec_width']
    aperture_height = sp_params['receiver.0.rec_height']
    depth = snout['depth']
    azimuth = np.radians(sp_params['receiver.0.rec_azimuth'])
    elevation = np.radians(sp_params['receiver.0.rec_elevation'])

    aperture_center = np.array([0.0, 0.0, sp_params['solarfield.0.tht']])                                   # Aperture frame
    unit_sideways = np.array([np.cos(azimuth), -np.sin(azimuth), 0.0])
    unit_up = np.array([-np.sin(elevation) * np.sin(azimuth), -np.sin(elevation) * np.cos(azimuth), np.cos(elevation)])
    unit_normal = np.array([np.cos(elevation) * np.sin(azimuth), np.cos(elevation) * np.cos(azimuth), np.sin(elevation)])

    side_flare = depth * np.tan(np.radians(snout['horiz']) / 2)                                            # Front plane dimensions
    top_front_height = aperture_height / 2 - depth * np.tan(np.radians(snout['top']))
    bottom_front_height = -aperture_height / 2 - depth * np.tan(np.radians(snout['bot']))

    def corner(sideways, up, out):
        return aperture_center + sideways * unit_sideways + up * unit_up + out * unit_normal              # Global point from the aperture center

    half_width, half_height = aperture_width / 2, aperture_height / 2
    panel_corners = {                                                       # Two corners on the aperture edge (out = 0), two on the front plane (out = depth)
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
    tunnel_axis_point = aperture_center + 0.5 * depth * unit_normal        # Halfway between the aperture and the front plane
    return panel_corners, tunnel_axis_point


#--- Rotation matrix from global (SolTrace "reference") to the element's local frame
def ref_to_local(aim_direction):
    alpha = np.arctan2(aim_direction[0], aim_direction[2])
    beta = np.arcsin(aim_direction[1])
    return np.array([[np.cos(alpha), 0.0, -np.sin(alpha)],
                     [-np.sin(alpha) * np.sin(beta), np.cos(beta), -np.cos(alpha) * np.sin(beta)],
                     [np.sin(alpha) * np.cos(beta), np.sin(beta), np.cos(alpha) * np.cos(beta)]])


#--- Add the SNOUT to the SolTrace scene: one trace-through stage with four flat panels between the field and the receiver.
#    Rays that hit a panel are 20 % absorbed and 80 % scattered diffusely (Mills et al. 2024); scattered rays can hit another panel or go on.
#    Returns the stage and one dict per panel, kept for binning the hits on it later.
def add_snout(PT, sp_params, snout, optic):
    snout_stage = PT.add_stage()
    snout_stage.name = 'SNOUT'
    snout_stage.is_tracethrough = True                                      # Rays that miss every panel go on to the next stage (the receiver);
                                                                            # stage position and aim keep their defaults, so panels are in global coordinates
    panel_corners, tunnel_axis_point = snout_corners(sp_params, snout)
    panels = []
    for panel_name, corners in panel_corners.items():
        corners = np.array(corners)
        centroid = corners.mean(axis = 0)                                   # Element origin

        panel_normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])   # From the two edges that share corner 0
        panel_normal /= np.linalg.norm(panel_normal)
        if panel_normal @ (tunnel_axis_point - centroid) < 0:
            panel_normal = -panel_normal                                    # Flipped to point into the tunnel; SolTrace's local z axis

        rotation = ref_to_local(panel_normal)
        local_xy = (corners - centroid) @ rotation[:2].T                    # Corners in the element's local x-y plane (local z = 0 on a flat panel)
        shoelace = np.sum(local_xy[:, 0] * np.roll(local_xy[:, 1], -1) - np.roll(local_xy[:, 0], -1) * local_xy[:, 1])
        if shoelace < 0:
            local_xy = local_xy[::-1]                                       # Counter-clockwise, as copylot.py lists them

        panel = snout_stage.add_element()
        panel.position = Point(*centroid.tolist())                          # Global, because the stage frame is the global frame
        panel.aim = Point(*(centroid + panel_normal).tolist())              # Any point along the normal works as the aim
        panel.zrot = 0.0                                                    # ref_to_local() assumes zrot = 0
        panel.aperture_quadrilateral(*local_xy.ravel().tolist())            # SolTrace's quadrilateral aperture takes the four local corners
        panel.surface_flat()
        panel.optic = optic
        panel.comment = 'SNOUT ' + panel_name

        hinge = corners[1] - corners[0]                                     # Grid frame for the thermal cells: x along the aperture edge, y toward the front.
        hinge /= np.linalg.norm(hinge)                                      # SolTrace's local x is horizontal in global coordinates, 19 deg off the edge
        across = 0.5 * (corners[2] + corners[3]) - 0.5 * (corners[0] + corners[1])   # on the top and bottom walls; a grid in that frame comes out slanted
        across -= (across @ hinge) * hinge
        across /= np.linalg.norm(across)
        grid_axes = np.array([hinge, across])
        grid_xy = (corners - centroid) @ grid_axes.T
        if np.sum(grid_xy[:, 0] * np.roll(grid_xy[:, 1], -1) - np.roll(grid_xy[:, 0], -1) * grid_xy[:, 1]) < 0:
            grid_xy = grid_xy[::-1]                                         # Counter-clockwise, as clip_cell and inside_quadrilateral expect
        panels.append(dict(name = panel_name, element = panel, centroid = centroid, rotation = rotation, local_xy = local_xy,
                           grid_axes = grid_axes, grid_xy = grid_xy))
    return snout_stage, panels


#--- True for the points (local x-y) inside a convex quadrilateral given counter-clockwise
def inside_quadrilateral(corners_xy, points_xy):
    inside = np.ones(len(points_xy), dtype = bool)
    for start_index in range(4):
        start, end = corners_xy[start_index], corners_xy[(start_index + 1) % 4]
        cross = (end[0] - start[0]) * (points_xy[:, 1] - start[1]) - (end[1] - start[1]) * (points_xy[:, 0] - start[0])
        inside &= cross >= -1e-12
    return inside


#--- Uniform nbins x nbins grid over the bounding box of a panel, in the grid frame; for plots (the thermal model uses its own mesh, SnoutThermal.panel_edges)
def panel_bin_edges(panel, nbins):
    grid_xy = panel['grid_xy']
    x_edges = np.linspace(grid_xy[:, 0].min(), grid_xy[:, 0].max(), nbins + 1)
    y_edges = np.linspace(grid_xy[:, 1].min(), grid_xy[:, 1].max(), nbins + 1)
    return x_edges, y_edges


#--- Incident ray power (W) in each bin of one panel from the last trace: hit counts times power per ray, no area involved.
#    Every hit counts, reflected or absorbed. The bin edges come from the thermal model, which divides by the true area of its cells.
def panel_power_map(PT, panel, x_edges, y_edges):
    rays = PT.raydata
    element = panel['element']
    hits = rays[(rays['stage'] == element.stage_id + 1) & (rays['element'].abs() == element.id + 1)]   # Rays that touched this panel
    hit_xy = (hits[['loc_x', 'loc_y', 'loc_z']].to_numpy() - panel['centroid']) @ panel['grid_axes'].T   # Hit points in the grid frame
    counts, _, _ = np.histogram2d(hit_xy[:, 1], hit_xy[:, 0], bins = [y_edges, x_edges])
    return counts * PT.powerperray


#--- Incident flux map (kW/m2) on one panel on any grid, for plots. Bins on the slanted sides are only partly inside the panel:
#    their inside area comes from a fine sub-grid, and bins less than a quarter inside are zeroed so a few rays on a sliver cannot fake a hot spot
def panel_flux_map(PT, panel, x_edges, y_edges, subsamples = 8):
    local_xy = panel['grid_xy']
    n_cols, n_rows = len(x_edges) - 1, len(y_edges) - 1
    offsets = (np.arange(subsamples) + 0.5) / subsamples
    sub_x = (x_edges[:-1, None] + np.diff(x_edges)[:, None] * offsets).ravel()
    sub_y = (y_edges[:-1, None] + np.diff(y_edges)[:, None] * offsets).ravel()
    grid_x, grid_y = np.meshgrid(sub_x, sub_y)
    inside = inside_quadrilateral(local_xy, np.c_[grid_x.ravel(), grid_y.ravel()]).reshape(grid_x.shape)
    fraction_inside = inside.reshape(n_rows, subsamples, n_cols, subsamples).mean(axis = (1, 3))
    bin_area = fraction_inside * np.outer(np.diff(y_edges), np.diff(x_edges))
    power = panel_power_map(PT, panel, x_edges, y_edges)
    flux = np.where(fraction_inside > 0.25, power / np.maximum(bin_area, 1e-12), 0.0)
    return flux / 1e3


#--- Absorbed power (kW) on each panel from the last trace; the peak incident flux per panel is reported by SnoutThermal.step on its own cells
def snout_results(PT, panels):
    rays = PT.raydata
    results = {}
    for panel in panels:
        element = panel['element']
        absorbed = rays[(rays['stage'] == element.stage_id + 1) & (rays['element'] == -(element.id + 1))]   # Negative element id = absorbed there
        results['SNOUT ' + panel['name'] + ' absorbed (kW)'] = len(absorbed) * PT.powerperray / 1e3
    return results