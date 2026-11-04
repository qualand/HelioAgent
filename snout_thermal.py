# SNOUT thermal model: wall temperatures from SolTrace
# Same methodology as Janna's falling_particle_receiver.py: gray body, diffuse enclosure
# Solved with radiosities

import numpy as np
from snout_soltrace import snout_corners, inside_quadrilateral
from viewfactor import view_factor, polygon_area, clean_polygon
import os

SIGMA = 5.6704e-8   # Stefan-Boltzmann constant (W/m2/K4)

# RSLE-57 face sheet, back face exposed to outside air

RSLE57 = dict(
    density=2100.0,             # kg/m3
    specific_heat=1000.0,       # J/kg/K                      TODO confirm
    conductivity=0.7,           # W/m/K
    thickness=0.025,            # m, face sheet               TODO confirm
    solar_absorptivity=0.2,     # 1 - 0.8 solar reflectivity (Bill's SNOUT optic)
    ir_emissivity=0.8,          # infrared emissivity         TODO confirm
    max_temp_c=1200.0,          # RSLE-57 use limit, the constraint that matters
)

# Air at the film temperature, fits to Incropera et al. (2007) table A.4 between 250 and 1500 K
AIR_PRANDTL = 0.71
def air_conductivity(t_k):
    return 0.0241 * (t_k / 273.15) ** 0.8            # W/m/K
def air_kinematic_viscosity(t_k):
    return 1.33e-5 * (t_k / 273.15) ** 1.7           # m2/s

# Natural convection coefficient of a flat plate in still air, one value per cell (W/m2/K).
# Film temperature properties, Ra = g beta dT L^3 Pr / nu^2, Nu correlations from Incropera et al. (2007) chapter 9:
#   vertical      : Churchill and Chu (1975), whole Ra range, the east and west walls
#   hot_face_down : 0.27 Ra^1/4, heated plate facing down (McAdams 1954), the top wall seen from inside the SNOUT
#   hot_face_up   : 0.54 Ra^1/4 then 0.15 Ra^1/3 above Ra = 1e7 (McAdams 1954; Lloyd and Moran 1974), the bottom wall
# Tilts are ignored (the bottom wall is 34 deg from horizontal); the air in the SNOUT is taken at ambient temperature.
def natural_convection_h(t_wall_k, t_air_k, length, orientation):
    t_film = 0.5 * (t_wall_k + t_air_k)
    rayleigh = 9.81 * np.abs(t_wall_k - t_air_k) / t_film * length ** 3 * AIR_PRANDTL / air_kinematic_viscosity(t_film) ** 2
    if orientation == 'vertical':
        nusselt = (0.825 + 0.387 * rayleigh ** (1 / 6) / (1 + (0.492 / AIR_PRANDTL) ** (9 / 16)) ** (8 / 27)) ** 2
    elif orientation == 'hot_face_up':
        nusselt = np.where(rayleigh < 1e7, 0.54 * rayleigh ** 0.25, 0.15 * rayleigh ** (1 / 3))
    else:
        nusselt = 0.27 * rayleigh ** 0.25
    return nusselt * air_conductivity(t_film) / length

WALL_ORIENTATION = dict(top='hot_face_down', bottom='hot_face_up', east='vertical', west='vertical')

# Split [lo, hi] into pieces of at most `size`, equal within the interval
def split_edges(lo, hi, size):
    return np.linspace(lo, hi, int(np.ceil((hi - lo) / size - 1e-9)) + 1)


def clip_cell(rect, trapezoid):
    # Cut a grid rectangle down to its part inside the panel trapezoid
    # (Sutherland-Hodgman: clip against each edge of the convex trapezoid in turn)
    def cross2(a, b):
        return a[0] * b[1] - a[1] * b[0]
    output = [np.array(p, dtype=float) for p in rect]
    for k in range(len(trapezoid)):
        a, b = np.array(trapezoid[k]), np.array(trapezoid[(k + 1) % len(trapezoid)])
        edge = b - a
        def inside(p):      # left of the edge, for a counterclockwise trapezoid
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-12
        kept, output = output, []
        for idx in range(len(kept)):
            cur, prev = kept[idx], kept[idx - 1]
            if inside(cur):
                if not inside(prev):   # edge crossed coming in: add the crossing point
                    t = cross2(a - prev, edge) / cross2(cur - prev, edge)
                    output.append(prev + t * (cur - prev))
                output.append(cur)
            elif inside(prev):         # edge crossed going out: add the crossing point
                t = cross2(a - prev, edge) / cross2(cur - prev, edge)
                output.append(prev + t * (cur - prev))
    return clean_polygon(output)


class SnoutThermal:
    # panels: the list returned by add_snout. The mesh is defined HERE and only here: panel_power_map bins the rays
    # on self.panel_edges, so a flux bin and a thermal cell are always the same patch of wall.
    # Two-zone mesh: a band of fine cells (cell_min x cell_min, two rows by default) along the aperture edge, where the spilled flux lands,
    # and the previous coarse n_coarse x n_coarse grid everywhere else (its first row is what the band replaces).
    def __init__(self, sp_params, snout, panels, material=None, cell_min=0.10, band=0.20, n_coarse=4, min_cell_fraction=0.25,
                 natural_convection=True, vf_file='snout_view_factors_analytic_v2.npz'):
        self.material = {**RSLE57, **(material or {})}
        self.panels = panels
        self.cell_min = cell_min                     # m, size of the fine cells touching the aperture
        self.band = band                             # m, depth of the fine band from the aperture edge (band / cell_min rows of fine cells)
        self.n_coarse = n_coarse                     # coarse grid: n_coarse columns over the wall width, n_coarse - 1 rows behind the band
        self.min_cell_fraction = min_cell_fraction   # a clipped cell smaller than this fraction of its bin is merged into its inner neighbour
        self.natural_convection = natural_convection
        self._build_geometry(sp_params, snout)
        self._view_factors(vf_file)

        # Lumped thermal mass of 1 m2 of wall
        m = self.material
        self.heat_capacity = m['density'] * m['specific_heat'] * m['thickness']   # J/m2/K

        # Conduction from the hot inside face through the RSLE sheet to outside air
        self.r_out = m['thickness'] / m['conductivity']                           # m2K/W

        self.wall_temp_k = None     # wall temperatures live between steps (transient)

    def _build_geometry(self, sp_params, snout):
        # Same aperture frame as snout_soltrace: sideways, up, normal toward the field
        azimuth = np.radians(sp_params['receiver.0.rec_azimuth'])
        aperture_center = np.array([0.0, 0.0, sp_params['solarfield.0.tht']])
        unit_sideways = np.array([np.cos(azimuth), -np.sin(azimuth), 0.0])
        unit_up = np.array([0.0, 0.0, 1.0])
        unit_normal = np.array([np.sin(azimuth), np.cos(azimuth), 0.0])
        width = sp_params['receiver.0.rec_width']
        height = sp_params['receiver.0.rec_height']
        depth = snout['depth']
        flare = depth * np.tan(np.radians(snout['horiz']) / 2)
        top_front = height / 2 - depth * np.tan(np.radians(snout['top']))
        bottom_front = -height / 2 - depth * np.tan(np.radians(snout['bot']))

        # The enclosure: the 4 walls plus the 2 openings.
        # An opening is not a material surface, but radiation that leaves through it
        # never comes back, so it acts as a perfect absorber: a black surface.
        # Opening 0 = the aperture (hot cavity), opening 1 = front opening (ambient)
        front_height = top_front - bottom_front
        front_center = aperture_center + depth * unit_normal \
                + 0.5 * (top_front + bottom_front) * unit_up
        self.openings = [
            dict(center=aperture_center, u=unit_sideways, v=unit_up, w=width, h=height,
                normal=unit_normal, area=width * height),
            dict(center=front_center, u=unit_sideways, v=unit_up,
                w=2 * (width / 2 + flare), h=front_height, normal=-unit_normal,
                area=2 * (width / 2 + flare) * front_height),
        ]

        # Cut each wall into cells in the panel's grid frame (x along the aperture edge, y from the aperture edge
        # to the front). The ray bins are fine columns (cell_min wide) everywhere, so panel_power_map keeps one
        # rectangular grid per panel; in the band each fine bin is a cell, behind the band the fine bins of one
        # coarse column are owned by one coarse cell. The panels are trapezoids, so cells cut by a slanted edge
        # are clipped to their inside part: the clipped polygon is the exact cell shape the view factors are
        # computed on. A clipped piece smaller than min_cell_fraction of its own bins is merged with its inner
        # neighbour along the row, so no cell is a sliver whose temperature would be ray noise.
        self.cells = []             # one entry per cell: (panel index, row, col) of its first bin
        self.cell_bins = []         # all (panel index, row, col) bins the cell collects flux from
        self.cell_polygons = []     # the clipped cell outline, 3D corners in order
        self.cell_area = []
        self.panel_edges = []       # (x_edges, y_edges) per panel, the mesh panel_power_map must use
        self.cell_length = []       # characteristic length of the wall the cell is on, for natural convection
        self.cell_orientation = []
        for ip, panel in enumerate(self.panels):
            local_xy = panel['grid_xy']
            x_coarse = np.linspace(local_xy[:, 0].min(), local_xy[:, 0].max(), self.n_coarse + 1)
            fine_columns = [split_edges(x_coarse[k], x_coarse[k + 1], self.cell_min) for k in range(self.n_coarse)]   # each coarse column cut into fine ones
            x_edges = np.concatenate([edges[:-1] for edges in fine_columns] + [x_coarse[-1:]])
            coarse_columns = []                                                     # (first fine column, last fine column + 1) of each coarse column
            for edges in fine_columns:
                first = coarse_columns[-1][1] if coarse_columns else 0
                coarse_columns.append((first, first + len(edges) - 1))
            y_front, y_hinge = local_xy[:, 1].max(), local_xy[:, 1].min()
            band_rows = split_edges(y_hinge, y_hinge + self.band, self.cell_min)   # fine rows touching the aperture
            y_edges = np.concatenate([band_rows, np.linspace(y_hinge + self.band, y_front, self.n_coarse)[1:]])
            n_band_rows = len(band_rows) - 1
            self.panel_edges.append((x_edges, y_edges))
            n_rows = len(y_edges) - 1
            wall_area = polygon_area([np.r_[p, 0.0] for p in local_xy])
            wall_perimeter = sum(np.linalg.norm(local_xy[k] - local_xy[k - 1]) for k in range(4))
            orientation = WALL_ORIENTATION[panel['name']]
            length = np.ptp(local_xy[:, 0]) if orientation == 'vertical' else wall_area / wall_perimeter   # plate height, or area over perimeter for a horizontal plate

            def to_3d(outline):
                return [panel['centroid'] + p[0] * panel['grid_axes'][0] + p[1] * panel['grid_axes'][1] for p in outline]

            def clipped_cell(row, col_lo, col_hi):
                # the part of bins [row, col_lo:col_hi] inside the trapezoid, as 3D corners and area
                rect = [(x_edges[col_lo], y_edges[row]), (x_edges[col_hi], y_edges[row]),
                        (x_edges[col_hi], y_edges[row + 1]), (x_edges[col_lo], y_edges[row + 1])]
                outline = clip_cell(rect, local_xy)
                if len(outline) < 3:
                    return None, 0.0
                corners_3d = to_3d(outline)
                return corners_3d, polygon_area(corners_3d)

            def bin_area(row, col_lo, col_hi):
                return (x_edges[col_hi] - x_edges[col_lo]) * (y_edges[row + 1] - y_edges[row])

            def is_sliver(row, col_lo, col_hi):
                return clipped_cell(row, col_lo, col_hi)[1] < self.min_cell_fraction * bin_area(row, col_lo, col_hi)

            # The slanted edges are the two sides, so slivers are merged along the row toward the middle
            small = []
            for row in range(n_rows):
                initial = [(col, col + 1) for col in range(len(x_edges) - 1)] if row < n_band_rows else coarse_columns
                present = [(a, b) for a, b in initial if clipped_cell(row, a, b)[1] > 1e-6 * bin_area(row, a, b)]
                groups, k = [], 0
                while k < len(present):
                    start, stop = present[k]
                    while is_sliver(row, start, stop) and k + 1 < len(present) and present[k + 1][0] == stop:
                        k += 1
                        stop = present[k][1]
                    groups.append((start, stop))
                    k += 1
                if len(groups) > 1 and is_sliver(row, *groups[-1]):
                    groups[-2:] = [(groups[-2][0], groups[-1][1])]       # trailing sliver joins the cell before it
                for start, stop in groups:
                    corners_3d, cell_area = clipped_cell(row, start, stop)
                    if corners_3d is None:
                        continue
                    if is_sliver(row, start, stop):
                        small.append((panel['name'], cell_area))
                    bins = [(ip, row, col) for col in range(start, stop)]
                    self.cells.append(bins[0])
                    self.cell_bins.append(bins)
                    self.cell_polygons.append(corners_3d)
                    self.cell_area.append(cell_area)
                    self.cell_length.append(length)
                    self.cell_orientation.append(orientation)
            if small:   # a whole row of slivers with nothing to merge into; rare, but its temperature is ray noise
                print('SnoutThermal: %d cell(s) still smaller than %.0f%% of a bin: %s' % (len(small), 100 * self.min_cell_fraction,
                      ', '.join('%s %.4f m2' % (name, area) for name, area in small)))
        self.n_openings = 2
        self.n_surfaces = self.n_openings + len(self.cells)
        self.areas = np.array([o['area'] for o in self.openings] + self.cell_area)
        self.cell_area = np.array(self.cell_area)
        self.cell_length = np.array(self.cell_length)
        self.cell_panel = np.array([p for p, _, _ in self.cells])
        self.convection_groups = [(orientation, np.array([k for k, o in enumerate(self.cell_orientation) if o == orientation]))
                                  for orientation in ('vertical', 'hot_face_down', 'hot_face_up')]

## Thermal part starts here ##

    def _surface_polygon(self, i):
        # The outline of surface i as 3D corners: a rectangle for an opening,
        # the clipped cell polygon for a wall cell
        if i < self.n_openings:
            o = self.openings[i]
            hw, hh = o['w'] / 2, o['h'] / 2
            return [o['center'] + sx * hw * o['u'] + sy * hh * o['v']
                    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        return self.cell_polygons[i - self.n_openings]

    def _view_factors(self, vf_file):
        # View factor F[i,j]: of all diffuse radiation leaving surface i, the fraction
        # that lands on surface j. Pure geometry, no temperatures (Jackson 7.3 p. 298).
        # Computed ANALYTICALLY with the closed-form expression of Narayanaswamy (2015)
        # for two arbitrarily oriented planar polygons (viewfactor.py, ported from the
        # UW-Madison esolab viewFactor.m). Reciprocity A_i F_ij = A_j F_ji is exact by
        # construction, and the row sums come out at 1 to ~1e-6 with no correction.
        # Geometry never changes, so this runs once and the result is saved to a file.
        if vf_file and os.path.exists(vf_file):
            data = np.load(vf_file)
            if data['vf'].shape[0] == self.n_surfaces and np.allclose(data['areas'], self.areas):   # same cells as when it was saved
                self.vf = data['vf']
                self._build_matrix()
                return

        # Which plane each surface lies in: coplanar surfaces cannot see each other
        plane_of = [4, 5] + [self.cells[k][0] for k in range(len(self.cells))]
        polygons = [self._surface_polygon(i) for i in range(self.n_surfaces)]
        vf = np.zeros((self.n_surfaces, self.n_surfaces))
        for i in range(self.n_surfaces):
            for j in range(i + 1, self.n_surfaces):
                if plane_of[i] == plane_of[j]:
                    continue        # same plane: F = 0, nothing to compute
                # One call gives both directions, sharing the same edge-pair sum
                vf[i, j], vf[j, i] = view_factor(polygons[i], polygons[j])
        self.vf = vf
        if vf_file:
            np.savez(vf_file, vf=vf, areas=self.areas)
        self._build_matrix()

    def _build_matrix(self):
        # Radiosity J_i: ALL energy leaving surface i per m2, emission plus reflection:
        #   J_i = E_i + rho_i * sum_j F_ij J_j        (Jackson 7.4 p. 311)
        # One such equation per surface -> a linear system (I - diag(rho) F) J = E
        # (7.4 p. 313). This is where the gray body enters: in the infrared the walls
        # reflect rho = 1 - eps of what arrives, and those reflections bounce between
        # the walls.
        # Openings are black: rho = 0, they reflect nothing.
        # Same construction as Janna's create_and_invert_coeff_matrix: K built from
        # rho and F, inverted ONCE, reused every step.
        eps = self.material['ir_emissivity']
        self.rho = np.array([0.0] * self.n_openings + [1.0 - eps] * len(self.cells))
        K = np.eye(self.n_surfaces) - self.rho[:, None] * self.vf
        self.Kinv = np.linalg.inv(K)

    def reset(self, ambient_c=20.0):
        # Walls start at ambient, the cold-start condition
        self.wall_temp_k = np.full(len(self.cells), ambient_c + 273.15)

    # Natural convection coefficient of every cell at its current temperature, W/m2/K
    def convection_h(self, temp_k, t_air_k):
        h = np.zeros(len(self.cells))
        if self.natural_convection:
            for orientation, cells in self.convection_groups:
                if len(cells):
                    h[cells] = natural_convection_h(temp_k[cells], t_air_k, self.cell_length[cells], orientation)
        return h

    # One environment step. bin_power_maps: the 4 arrays from panel_power_map(PT, panel, *panel_edges[k]),
    # panel order as in add_snout, incident ray power per bin in W.
    # t_aperture_c: temperature of the hot cavity seen through the aperture.
    # h_wind: forced convection on the inside face, W/m2/K, zero by default; natural convection is always on top of it.
    def step(self, bin_power_maps, t_aperture_c, ambient_c=20.0, dt_s=60.0, h_wind=0.0):
        if self.wall_temp_k is None:
            self.reset(ambient_c)
        m = self.material
        eps = m['ir_emissivity']
        t_amb = ambient_c + 273.15

        # Alpha*q_sol term. The sunlight band is NOT solved here: SolTrace
        # already traced it, including the 80% bouncing diffusely between the walls.
        # The maps are incident power per bin; a cell's flux is the power of the bins it
        # owns over its true (clipped, merged) area, and absorbed = 0.20 x incident.
        q_incident = np.array([sum(bin_power_maps[p][r, c] for (p, r, c) in bins) for bins in self.cell_bins]) / self.cell_area
        q_solar = m['solar_absorptivity'] * q_incident

        # Openings emit as blackbodies at known temperatures (J = sigma*T^4, 7.4 p.313)
        emission = np.zeros(self.n_surfaces)
        emission[0] = SIGMA * (t_aperture_c + 273.15) ** 4     # aperture: hot cavity
        emission[1] = SIGMA * t_amb ** 4                       # front: sky and ground

        # March the wall temperatures through the step in small sub-steps (stability)
        n_sub = max(1, int(dt_s / 5.0))
        dt = dt_s / n_sub
        for _ in range(n_sub):
            temp = self.wall_temp_k

            # Infrared solve, the three lines of Janna's solve_radiative_exchange:
            emission[self.n_openings:] = eps * SIGMA * temp ** 4    # wall emission E
            radiosity = self.Kinv @ emission                        # J = Kinv @ E
            irradiation = self.vf @ radiosity                       # arriving: G = F @ J
            # Net infrared INTO each wall cell: absorbed eps*G minus emitted eps*sigma*T^4.
            # (infrared from the hot aperture AND radiation to sky and ground).
            q_ir_net = eps * irradiation[self.n_openings:] - emission[self.n_openings:]

            # rho*c*delta dT/dt = alpha*q_sol + q_ir - natural convection - forced convection - conduction to the back face
            h_conv = self.convection_h(temp, t_amb) + h_wind
            q_convection = h_conv * (temp - t_amb)
            q_total = q_solar + q_ir_net - q_convection - (temp - t_amb) / self.r_out
            self.wall_temp_k = temp + dt * q_total / self.heat_capacity

        # Report: peak wall temperature and peak incident flux per panel, compared to the 1200 C limit
        results = {}
        temp_c = self.wall_temp_k - 273.15
        for ip, panel in enumerate(self.panels):
            on_panel = self.cell_panel == ip
            results['SNOUT ' + panel['name'] + ' wall T max (C)'] = temp_c[on_panel].max()
            results['SNOUT ' + panel['name'] + ' peak flux (kW/m^2)'] = q_incident[on_panel].max() / 1e3
        results['SNOUT wall T max (C)'] = temp_c.max()
        results['SNOUT convection loss (kW)'] = float(np.sum(q_convection * self.cell_area)) / 1e3
        results['SNOUT over temp limit'] = bool(temp_c.max() > m['max_temp_c'])
        return results