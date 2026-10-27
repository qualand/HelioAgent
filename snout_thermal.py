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
    # panels: the list returned by add_snout. n_grid must match the nbins used in
    # panel_flux_map, so a solar flux cell and a thermal cell are the same cell.
    def __init__(self, sp_params, snout, panels, material=None, n_grid=4,
                 vf_file='snout_view_factors_analytic.npz'):
        self.material = {**RSLE57, **(material or {})}
        self.panels = panels
        self.n_grid = n_grid
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
        n = self.n_grid
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

        # Cut each wall into n x n cells, on the SAME grid panel_flux_map uses, so cell
        # [row, col] here receives flux[row, col] from the trace. The panels are
        # trapezoids, so cells cut by a slanted edge are clipped to their inside part:
        # the clipped polygon is the exact cell shape the view factors are computed on.
        self.cells = []             # one entry per cell: (panel index, row, col)
        self.cell_polygons = []     # the clipped cell outline, 3D corners in order
        self.cell_area = []
        self.panel_edges = []
        for ip, panel in enumerate(self.panels):
            local_xy = panel['local_xy']
            x_edges = np.linspace(local_xy[:, 0].min(), local_xy[:, 0].max(), n + 1)
            y_edges = np.linspace(local_xy[:, 1].min(), local_xy[:, 1].max(), n + 1)
            self.panel_edges.append((x_edges, y_edges))
            for row in range(n):
                for col in range(n):
                    rect = [(x_edges[col], y_edges[row]), (x_edges[col + 1], y_edges[row]),
                            (x_edges[col + 1], y_edges[row + 1]), (x_edges[col], y_edges[row + 1])]
                    clipped = clip_cell(rect, local_xy)
                    if len(clipped) < 3:    # the cell lies entirely outside the trapezoid
                        continue
                    # Map the 2D clipped outline to 3D: centroid + x*sideways + y*up of the panel
                    corners_3d = [panel['centroid'] + p[0] * panel['rotation'][0]
                                  + p[1] * panel['rotation'][1] for p in clipped]
                    cell_area = polygon_area(corners_3d)
                    if cell_area < 1e-3 * (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]):
                        continue    # only a corner touches the trapezoid: zero-area sliver, not a cell
                    self.cells.append((ip, row, col))
                    self.cell_polygons.append(corners_3d)
                    self.cell_area.append(cell_area)
        self.n_openings = 2
        self.n_surfaces = self.n_openings + len(self.cells)
        self.areas = np.array([o['area'] for o in self.openings] + self.cell_area)

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
            if data['vf'].shape[0] == self.n_surfaces:
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

    # One environment step. flux_maps: the 4 arrays from panel_flux_map(PT, panel,
    # nbins=n_grid), panel order as in add_snout, incident solar in kW/m2.
    # t_aperture_c: temperature of the hot cavity seen through the aperture.
    def step(self, flux_maps, t_aperture_c, ambient_c=20.0, dt_s=60.0, h_wind=0.0):
        if self.wall_temp_k is None:
            self.reset(ambient_c)
        m = self.material
        eps = m['ir_emissivity']
        t_amb = ambient_c + 273.15

        # Alpha*q_sol term. The sunlight band is NOT solved here: SolTrace
        # already traced it, including the 80% bouncing diffusely between the walls.
        # The maps are incident flux, so absorbed = 0.20 x incident.
        q_solar = np.array([m['solar_absorptivity'] * flux_maps[p][r, c] * 1e3
                            for (p, r, c) in self.cells])

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

            # rho*c*delta dT/dt = alpha*q_sol + q_ir - conv - cond
            q_total = q_solar + q_ir_net - h_wind * (temp - t_amb) \
                - (temp - t_amb) / self.r_out
            self.wall_temp_k = temp + dt * q_total / self.heat_capacity

        # Report: peak wall temperature per panel, compared to the 1200 C limit
        results = {}
        temp_c = self.wall_temp_k - 273.15
        for ip, panel in enumerate(self.panels):
            on_panel = [i for i, (p, _, _) in enumerate(self.cells) if p == ip]
            results['SNOUT ' + panel['name'] + ' wall T max (C)'] = temp_c[on_panel].max()
        results['SNOUT wall T max (C)'] = temp_c.max()
        results['SNOUT over temp limit'] = bool(temp_c.max() > m['max_temp_c'])
        return results