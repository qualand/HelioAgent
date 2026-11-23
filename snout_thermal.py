import numpy as np
import os
from viewfactor import view_factor, polygon_area, clean_polygon

SIGMA = 5.6704e-8   # Stefan-Boltzmann constant (W/m2/K4)


class SnoutThermal:
    def __init__(self, field, snout, panels, params = {}, **overrides):

        #--- Geometry, from the SolTrace scene (snout_soltrace.add_snout)
        self.field = field                  # Aperture width and height, tower height and aperture azimuth are read from it
        self.snout = snout                  # SNOUT depth (m) and panel angles (deg)
        self.panels = panels                # The four panels with their SolTrace element, centroid, grid frame and outline

        #--- Wall material: RSLE-57 face sheet, back face exposed to outside air
        self.density = 2100.0               # Density (kg/m3)
        self.specific_heat = 1000.0         # Specific heat (J/kg/K)                            TODO confirm
        self.conductivity = 0.7             # Thermal conductivity (W/m/K)
        self.thickness = 0.025              # Face sheet thickness (m)                          TODO confirm
        self.solar_absorptivity = 0.2       # Solar absorptivity = 1 - 0.80 reflectivity (Bill's SNOUT optic)
        self.ir_emissivity = 0.8            # Infrared emissivity                                TODO confirm
        self.max_temp_c = 1200.0            # RSLE-57 use limit (C), the stop criterion in training

        #--- Mesh: square cells cell_min on a side at the aperture edge, growing linearly to at most cell_max at the outer border
        #    (snout_mesh_convergence.py, day 355 noon: the no-noise maximum converges at 5 cm; 10 cm reads 15 C low, 2.5 cm adds 1 C)
        self.cell_min = 0.05                # Side of the cells touching the aperture (m); also the width of the ray-counting bins
        self.cell_max = 0.20                # Largest cell allowed at the outer border (m)
        self.cell_along = None              # Fixed cell width along the aperture edge (m); None = as wide as the row is tall (squares)
        self.min_cell_fraction = 0.25       # A cell clipped by a slanted side to less than this fraction of its bins is merged into its neighbour
        self.vf_file = None                 # View factor cache file; None = one file per mesh, named after cell_min / cell_along / cell_max

        #--- Air at the film temperature, fits to Incropera et al. (2007) table A.4 between 250 and 1500 K
        self.air_prandtl = 0.71             # Prandtl number
        self.air_k0 = 0.0241                # Conductivity at 273 K (W/m/K), scaled by (T/273)^0.8
        self.air_nu0 = 1.33e-5              # Kinematic viscosity at 273 K (m2/s), scaled by (T/273)^1.7

        #--- Conduction along the wall
        self.lateral_conduction = True      # Heat flow between neighbouring cells of a wall through the sheet, k delta L / d per shared edge; the walls are not joined to each other

        #--- Convection on the inside face
        self.natural_convection = True      # Natural convection in still air; forced convection is added per step with h_wind
        self.wall_orientation = dict(top = 'hot_face_down', bottom = 'hot_face_up', east = 'vertical', west = 'vertical')   # Correlation per wall; tilts ignored

        #--- Simulation options
        self.sub_step_s = 5.0               # Explicit time sub-step (s); the wall time constant is about 90 s

        #--- State and results
        self.wall_temp_k = None             # Cell temperatures (K), carried between steps
        self.results = {}

        self.update_inputs({**params, **overrides})
        self.heat_capacity = self.density * self.specific_heat * self.thickness   # Lumped thermal mass of 1 m2 of wall (J/m2/K)
        self.r_out = self.thickness / self.conductivity                           # Conduction resistance through the sheet to outside air (m2K/W)
        self.set_up_mesh()
        self.set_up_conduction()
        self.calculate_view_factors()
        return

    def update_inputs(self, D):
        for k in D.keys():
            setattr(self, k, D[k])
        return

    #--- Air properties at temperature T (K)
    def air_conductivity(self, T):
        return self.air_k0 * (T / 273.15) ** 0.8            # W/m/K

    def air_kinematic_viscosity(self, T):
        return self.air_nu0 * (T / 273.15) ** 1.7           # m2/s

    #--- Natural convection coefficient of a flat plate in still air (W/m2/K), one value per cell
    #    Ra = g beta dT L^3 Pr / nu^2 at the film temperature; Nu correlations from Incropera et al. (2007) chapter 9:
    #    vertical: Churchill and Chu (1975); hot_face_down: 0.27 Ra^1/4 (McAdams 1954); hot_face_up: 0.54 Ra^1/4, 0.15 Ra^1/3 above Ra = 1e7
    def natural_convection_h(self, t_wall_k, t_air_k, length, orientation):
        t_film = 0.5 * (t_wall_k + t_air_k)
        rayleigh = 9.81 * np.abs(t_wall_k - t_air_k) / t_film * length ** 3 * self.air_prandtl / self.air_kinematic_viscosity(t_film) ** 2
        if orientation == 'vertical':
            nusselt = (0.825 + 0.387 * rayleigh ** (1 / 6) / (1 + (0.492 / self.air_prandtl) ** (9 / 16)) ** (8 / 27)) ** 2
        elif orientation == 'hot_face_up':
            nusselt = np.where(rayleigh < 1e7, 0.54 * rayleigh ** 0.25, 0.15 * rayleigh ** (1 / 3))
        else:
            nusselt = 0.27 * rayleigh ** 0.25
        return nusselt * self.air_conductivity(t_film) / length

    #--- Row boundaries from lo to hi: thickness growing linearly from `first` at lo to at most `largest` at hi
    def graded_edges(self, lo, hi, first, largest):
        depth = hi - lo
        n_rows = max(int(np.ceil(2 * depth / (first + largest))), 1)      # Fewest rows whose linear ramp from `first` ends at or below `largest`
        last = max(2 * depth / n_rows - first, first)                     # n rows averaging (first + last) / 2 fill the depth
        thickness = np.linspace(first, last, n_rows) if n_rows > 1 else np.array([depth])
        thickness *= depth / thickness.sum()                              # Exact fill after rounding
        return lo + np.concatenate([[0.0], np.cumsum(thickness)])

    #--- Part of a grid rectangle inside the panel trapezoid (Sutherland-Hodgman clipping, trapezoid counter-clockwise)
    def clip_cell(self, rect, trapezoid):
        def cross2(a, b):
            return a[0] * b[1] - a[1] * b[0]
        output = [np.array(p, dtype = float) for p in rect]
        for k in range(len(trapezoid)):
            a, b = np.array(trapezoid[k]), np.array(trapezoid[(k + 1) % len(trapezoid)])
            edge = b - a
            def inside(p):
                return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-12    # Left of the edge
            kept, output = output, []
            for idx in range(len(kept)):
                cur, prev = kept[idx], kept[idx - 1]
                if inside(cur):
                    if not inside(prev):
                        t = cross2(a - prev, edge) / cross2(cur - prev, edge)
                        output.append(prev + t * (cur - prev))               # Crossing point, coming in
                    output.append(cur)
                elif inside(prev):
                    t = cross2(a - prev, edge) / cross2(cur - prev, edge)
                    output.append(prev + t * (cur - prev))                   # Crossing point, going out
        return clean_polygon(output)

    #--- Set up the enclosure: two openings (aperture, front) and the four walls cut into cells
    #    Each wall is meshed in its own grid frame: x along the aperture edge, y from the aperture edge toward the front.
    #    Bins: a regular counting grid, cell_min wide, rows from graded_edges; panel_power_map counts the rays into these bins.
    #    Cells: groups of neighbouring bins in one row, as many as makes the cell as wide as the row is tall (or cell_along).
    #    Cells cut by a slanted side are clipped to the inside part; a clipped piece below min_cell_fraction joins its inner neighbour.
    def set_up_mesh(self):
        azimuth = np.radians(self.field.aperture_azimuth)
        aperture_center = np.array([0.0, 0.0, self.field.tower_height])
        unit_sideways = np.array([np.cos(azimuth), -np.sin(azimuth), 0.0])
        unit_up = np.array([0.0, 0.0, 1.0])
        unit_normal = np.array([np.sin(azimuth), np.cos(azimuth), 0.0])
        width = self.field.aperture_width
        height = self.field.aperture_height
        depth = self.snout['depth']
        flare = depth * np.tan(np.radians(self.snout['horiz']) / 2)
        top_front = height / 2 - depth * np.tan(np.radians(self.snout['top']))
        bottom_front = -height / 2 - depth * np.tan(np.radians(self.snout['bot']))

        # Openings: not material, but radiation leaving through them never returns, so they act as black surfaces
        front_height = top_front - bottom_front
        front_center = aperture_center + depth * unit_normal + 0.5 * (top_front + bottom_front) * unit_up
        self.openings = [
            dict(center = aperture_center, u = unit_sideways, v = unit_up, w = width, h = height, normal = unit_normal, area = width * height),            # 0: aperture (hot cavity)
            dict(center = front_center, u = unit_sideways, v = unit_up, w = 2 * (width / 2 + flare), h = front_height, normal = -unit_normal,
                 area = 2 * (width / 2 + flare) * front_height),                                                                                       # 1: front opening (ambient)
        ]

        self.cells = []             # One entry per cell: (panel index, row, col) of its first bin
        self.cell_bins = []         # All (panel index, row, col) bins the cell collects rays from
        self.cell_polygons = []     # Clipped cell outline, 3D corners in order
        self.cell_area = []         # True (clipped) area (m2)
        self.panel_edges = []       # (x_edges, y_edges) per panel: the bin grid panel_power_map must use
        self.cell_length = []       # Characteristic length of the wall the cell is on, for natural convection (m)
        self.cell_orientation = []  # Convection correlation of the wall the cell is on
        self.cell_span = []         # (panel index, row, first bin, last bin + 1) of each cell, for the neighbour search
        self.cell_center_xy = []    # Cell centroid in the panel's grid frame (m)
        for ip, panel in enumerate(self.panels):
            local_xy = panel['grid_xy']
            x_min, x_max = local_xy[:, 0].min(), local_xy[:, 0].max()
            n_half = int(np.ceil((x_max - x_min) / (2 * self.cell_min) - 1e-9))     # Bins cell_min wide, an even number so the mesh mirrors about the middle
            x_edges = np.linspace(x_min, x_max, 2 * n_half + 1)
            bin_width = (x_max - x_min) / (2 * n_half)
            y_front, y_hinge = local_xy[:, 1].max(), local_xy[:, 1].min()
            y_edges = self.graded_edges(y_hinge, y_front, self.cell_min, self.cell_max)   # Rows thin at the aperture, growing toward the front
            self.panel_edges.append((x_edges, y_edges))
            n_rows = len(y_edges) - 1

            wall_area = polygon_area([np.r_[p, 0.0] for p in local_xy])
            wall_perimeter = sum(np.linalg.norm(local_xy[k] - local_xy[k - 1]) for k in range(4))
            orientation = self.wall_orientation[panel['name']]
            length = np.ptp(local_xy[:, 0]) if orientation == 'vertical' else wall_area / wall_perimeter   # Plate height, or area / perimeter for a horizontal plate

            def row_columns(thickness):
                # Cells of one row as (first bin, last bin + 1), grouped outward from the middle so the two halves mirror
                cell_width = thickness if self.cell_along is None else self.cell_along
                group = max(1, int(round(cell_width / bin_width)))
                right = [(n_half + j * group, min(n_half + (j + 1) * group, 2 * n_half)) for j in range(int(np.ceil(n_half / group)))]
                return sorted([(2 * n_half - stop, 2 * n_half - start) for start, stop in right] + right)

            def to_3d(outline):
                return [panel['centroid'] + p[0] * panel['grid_axes'][0] + p[1] * panel['grid_axes'][1] for p in outline]

            def clipped_cell(row, col_lo, col_hi):
                # Part of bins [row, col_lo:col_hi] inside the trapezoid, as 3D corners and area
                rect = [(x_edges[col_lo], y_edges[row]), (x_edges[col_hi], y_edges[row]),
                        (x_edges[col_hi], y_edges[row + 1]), (x_edges[col_lo], y_edges[row + 1])]
                outline = self.clip_cell(rect, local_xy)
                if len(outline) < 3:
                    return None, 0.0
                corners_3d = to_3d(outline)
                return corners_3d, polygon_area(corners_3d)

            def bin_area(row, col_lo, col_hi):
                return (x_edges[col_hi] - x_edges[col_lo]) * (y_edges[row + 1] - y_edges[row])

            def is_sliver(row, col_lo, col_hi):
                return clipped_cell(row, col_lo, col_hi)[1] < self.min_cell_fraction * bin_area(row, col_lo, col_hi)

            small = []
            for row in range(n_rows):
                initial = row_columns(y_edges[row + 1] - y_edges[row])
                present = [(a, b) for a, b in initial if clipped_cell(row, a, b)[1] > 1e-6 * bin_area(row, a, b)]   # Drop cells entirely outside the wall
                groups, k = [], 0
                while k < len(present):
                    start, stop = present[k]
                    while is_sliver(row, start, stop) and k + 1 < len(present) and present[k + 1][0] == stop:   # Sliver on a slanted side: merge inward
                        k += 1
                        stop = present[k][1]
                    groups.append((start, stop))
                    k += 1
                if len(groups) > 1 and is_sliver(row, *groups[-1]):
                    groups[-2:] = [(groups[-2][0], groups[-1][1])]       # Trailing sliver joins the cell before it
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
                    self.cell_span.append((ip, row, start, stop))
                    self.cell_center_xy.append(((np.asarray(corners_3d) - panel['centroid']) @ panel['grid_axes'].T).mean(axis = 0))
            if small:   # A whole row of slivers with nothing to merge into; rare, its temperature is ray noise
                print('SnoutThermal: %d cell(s) still smaller than %.0f%% of a bin: %s' % (len(small), 100 * self.min_cell_fraction,
                      ', '.join('%s %.4f m2' % (name, area) for name, area in small)))

        self.n_openings = 2
        self.n_surfaces = self.n_openings + len(self.cells)
        self.areas = np.array([o['area'] for o in self.openings] + self.cell_area)
        self.cell_area = np.array(self.cell_area)
        self.cell_length = np.array(self.cell_length)
        self.cell_center_xy = np.array(self.cell_center_xy)
        self.cell_panel = np.array([p for p, _, _ in self.cells])
        self.convection_groups = [(orientation, np.array([k for k, o in enumerate(self.cell_orientation) if o == orientation]))
                                  for orientation in ('vertical', 'hot_face_down', 'hot_face_up')]
        return

    #--- Conduction between neighbouring cells of the same wall: C_ij = k delta L_ij / d_ij (W/K) per shared edge, L_ij the length of the edge
    #    inside the trapezoid and d_ij the distance between the cell centroids. Assembled once as a matrix so that (C @ T) is the heat flow into
    #    each cell. The four walls are not joined to each other (no flow across the corners). Only through-thickness conduction otherwise.
    def set_up_conduction(self):
        n = len(self.cells)
        self.conduction = np.zeros((n, n))
        if not self.lateral_conduction:
            return
        for ip, panel in enumerate(self.panels):
            x_edges, y_edges = self.panel_edges[ip]
            local_xy = panel['grid_xy']
            y_hinge, y_front = local_xy[:, 1].min(), local_xy[:, 1].max()
            hinge_x = np.sort(local_xy[np.isclose(local_xy[:, 1], y_hinge), 0])                    # Left and right end of the aperture edge
            front_x = np.sort(local_xy[np.isclose(local_xy[:, 1], y_front), 0])                    # Left and right end of the outer border

            def x_sides(y):                                                                         # Left and right boundary of the trapezoid at height y
                f = (y - y_hinge) / (y_front - y_hinge)
                return hinge_x[0] + (front_x[0] - hinge_x[0]) * f, hinge_x[1] + (front_x[1] - hinge_x[1]) * f

            def y_lowest(x):                                                                        # Lowest y inside the trapezoid on the line x = const
                if hinge_x[0] <= x <= hinge_x[1]:
                    return y_hinge
                x0, x1 = (hinge_x[0], front_x[0]) if x < hinge_x[0] else (hinge_x[1], front_x[1])  # The slanted side it crosses
                return y_hinge + (x - x0) / (x1 - x0) * (y_front - y_hinge) if x1 != x0 else y_front

            on_panel = [k for k, span in enumerate(self.cell_span) if span[0] == ip]
            for a in on_panel:
                _, row_a, start_a, stop_a = self.cell_span[a]
                for b in on_panel:
                    if b <= a:
                        continue
                    _, row_b, start_b, stop_b = self.cell_span[b]
                    if row_a == row_b and stop_a == start_b:                                        # Side by side: shared vertical edge at x = x_edges[stop_a]
                        x = x_edges[stop_a]
                        shared = max(0.0, y_edges[row_a + 1] - max(y_edges[row_a], y_lowest(x)))
                    elif abs(row_a - row_b) == 1 and start_a < stop_b and start_b < stop_a:         # One above the other: shared horizontal edge
                        y = y_edges[max(row_a, row_b)]
                        left, right = x_sides(y)
                        shared = max(0.0, min(x_edges[min(stop_a, stop_b)], right) - max(x_edges[max(start_a, start_b)], left))
                    else:
                        continue
                    if shared <= 0:
                        continue
                    distance = np.linalg.norm(self.cell_center_xy[a] - self.cell_center_xy[b])
                    c = self.conductivity * self.thickness * shared / distance
                    self.conduction[a, b] += c; self.conduction[b, a] += c
                    self.conduction[a, a] -= c; self.conduction[b, b] -= c
        return

    #--- Outline of surface i as 3D corners: a rectangle for an opening, the clipped polygon for a wall cell
    def surface_polygon(self, i):
        if i < self.n_openings:
            o = self.openings[i]
            hw, hh = o['w'] / 2, o['h'] / 2
            return [o['center'] + sx * hw * o['u'] + sy * hh * o['v'] for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        return self.cell_polygons[i - self.n_openings]

    #--- View factors F[i, j]: fraction of the diffuse radiation leaving surface i that lands on surface j (Jackson 7.3)
    #    Closed form of Narayanaswamy (2015) for two planar polygons (viewfactor.py, ported from the UW-Madison esolab viewFactor.m);
    #    reciprocity is exact by construction and row sums come out at 1 to ~1e-6. Geometry never changes, so computed once and cached.
    def calculate_view_factors(self):
        vf_file = self.vf_file
        if vf_file is None:
            vf_file = 'snout_vf_%gcm_%s_%gcm.npz' % (100 * self.cell_min, 'square' if self.cell_along is None else '%gcm' % (100 * self.cell_along), 100 * self.cell_max)
        if vf_file and os.path.exists(vf_file):
            data = np.load(vf_file)
            if data['vf'].shape[0] == self.n_surfaces and np.allclose(data['areas'], self.areas):   # Same cells as when it was saved
                self.vf = data['vf']
                self.create_and_invert_coeff_matrix()
                return

        plane_of = [4, 5] + [self.cells[k][0] for k in range(len(self.cells))]      # Coplanar surfaces cannot see each other
        polygons = [self.surface_polygon(i) for i in range(self.n_surfaces)]
        vf = np.zeros((self.n_surfaces, self.n_surfaces))
        for i in range(self.n_surfaces):
            for j in range(i + 1, self.n_surfaces):
                if plane_of[i] == plane_of[j]:
                    continue
                vf[i, j], vf[j, i] = view_factor(polygons[i], polygons[j])         # One call gives both directions
        self.vf = vf
        if vf_file:
            np.savez(vf_file, vf = vf, areas = self.areas)
        self.create_and_invert_coeff_matrix()
        return

    #--- Radiosity system (I - diag(rho) F) J = E (Jackson 7.4): built from rho and F, inverted once, reused every step
    #    Gray walls reflect rho = 1 - eps in the infrared; the openings are black (rho = 0)
    def create_and_invert_coeff_matrix(self):
        self.rho = np.array([0.0] * self.n_openings + [1.0 - self.ir_emissivity] * len(self.cells))
        K = np.eye(self.n_surfaces) - self.rho[:, None] * self.vf
        self.Kinv = np.linalg.inv(K)
        return

    #--- Walls at ambient: the cold-start condition
    def reset(self, ambient_c = 20.0):
        self.wall_temp_k = np.full(len(self.cells), ambient_c + 273.15)
        return

    #--- Convection coefficient of every cell at its current temperature (W/m2/K)
    def calculate_convection_coeff(self, temp_k, t_air_k):
        h = np.zeros(len(self.cells))
        if self.natural_convection:
            for orientation, cells in self.convection_groups:
                if len(cells):
                    h[cells] = self.natural_convection_h(temp_k[cells], t_air_k, self.cell_length[cells], orientation)
        return h

    #--- One environment step: march the wall temperatures through dt_s and report the maxima
    #    bin_power_maps: the four arrays from panel_power_map(PT, panel, *panel_edges[k]), incident ray power per bin (W), panel order as in add_snout
    #    t_aperture_c: temperature of the hot cavity seen through the aperture (C)
    #    h_wind: forced convection on the inside face (W/m2/K), on top of the natural convection
    #    power_per_ray: W carried by one ray in bin_power_maps (PT.powerperray). Given, every cell also gets its ray-count uncertainty
    #    dT = (dT/dq) q / sqrt(n) (Kline and McClintock 1953), and the 'upper' results are the maxima of T + dT
    def step(self, bin_power_maps, t_aperture_c, ambient_c = 20.0, dt_s = 60.0, h_wind = 0.0, power_per_ray = None):
        if self.wall_temp_k is None:
            self.reset(ambient_c)
        eps = self.ir_emissivity
        t_amb = ambient_c + 273.15

        # Absorbed sunlight: SolTrace already traced the solar band, including the 80 % bouncing diffusely between the walls
        q_incident = np.array([sum(bin_power_maps[p][r, c] for (p, r, c) in bins) for bins in self.cell_bins]) / self.cell_area   # Sum of the cell's bins over its true area (W/m2)
        q_solar = self.solar_absorptivity * q_incident

        # Openings emit as black bodies at known temperatures
        emission = np.zeros(self.n_surfaces)
        emission[0] = SIGMA * (t_aperture_c + 273.15) ** 4     # Aperture: hot cavity
        emission[1] = SIGMA * t_amb ** 4                       # Front: sky and ground

        n_sub = max(1, int(dt_s / self.sub_step_s))
        dt = dt_s / n_sub
        for _ in range(n_sub):
            temp = self.wall_temp_k
            emission[self.n_openings:] = eps * SIGMA * temp ** 4    # Wall emission E
            radiosity = self.Kinv @ emission                        # J = Kinv @ E
            irradiation = self.vf @ radiosity                       # Arriving: G = F @ J
            q_ir_net = eps * irradiation[self.n_openings:] - emission[self.n_openings:]   # Net infrared into each cell: absorbed minus emitted
            h_conv = self.calculate_convection_coeff(temp, t_amb) + h_wind
            q_convection = h_conv * (temp - t_amb)
            q_lateral = (self.conduction @ temp) / self.cell_area                       # Conduction from the neighbouring cells of the wall (W/m2), zero net over the wall
            q_total = q_solar + q_ir_net - q_convection - (temp - t_amb) / self.r_out + q_lateral   # rho c delta dT/dt = solar + infrared - convection - conduction to the back face + along the wall
            self.wall_temp_k = temp + dt * q_total / self.heat_capacity

        results = {}
        temp_c = self.wall_temp_k - 273.15

        def uncertainty_c(cell):
            # Ray-count uncertainty of one cell's temperature: dT = (dT/dq) q / sqrt(n), with dT/dq = alpha / (4 eps sigma T^3 + h + k/delta); lateral conduction ignored
            if not power_per_ray:
                return 0.0
            temperature_per_flux = self.solar_absorptivity / (4 * eps * SIGMA * self.wall_temp_k[cell] ** 3 + h_conv[cell] + 1 / self.r_out)
            rays_in_cell = q_incident[cell] * self.cell_area[cell] / power_per_ray
            return temperature_per_flux * q_incident[cell] / np.sqrt(max(rays_in_cell, 1.0))

        for ip, panel in enumerate(self.panels):
            on_panel = np.flatnonzero(self.cell_panel == ip)
            hottest_on_panel = on_panel[np.argmax(temp_c[on_panel])]
            results['SNOUT ' + panel['name'] + ' wall T max (C)'] = temp_c[hottest_on_panel]
            results['SNOUT ' + panel['name'] + ' wall T max upper (C)'] = temp_c[hottest_on_panel] + uncertainty_c(hottest_on_panel)   # The panel's hottest cell plus its own dT
            results['SNOUT ' + panel['name'] + ' peak flux (kW/m^2)'] = q_incident[on_panel].max() / 1e3
        hottest = int(np.argmax(temp_c))
        results['SNOUT wall T max (C)'] = temp_c[hottest]
        results['SNOUT T max uncertainty (C)'] = uncertainty_c(hottest)
        results['SNOUT wall T max upper (C)'] = temp_c[hottest] + results['SNOUT T max uncertainty (C)']   # T + dT of the hottest cell, what the environment uses
        results['SNOUT dT/dq (K per W/m2)'] = self.solar_absorptivity / (4 * eps * SIGMA * self.wall_temp_k[hottest] ** 3 + h_conv[hottest] + 1 / self.r_out)
        results['SNOUT convection loss (kW)'] = float(np.sum(q_convection * self.cell_area)) / 1e3
        results['SNOUT over temp limit'] = bool(temp_c.max() > self.max_temp_c)
        self.results = results
        return results


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#--- Example: build the mesh on the HelioAgent scene and print its size
if __name__ == "__main__":
    from functions_4_environment import build_field

    field, PT = build_field()
    snout = SnoutThermal(field, field.snout, field.snout_panels, cell_min = 0.05)
    print(len(snout.cells), 'cells,', snout.n_surfaces, 'surfaces with the two openings')
    for panel, (x_edges, y_edges) in zip(snout.panels, snout.panel_edges):
        print('%6s wall: bins %.1f cm wide, rows (cm): %s' % (panel['name'], 100 * np.diff(x_edges)[0], ', '.join('%.1f' % r for r in 100 * np.diff(y_edges))))