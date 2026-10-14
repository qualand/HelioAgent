# SNOUT thermal model: wall temperatures from SolTrace
# Same methodology as Janna's falling_particle_receiver.py: gray body, diffuse enclosure
# Solved with radiosities

import numpy as np
from snout_soltrace import snout_corners, inside_quadrilateral
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

class SnoutThermal:
    # panels: the list returned by add_snout. n_grid must match the nbins useing in 
    # panel_flux_map, so a solar flux cell and a thermal cell are the same cell. 
    def __init__(self, sp_params, snout, panels, material=None, n_grid=4,
                 n_rays=20000, vf_file='snout_view_factors.npz', seed=1):
        self.material = {**RSLE57, **(material or {})}
        self.panels = panels
        self.n_grid = n_grid
        self._build_geometry(sp_params, snout)
        self._view_factors(n_rays, vf_file, seed)
        
        # Lumped thermal mass of 1 m2 of wall 
        m = self.material
        self.heat_capacity = m['density'] * m['specific_heat'] * m['thickness']   # J/m2/K
        
        # Conduction from the hot inside face through the RSLE sheet to outside air
        self.r_out = m['thickness'] / m['conductivity']                           # m2K/W
        
        self.wall_temp_k = None     # wall temperatures live between steps (transient)
        
    def _build_geometry(self, sp_params, snout):
        # Same aperture fram as snout_soltrace: sideways, up, normal toward the field
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
        # An opening is not a material surface, but radiations leaves through it never comes back
        # So it acts as a perfect absorber: a black surface
        # Opening 0 = the aperture (hot cavity), opening 1 = front opening (ambient)
        
        # The tunnel is convex, bounded by 6 planes. Store each as a point on the plane
        # and a normal pointing INTO the tunnel; a ray from inside always exits through
        # the plane with the smallest positive travel distance (used in _view_factors).
        self.plane_points = [p['centroid'] for p in self.panels]
        self.plane_normals = [p['rotation'][2] for p in self.panels]   # panel normals
        self.plane_points += [aperture_center, aperture_center + depth * unit_normal]
        self.plane_normals += [unit_normal, -unit_normal]
        
        # Both openings are rectangles: center, in-plane axes, size, inward normal
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
         
        # Cut each wall into n x n cells, on the SAME grid panel_flux map uses, so cell
        # [row, col] here receives flux[row, col] from the trace. The panels are trapezoids
        # So cells cut by a slanted edge only keep theirs inside area (measured with a fine
        # sub-grid, as in panel_flux_map)

        self.cells = []             # one entry per cell: (panel index, row, col)
        self.cell_area = []
        self.panel_edges = []
        for ip, panel in enumerate(self.panels):
            local_xy = panel['local_xy']
            x_edges = np.linspace(local_xy[:, 0].min(), local_xy[:, 0].max(), n + 1)
            y_edges = np.linspace(local_xy[:, 1].min(), local_xy[:, 1].max(), n + 1)
            self.panel_edges.append((x_edges, y_edges))
            cell_w, cell_h = x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]
            sub = 8
            sx = x_edges[0] + (np.arange(n * sub) + 0.5) * cell_w / sub
            sy = y_edges[0] + (np.arange(n * sub) + 0.5) * cell_h / sub
            gx, gy = np.meshgrid(sx, sy)
            inside = inside_quadrilateral(local_xy, np.c_[gx.ravel(), gy.ravel()])
            fraction = inside.reshape(n, sub, n, sub).mean(axis=(1, 3))
            for row in range(n):
                for col in range(n):
                    if fraction[row, col] > 0:
                        self.cells.append((ip, row, col))
                        self.cell_area.append(fraction[row, col] * cell_w * cell_h)
    
## Thermal part starts here ##

    def _view_factors(self, n_rays, vf_file, seed):
        # View factor F[i,j]: of all diffuse radiation leaving surface i, the fraction
        # that lands on surface j. Pure geometry, no temperatures (Jackson 7.3 p. 298).
        # Janna computes hers from a textbook formula because her surfaces are parallel
        # rectangles; our tilted trapezoids have no formula, so we MEASURE them:
        # !!! fire random rays from each surface and count where they land (Monte Carlo).
        # Geometry never changes, so this runs once and the result is saved to a file.
        # That file is the lookup table.
        if vf_file and os.path.exists(vf_file):
            data = np.load(vf_file)
            if data['vf'].shape[0] == self.n_surfaces:
                self.vf = data['vf']
                self._build_matrix()
                return

        rng = np.random.default_rng(seed)
        counts = np.zeros((self.n_surfaces, self.n_surfaces))
        for i in range(self.n_surfaces):
            # Random starting points on surface i
            origins, frame = self._sample_surface(i, n_rays, rng)

            # Random directions, cosine-weighted: a diffuse surface sends more energy
            # near its normal (the cos(theta) in the view factor integral, 7.3 p. 298)
            u1, u2 = rng.random(n_rays), rng.random(n_rays)
            cos_theta, phi = np.sqrt(u1), 2 * np.pi * u2
            sin_theta = np.sqrt(1.0 - u1)
            direction_local = np.c_[sin_theta * np.cos(phi),
                                    sin_theta * np.sin(phi), cos_theta]
            directions = direction_local @ frame    # rows of frame = local axes

            # Where does each ray exit? Smallest positive distance among the 6 planes
            t_exit = np.full(n_rays, np.inf)
            plane_hit = np.zeros(n_rays, dtype=int)
            for j, (point, normal) in enumerate(zip(self.plane_points, self.plane_normals)):
                denominator = directions @ normal
                t = np.where(denominator < -1e-12,
                             ((point - origins) @ normal) / denominator, np.inf)
                closer = (t > 1e-9) & (t < t_exit)
                t_exit[closer] = t[closer]
                plane_hit[closer] = j
            hit_points = origins + t_exit[:, None] * directions

            # Count the hits. Planes 4 and 5 are the openings (surfaces 0 and 1)
            counts[i, 0] += (plane_hit == 4).sum()
            counts[i, 1] += (plane_hit == 5).sum()
            # A hit on a wall is assigned to its cell by binning in the panel frame
            for ip in range(4):
                on_panel = plane_hit == ip
                if not on_panel.any():
                    continue
                panel = self.panels[ip]
                local = (hit_points[on_panel] - panel['centroid']) @ panel['rotation'][:2].T
                x_edges, y_edges = self.panel_edges[ip]
                col = np.clip(np.digitize(local[:, 0], x_edges) - 1, 0, self.n_grid - 1)
                row = np.clip(np.digitize(local[:, 1], y_edges) - 1, 0, self.n_grid - 1)
                for cell_index, (p, r, c) in enumerate(self.cells):
                    if p == ip:
                        counts[i, self.n_openings + cell_index] += \
                            ((row == r) & (col == c)).sum()
        vf = counts / n_rays
        
        # Clean the Monte Carlo noise with the two exact rules the estimates must obey:
        # reciprocity A_i F_ij = A_j F_ji (7.3 p. 299) and every row summing to 1
        vf = 0.5 * (vf + (self.areas[None, :] / self.areas[:, None]) * vf.T)
        vf /= vf.sum(axis=1, keepdims=True)
        self.vf = vf
        if vf_file:
            np.savez(vf_file, vf=vf, areas=self.areas)
        self._build_matrix()
        
    def _sample_surface(self, i, n_rays, rng):
        # Random points on surface i, plus its frame (rows = local axes in global)
        if i < self.n_openings:
            opening = self.openings[i]
            s = rng.uniform(-opening['w'] / 2, opening['w'] / 2, n_rays)
            u = rng.uniform(-opening['h'] / 2, opening['h'] / 2, n_rays)
            origins = opening['center'] + s[:, None] * opening['u'] + u[:, None] * opening['v']
            return origins, np.array([opening['u'], opening['v'], opening['normal']])
        ip, row, col = self.cells[i - self.n_openings]
        panel = self.panels[ip]
        x_edges, y_edges = self.panel_edges[ip]
        kept = np.zeros((0, 2))
        while len(kept) < n_rays:   # keep only points inside the trapezoid
            x = rng.uniform(x_edges[col], x_edges[col + 1], n_rays)
            y = rng.uniform(y_edges[row], y_edges[row + 1], n_rays)
            points = np.c_[x, y]
            kept = np.vstack([kept, points[inside_quadrilateral(panel['local_xy'], points)]])
        kept = kept[:n_rays]
        origins = panel['centroid'] + kept[:, [0]] * panel['rotation'][0] \
            + kept[:, [1]] * panel['rotation'][1]
        return origins, panel['rotation']
    
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
