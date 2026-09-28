import numpy as np
from util import sun_vector

def _sun(sun_az, sun_el):
    s = sun_vector(sun_az, sun_el)          # unit vector pointing at the sun
    return np.array([s.x, s.y, s.z])

def _unit(v):                                # turns each row into a unit vector
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

class HelioKinematics:
    # Heliostat rotates in azimuth and elevation (as in heliostat.update_geometry).
    # This class adds a speed limit: each motor moves at most rate_mrad_s.

    def __init__(self, field, rate_mrad_s=1.3):
        self.rate = rate_mrad_s * 1e-3                                          # rad/s
        self.pos = np.array([[h.position.x, h.position.y, h.position.z] for h in field.heliostats])
        self.az = None
        self.el = None

    # Normal that sends the sun to aim_xyz (law of reflection, as in heliostat.update_geometry)
    def target_angles(self, aim_xyz, sun_az, sun_el):
        n = _unit(_unit(aim_xyz - self.pos) + _sun(sun_az, sun_el))
        return np.arctan2(n[:, 0], n[:, 1]), np.arcsin(n[:, 2])

    # Stow as heliostat.create_geometry does it: face up, azimuth toward the aim point
    def stow_angles(self, aim_xyz):
        d = aim_xyz - self.pos
        return np.arctan2(d[:, 0], d[:, 1]), np.full(len(d), np.pi / 2)

    # Jump straight there, no speed limit. Used at reset.
    def snap(self, az, el):
        self.az = np.array(az, dtype=float)
        self.el = np.array(el, dtype=float)

    # Each motor moves at most rate*dt_s radians this step
    def move_toward(self, az_t, el_t, dt_s):
        lim = self.rate * dt_s
        d_az = (az_t - self.az + np.pi) % (2 * np.pi) - np.pi                   # shortest way round
        d_el = el_t - self.el
        self.az = self.az + np.clip(d_az, -lim, lim)
        self.el = self.el + np.clip(d_el, -lim, lim)
        return np.maximum(np.abs(d_az), np.abs(d_el)) <= lim                    # True = arrived

    # Reflect the sun about the current normal, walk dist m along it.
    # That point goes into aim_point; it is on the receiver only once the mirror has arrived.
    def aim_points(self, sun_az, sun_el, dist):
        n = np.stack([np.sin(self.az) * np.cos(self.el),
                      np.cos(self.az) * np.cos(self.el),
                      np.sin(self.el)], axis=1)
        s = _sun(sun_az, sun_el)
        r = 2.0 * (n @ s)[:, None] * n - s
        return self.pos + np.asarray(dist)[:, None] * r
       
    