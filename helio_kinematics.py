import numpy as np
from util import sun_vector


#--- Unit vector toward the sun, as a numpy array
def _sun(sun_az, sun_el):
    s = sun_vector(sun_az, sun_el)
    return np.array([s.x, s.y, s.z])

#--- Each row of v turned into a unit vector
def _unit(v):
    return v / np.linalg.norm(v, axis = -1, keepdims = True)


#--- Heliostat motors: azimuth and elevation (as in heliostat.update_geometry) with a speed limit, so an aim-point change takes time
class HelioKinematics:
    def __init__(self, field, rate_mrad_s = 1.3):

        #--- Motors
        self.rate = rate_mrad_s * 1e-3                                          # Slew rate of each axis (rad/s)

        #--- State
        self.pos = np.array([[h.position.x, h.position.y, h.position.z] for h in field.heliostats])   # Heliostat positions (m), fixed
        self.az = None                                                          # Current azimuth of every mirror normal (rad)
        self.el = None                                                          # Current elevation (rad)
        self.travel = 0.0                                                       # Radians moved by all motors in the last step
        return

    #--- Normal that sends the sun to aim_xyz (law of reflection, as in heliostat.update_geometry), as azimuth and elevation
    def target_angles(self, aim_xyz, sun_az, sun_el):
        n = _unit(_unit(aim_xyz - self.pos) + _sun(sun_az, sun_el))
        return np.arctan2(n[:, 0], n[:, 1]), np.arcsin(n[:, 2])

    #--- Stow as heliostat.create_geometry does it: face up, azimuth toward the aim point
    def stow_angles(self, aim_xyz):
        d = aim_xyz - self.pos
        return np.arctan2(d[:, 0], d[:, 1]), np.full(len(d), np.pi / 2)

    #--- Jump straight to the angles, no speed limit; used at reset
    def snap(self, az, el):
        self.az = np.array(az, dtype = float)
        self.el = np.array(el, dtype = float)
        self.travel = 0.0
        return

    #--- Move each motor toward its target by at most rate * dt_s radians; returns True where the mirror arrived
    def move_toward(self, az_t, el_t, dt_s):
        lim = self.rate * dt_s
        d_az = (az_t - self.az + np.pi) % (2 * np.pi) - np.pi                   # Shortest way round
        d_el = el_t - self.el
        step_az = np.clip(d_az, -lim, lim)
        step_el = np.clip(d_el, -lim, lim)
        self.az = self.az + step_az
        self.el = self.el + step_el
        self.travel = float(np.sum(np.abs(step_az) + np.abs(step_el)))      # Radians moved by all motors this step, for the motor-travel term
        return np.maximum(np.abs(d_az), np.abs(d_el)) <= lim

    #--- Where each beam lands now: reflect the sun about the current normal and walk dist m along it.
    #    That point goes into aim_point; it is on the receiver only once the mirror has arrived.
    def aim_points(self, sun_az, sun_el, dist):
        n = np.stack([np.sin(self.az) * np.cos(self.el),
                      np.cos(self.az) * np.cos(self.el),
                      np.sin(self.el)], axis = 1)
        s = _sun(sun_az, sun_el)
        r = 2.0 * (n @ s)[:, None] * n - s                                      # Reflected direction
        return self.pos + np.asarray(dist)[:, None] * r