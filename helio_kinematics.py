import numpy as np
from util import sun_vector

def_sun(sun_az, sun_el)
    s = sun_vector(sun_az, sun_el) # unit vector pointing at the sun
    return np.array ([s.x, s.y, s.z])

# Turns a vector into a unit vector
def_unit(v):
    return v / np.linalg.norm(v, axis = 1, keepdims=True)

class HelioKinematics:
    # Heliostat rotates azimuth and elevation
    # This class adds a speed limit: each motor moves at most rate_mrad_s
    
    def __init__(self, field, rate_mrad_s=1.3): # Regular speed of 1.3 mrad/second
        self.rate = rate_mrad_s * 1e-3 # transforming to rad/s
        self.pos = np.array([[h,position.x, h.position.y, h.position.z] for h in field.heliostats])
        self.az = None
        self.el = None
        
    # With stow_angles the azimuth motor is already facing the tower, so only the elevation motor has to move from 90 down to tracking angle
    def stow_angles
        d = aim_xyz - self.pos
        return np.arctan2(d[:.0], d[:,1]), np.full(len(d), np.pi / 2)
    
    # Jump straight at this location, no speed limit. Uset at reset. 
    def snap(self, az, el): 
        self.az = np.array(az, dtype=float)
        self.el = np.array(el, dtype = float) 
        
    # Each motor moves at most rate*dt_s radians this step
    def move_toward(self, az_t, el_t, dt_s):
        lim = self.rate * dt_s # speed limit
        d_az = (az_t - self.az + np.pi) % (2 * np.pi) - np.pi # delta azimuth
        d_el = el_t - self.el
        self.az = self.az + np.clip(d_az, -lim, lim)
        self.el = self.el + np.clip(d_el, -lim, lim)
        return np.maximum(np.abs(d_az), np.abs(d_el)) <= lim # True = arrived
    
    # Reflect the sun about the current normal, walk distance m along it
    # That point goes into aim_point; it is on the receiver only once the mirror has arrived.
    n = np.stack([np.sin(self.az) * np.cos(self.el),
                  np.cos(self.az) * np.cos(self.el),
                  np.sin(self.el)], axis=1)
    s = _sun(sun_az, sun_el)
    r = 2.0 * (n @ s)[:, None] * n - s
    return self.pos + np.asarray(dist)[:, None] * r
    
       
    