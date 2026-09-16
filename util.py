"""
@author: whamilton

Utility functions for SolTrace simulations
"""

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))

from api.pysoltrace import PySolTrace, Point
import numpy as np

def meinel_clearsky(day: int, zenith: float, altitude: float):
    """
    The Meinel model calculates solar intensity as a function of extraterrestrial radiation, site altitude, and solar zenith angle.
    NOTE: Taken from SolarPILOT Ambient.cpp

    :param day: [-] day of the year
    :param zenith: [radians] solar zenith angle
    :param altitude: [km] altitude of location
    
    :return dni [W/m^2]
    """
    czen = np.cos(zenith)
    s0 = 1.353*(1.+.0335*np.cos(2.*np.pi*(day+10.)/365.))  #[kW/m^2]      # NOTE: This is a different equation than the one in SolarPILOT's help documentation 
    if czen <= 0.0:
        dni = 0.0
    else:
        dni = (1.-.14*altitude)*np.exp(-.357/pow(czen,.678))+.14*altitude
    return s0 * dni * 1000.0 # [W/m^2]

def sun_position(latitude: float, day: int, hour: float)->tuple:
    """
    Computes the sun vector xyz given arguments (Copied from st_sun_position from stapi.cpp)
    TODO: This is just a place holder function. We should probably use NREL's Solar Position Algorithm (SPA)

	:param latitude: [deg] latitude 
	:param day: [-] day of the year 
	:param hour: [hour] solar time. 12.00 corresponds to sun at maximum elevation and does not necessarily match local time

    :return (azimuth, elevation): [degrees]
    """
    D2R = np.pi / 180
    R2D = 180 / np.pi
    declination = R2D * np.asin(0.39795 * np.cos(0.98563 * D2R * (day - 173)))
    hourAngle = 15 * (hour - 12)
    elevation = R2D * np.asin(np.sin(declination * D2R) * np.sin(latitude * D2R) + np.cos(declination * D2R) * np.cos(hourAngle * D2R) * np.cos(latitude * D2R))
    azimuth = R2D * np.acos((np.sin(D2R * declination) * np.cos(D2R * latitude) - np.cos(D2R * declination) * np.sin(D2R * latitude) * np.cos(D2R * hourAngle)) / np.cos(D2R * elevation) + 0.0000000001)
    if (np.sin(hourAngle * D2R) > 0.0):
        azimuth = 360 - azimuth
    return (azimuth, elevation)

def sun_vector(azimuth: float, elevation: float)->Point:
    """
    Computes the sun vector xyz given arguments (Copied from st_sun_position from stapi.cpp)

    Assumes xyz coordinate system:
		x: +east
		y: +north
		z: +zenith

	:param azimuth: [deg] solar azimuth 
	:param elevation: [deg] solar elevation 

    :return solar position (x, y, z):
    """
    D2R = np.pi / 180
    x = np.sin(azimuth * D2R) * np.cos(elevation * D2R)
    y = np.cos(azimuth * D2R) * np.cos(elevation * D2R)
    z = np.sin(elevation * D2R)
    return Point(x, y, z)

def sun_vector_from_latitude(latitude: float, day: int, hour: float)->Point:
    """
    Computes the sun vector xyz given arguments (Copied from st_sun_position from stapi.cpp)

    Assumes xyz coordinate system:
		x: +east
		y: +north
		z: +zenith

	:param latitude: [deg] latitude 
	:param day: [-] day of the year 
	:param hour: [hour] solar time. 12.00 corresponds to sun at maximum elevation and does not necessarily match local time

    :return solar position (x, y, z):
    """
    azimuth, elevation = sun_position(latitude, day, hour)
    return sun_vector(azimuth, elevation)

def compute_euler_angles(rotation: np.array):
    """
    Computes euler angles based on rotation matrix. This is consistant with the assumed rotation matrax within SolTrace.
    Source: https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf

    :param rotation: [3x3] matrix

    :return euler angles: [deg] array [beta, alpha, gamma]
    """
    if rotation[2,1] != 1.0 and rotation[2,1] != -1.0:
        beta = np.asin(rotation[2,1])
        alpha = np.atan2(rotation[2,0] / np.cos(beta), rotation[2,2] / np.cos(beta))
        gamma = - np.atan2(rotation[0,1] / np.cos(beta), rotation[1,1] / np.cos(beta))
    else:
        gamma = 0.0
        if rotation[2,1] == 1.0:
            beta = np.pi / 2
            alpha = np.atan2(rotation[0,2], rotation[1,2])
        else:
            beta = - np.pi / 2
            alpha = np.atan2(-rotation[0,2], -rotation[1,2])

    return np.array([beta, alpha, gamma])*180/np.pi

def trace(PT: PySolTrace, dni: float = 1000.0, nrays: int = 1e6, plot_trace: bool = False, nthreads: int = 1):
    PT.num_ray_hits = nrays
    PT.max_rays_traced = nrays*200
    PT.is_sunshape = True
    PT.is_surface_errors = True
    PT.dni = dni
    
    res = PT.run(123, nthread=nthreads, no_callback=True)
    if plot_trace:
        PT.plot_trace()

    return res