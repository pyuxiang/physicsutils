#!/usr/bin/env python3
# Justin, 2022-08-11
# Simple implementation of rotating spectrometer code.

import datetime as dt
import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy

from ThorlabsPM100 import ThorlabsPM100, USBTMC
from k10cr1 import ThorlabsK10CR1

class BaseSpectrometer:
    """Implements rotating reflective grating spectrometer calibration methods.

    Assumes coupling of n=-1 grating mode. Typically to couple into single-mode
    fiber, so precision is important for sufficient mode matching.
    
    Methods listed:
        1. Offset angle with half angle for coupling.
        2. Two known wavelengths for interpolation.
        3. Offset angle with one known wavelength (usually defer to method 1).
        4. TODO: Account for errors in axis of rotation.
    """

    def __init__(self, lines_per_mm: int):
        """
        
        Args:
            lines_per_mm: Number of grating lines.
            offset_angle: Angle marking on motor that backreflects source light.
        """
        self.d = 1e-3 / lines_per_mm
        self.lambda2angle = None  # set by calibration method
        
        # For plotting
        self.fig, self.ax = plt.subplots()
        self.ylim = None; self.xlim = None
        plt.xlabel("Wavelength (nm)")
        plt.ion()  # enable interactive plotting

    def calibrate_halfangle(self, offsetangle: float, halfangle: float):
        """Defines conversion formula based on offset angle and half angle.

        Preconditions:
        1. Light from source port is backreflected when rotation stage
           set to 'offsetangle',
        2. Beam center on source port intersects axis of rotation orthogonally,
        3. Axis of rotation of grating is near grating surface,
        4. Light coupling into collection port is optimized at zeroth-order
           when rotation stage is set to 'halfangle'.

        Args:
            offsetangle: Angular readout by rotation stage at backreflection,
                in degrees.
            halfangle: Angular readout by rotation stage at zeroth-order
                coupling, in degrees.
        """
        def _lambda2angle(self, wl: float):
            """Converts target wavelength to target angle on rotation stage.
            
            Args:
                wl: Wavelength, in nm.
            Note:
                From grating equation,
                    d(sin I - sin R) = nL
                    2cos((I+R)/2)sin((I-R)/2) = nL/d
                Since I + R = 2 * halfangle, and using n = -1,
                    (I-R)/2 = (I - (2*halfangle - I))/2 = I - halfangle
                    I = halfangle - arcsin(nL/(2dcos(halfangle)))
            """
            wl_component = (wl*1e9) / self.d
            cos_component = 2*np.cos(np.deg2rad(halfangle))
            return offsetangle + halfangle - np.rad2deg(np.arcsin(wl_component/cos_component))
        return _lambda2angle

    def calibrate_knownwavelength(self, wls: list[float], angles: list[float]):
        """Defines conversion formula based on two calibrated wavelengths.

        This relies on the fact that a sufficiently narrow wavelength range (around
        say 1um) is approximately linear, so we can simply do a linear interpolation.
        This can be generalized to cubic splines.

        Preconditions:
        1. Roll of grating coupling is very precisely calibrated, i.e. rotation
           of grating is able to couple both n=1 and n=-1 (or even higher) order modes,
        2. Beam center on source port intersects axis of rotation orthogonally,
        3. Axis of rotation of grating is near grating surface,
        4. Two known wavelengths are available.

        Args:
            wls: List of calibrated wavelengths, in nm.
            angles: List of corresponding rotation stage angles, in degrees.
        
        Note:
            This method is susceptible to a number of issues including:
                1. Uneven coupling efficiencies between two endpoints
                2. Slight error due to wavelength interpolation/extrapolation
        TODO:
            - Explore using grating equation to perform the approximate fitting instead.
        """
        assert len(set(wls)) > 1, "One wavelength is insufficient for interpolation."
        assert len(wls) == len(angles), "Mismatch in number of coordinates provided."

        data = sorted(list(zip(wls, angles)))  # ensure wavelengths are strictly increasing
        wls, angles = list(zip(*data))
        cs = scipy.interpolate.CubicSpline(wls, angles)

        def _lambda2angle(self, wl: float):
            """Converts target wavelength to target angle on rotation stage.

            Args:
                wl: Wavelength, in nm.
            """
            return cs(wl)
        return _lambda2angle

    def scan(self, start_wl: float, end_wl: float, step: float):
        """Performs spectrum scan and plotting + logging.
        
        Note:
            Abstracted away into base class since certain rotation stages have
            a preferential direction for rotation, e.g. automatic backlash
            compensation during reverse.
        """

        # Using linspace instead of arange to minimize floating point errors
        wavelengths = np.linspace(start_wl, end_wl, int((end_wl-start_wl)/step)+1)
        angles = self.lambda2angle(wavelengths)

        xs = []; ys = []; line, = self.ax.plot(xs, ys)
        for wavelength, angle in zip(wavelengths, angles):

            # Set and measure
            self.set_angle(angle)
            x = wavelength
            y = self.measure()

            # Get axis bounds
            if not ylim:
                ylim = [y, y]
            else:
                if y > ylim[1]: ylim[1] = y
                if y < ylim[0]: ylim[0] = y
            if not xlim:
                xlim = [x, x]
            else:
                if x > xlim[1]: xlim[1] = x
                if x < xlim[0]: xlim[0] = x
            
            # Plot
            xs.append(x); ys.append(y)
            line.set_data(xs, ys)
            self.fig.canvas.draw()
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            plt.pause(0.01)

        np.savetxt(
            f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_spectrum.log",
            np.array([wavelengths, angles, ys]).T,
            fmt="%s",
        )

    def set_angle(self, angle: float):
        raise NotImplementedError
    
    def get_measurement(self):
        raise NotImplementedError


class SpectrometerK10CR1(BaseSpectrometer):
    """Example extension. User-defined to fit own measuring equipment."""

    def __init__(self, lines_per_mm: int, rot_path: str = "/dev/ttyACM0", det_path: str = "/dev/usbtmc0"):
        super().__init__(lines_per_mm)
        self.calibrate_halfangle(0.0, 25.8585)
        self.rot = ThorlabsK10CR1(rot_path, blocking=True)
        self.det = ThorlabsPM100(inst=USBTMC(device=det_path))

    def set_angle(self, angle: float):
        self.rot.angle = angle
    
    def get_measurement(self):
        return np.mean([self.det.read for i in range(20)])
