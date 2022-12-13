#!/usr/bin/env python3
# Base classes for non-linear crystal simulations
# Justin, 2021-11-24
#
# Original author: oem
# Comments:
#   Created on Tue Nov 21 09:53:01 2017
#   I have been using this for many many different configurations and types, thus all the unrelated stuff. In our case 
#   just used to calculate the Sellmeijer equations, the walkoff (maybe) and the phase difference.
#
# Changelog:
#   2021-11-23, Syed: Checked nonlinearCrystal1.py
#   2021-11-24, Justin: Refactored:
#     - Using more Pythonic conventions and documentation, for readability
#     - See 'migration_tests.py' for verification tests

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import fsolve

# Helper trigonometric functions in degrees
cosd = lambda deg: np.cos(deg*np.pi/180)
sind = lambda deg: np.sin(deg*np.pi/180)
tand = lambda deg: sind(deg)/cosd(deg)
arcsind = lambda x: np.arcsin(x)*180/np.pi
arctand = lambda x: np.arctan(x)*180/np.pi

class NonLinearCrystal(ABC):

    def __init__(self, length: float, theta: float, axis: str, model: str):
        """Defines the interface for all derivative non-linear crystal classes.

        This is an abstract class, which cannot be instantiated - use one of the
        non-linear crystal classes that inherit from this instead.

        Args:
            length: Crystal length along propagation direction, in mm.
            theta: Angle between propagation direction and optical axis, in degrees.
            axis:
                Direction along which the optical axis is rotated from the canonical
                propagation direction, i.e. z-axis: 'v' for vertical/pitch (+ve up), or
                'h' for horizontal/yaw (+ve right).
            model: One of the models defined in the respective non-linear crystal class.
        Notes:
            Arguments are intentionally forced to be explicit, for code clarity.
            Internally saves a reference to the equation used to calculate index, and
            the coefficients for the equation used.

            For extension, MODELS should define for each manufacturer, the equation used and
            coefficients, for ordinary and extraordinary polarizations respectively.

            Other types of cut are not included in the program, only rotation of optical axis
            about x- and y-axes (z-axis being the canonical propagation axis).
        """

        # Set refractive indices
        models = self.MODELS
        for k, v in models.items():
            if type(k) != tuple: k = (k,)  # force validation of the entire 'model' string
            if model in k: break
        else:
            raise ValueError(f"Invalid model - should be one of {tuple(models)}, not '{model}'")
        self._index_eqn_no, self._ao, self._index_eqn_ne, self._ae = v

        # Check optical axis definition
        if axis not in ("v", "h"):
            raise ValueError(f"Invalid value for 'axis' - can only be 'v' or 'h', not {axis}")
        
        self.length = length
        self.theta = theta
        self.axis = axis

    def index(self, wl: float, theta: float = None):
        """Returns refractive index of 'e' polarization along 'theta' degrees from optical axis.

        Args:
            wl: Wavelength, in um.
            theta: Angle from optical axis, in degrees.
        References:
            Refractive index mixture: https://www.rp-photonics.com/birefringence.html
        """
        if theta is None:
            theta = self.theta
        
        # Short-circuit
        if theta == 0: return self._index_no(wl)
        if theta == 90: return self._index_ne(wl)

        no = self._index_no(wl)
        ne = self._index_ne(wl)
        # Note the equivalent formula: no*((1+tand(theta)**2)/(1+(no*tand(theta)/ne)**2))**0.5
        # which has less precision error, but fails near 90 deg.
        return 1/( (cosd(theta)/no)**2 + (sind(theta)/ne)**2 )**0.5

    def _index_no(self, wl):
        """Returns refractive index for polarizations orthogonal to optical axis."""
        return self._index_eqn_no(self._ao, wl)  # wavelength in um
    
    def _index_ne(self, wl):
        """Returns refractive index for polarization parallel to optical axis."""
        return self._index_eqn_ne(self._ae, wl)  # wavelength in um

    def phase(self, wl, pol, alpha=0, gamma=0):
        """Returns the additional phase picked up in the crystal, in radians.

        Assumes the crystal input and output apertures are parallel.

        Args:
            wl: Wavelength, in um.
            pol: Polarization component of incident beam, with alternative definitions:
                 - 'h' for horizontal component
                 - 'v' for vertical component
                 - 'o' for component orthogonal to optical axis
                 - 'e' for component with projection along optical axis.
            alpha: Vertical component of incidence angle, in degrees. Down is +ve.
            gamma: Horizontal component of incidence angle, in degrees. Left is +ve.
        References:
            Phase-compensation: (Altepeter, 2005) https://doi.org/10.1364/OPEX.13.008951
                Note the errata is not relevant, which corrects for Type-I SPDC
                down-conversion photons originating from crystal center.
        Notes:
            Using Fig. 2 from (Altepeter, 2005) as reference diagram:
                 ^ x
                 |
                 +----/-----+
               e |   /axis  |
            o _|/|  /       |
               / +-/--------+--> z
              /α |
            
            In 'nonlinearCrystal1.py', the angles 'alpha' and 'gamma' represent the propagation
            angle inside the crystal. Here, 'alpha' and 'gamma' represent the incident angle
            *outside* the crystal, using Snell's law, assuming incidence at air-crystal interface.
            
            Note also that the orthogonal component still uses the small-angle approximation,
            and assumes gamma > 0 does not significantly affect the effective theta angle with
            optical axis.

            Includes some errata to 'nonlinearCrystal1.py':
            - self.calcN(wl) => ne, but should be no
            - self.calcN(wl,thetaEff) => n, but should be ne
            - inconsistent definition of alpha/gamma in return statement
        """
        if pol not in ("o", "e", "h", "v"):
            raise ValueError(f"Invalid value for 'pol' - can only be one of \"hvoe\", not '{pol}'")
        length = 1e3 * self.length  # mm -> um

        # Change definition of alpha/gamma from HV to incidence angle relative to optical axis,
        # i.e. using Fig.2 definition in 'Notes' for 'alpha',
        # n.b. 'alpha' retains same definition if axis is also vertical, otherwise replaced by 'gamma'.
        if self.axis == "h":
            alpha, gamma = gamma, alpha

        if pol in ("h", "v"):
            pol = "e" if pol == self.axis else "o"
        
        if pol == "e":

            # Calculate the actual deflection
            # If root-finding too computationally-intensive, consider the small-angle approximation:
            # theta_eff = self.theta - alpha; psi_e = alpha
            def func(t, theta, alpha):
                psi_e = arcsind(sind(alpha)/self.index(wl,t))  # refracted angle into crystal
                return t + psi_e - theta
            theta_eff = fsolve(func, x0=45, args=(self.theta, alpha), xtol=1e-3)[0]  # convert ndarray -> number

            n = self.index(wl, theta_eff)
            no = self._index_no(wl)
            ne = self._index_ne(wl)
            rho = (theta_eff - arctand(tand(theta_eff)*(no/ne)**2)) * (1 if no > ne else -1)
            kS = cosd(rho)  # correction factor, see 'References'
            beta = rho - (self.theta - theta_eff)  # psi_e = self.theta - theta_eff
            return (2*np.pi/wl) * (length*n*kS) /cosd(beta) /cosd(gamma)
    
        else: # pol == "o"
            no = self._index_no(wl)
            psi_o = arcsind(sind(alpha)/no)  # deflection independent of theta
            return (2*np.pi/wl) * (length*no) /cosd(psi_o) /cosd(gamma)
    
    def walkoff(self, wl, pol, alpha=0):
        """Calculates the walkoff angle, in degrees.

        For negative uniaxial crystals, the walkoff angle is always negative for positive 'alpha',
        since 'e' beam will be deflected further away from optical axis.

        'gamma' is not provided as an argument, assuming small angle. And calculating this is
        a lot more involved - although can consider resolving to plane that includes both
        incident angle and optical axis.

        Args:
            wl: Wavelength, in um.
            pol: Polarization component of incident beam, with alternative definitions:
                 - 'h' for horizontal component
                 - 'v' for vertical component
                 - 'o' for component orthogonal to optical axis
                 - 'e' for component with projection along optical axis.
            alpha: Incidence angle with component along optical axis.
                   Note this is different from that in 'NonLinearCrystal.phase'.
        Note:
            Uses the same equations as in phase calculation.
            Check 'NonLinearCrystal.phase' for correctness, before propagating changes here.

            Consistent with Eqn.1 in https://www.pmoptics.com/files/Beam_Displacer.pdf,
            for normal incidence, i.e. alpha = 0.
        """
        if pol not in ("o", "e", "h", "v"):
            raise ValueError(f"Invalid value for 'pol' - can only be one of \"hvoe\", not '{pol}'")

        if pol in ("h", "v"):
            pol = "e" if pol == self.axis else "o"

        if pol == "e":
            def func(t, theta, alpha):
                psi_e = arcsind(sind(alpha)/self.index(wl,t))  # refracted angle into crystal
                return t + psi_e - theta
            theta_eff = fsolve(func, x0=45, args=(self.theta, alpha), xtol=1e-3)[0]  # convert ndarray -> number

            no = self._index_no(wl)
            ne = self._index_ne(wl)
            rho = (theta_eff - arctand(tand(theta_eff)*(no/ne)**2)) * (1 if no > ne else -1)
            beta = rho - (self.theta - theta_eff)  # psi_e = self.theta - theta_eff
            return alpha + beta
        
        else: # pol == "o"
            no = self._index_no(wl)
            psi_o = arcsind(sind(alpha)/no)  # deflection independent of theta
            return alpha - psi_o
    
    def beam_separation(self, wl, alpha=0):
        """Calculates the beam separation, in mm.

        Value is the separation of the 'e' ray from the 'o' ray, with +ve direction along
        'up' for axis='v', and 'right' for axis='h'.

        Args:
            wl: Wavelength, in um.
            alpha: Incidence angle with component along optical axis.
                   Note this is different from that in 'NonLinearCrystal.phase',
                   see 'NonLinearCrystal.walkoff' for details.
                   
        Notes:
            Cross-verified by Darren, 2022-12-13.
        """
        walkoff_e = self.walkoff(wl, "e", alpha=alpha)
        walkoff_o = self.walkoff(wl, "o", alpha=alpha)

        # Resolve to incident angle from crystal facet normal
        walkoff_norm_e = walkoff_e - alpha
        walkoff_norm_o = alpha - walkoff_o
        return cosd(alpha) * self.length * (tand(walkoff_norm_o)+tand(walkoff_norm_e))

    @property
    @abstractmethod
    def MODELS(self):
        """Should be overridden in subclass with the following format:
        {
            "manufacturer": (
                no_equation, no_equation_coefficients,
                ne_equation, ne_equation_coefficients,
            ),
            ...
        }
        """
        raise NotImplementedError
            
    @staticmethod
    def index_eqn1(coeffs, wl):
        """A + B/(L^2-C) - D*L^2"""
        a, b1, c1, b2 = coeffs
        n_sq = a + b1/(wl**2-c1) - b2*wl**2
        return n_sq**0.5

    @staticmethod
    def index_eqn2(coeffs, wl):
        """A + B*L^2/(L^2-C) + ..."""
        n_sq, *bs = coeffs
        assert len(bs) % 2 == 0
        for i in range(len(bs)//2):
            b, c = bs[2*i : 2*i+2]
            n_sq += b*wl**2/(wl**2 - c)
        return n_sq**0.5


class AlphaBBO(NonLinearCrystal):
    MODELS = {
        # https://www.mt-optics.net/a-BBO.html
        "MTOptics": ( 
            NonLinearCrystal.index_eqn1, [2.7471, 0.01878, 0.01822, 0.01354],
            NonLinearCrystal.index_eqn1, [2.37153, 0.01224, 0.01667, 0.01516],
        ),
        # https://www.agoptics.com/Alpha-BBO.html
        "AgOptics": ( 
            NonLinearCrystal.index_eqn1, [2.7471, 0.01878, 0.01822, 0.01354],
            NonLinearCrystal.index_eqn1, [2.3174, 0.01224, 0.01667, 0.01516],
        ),
        # https://www.castech.com/product/%CE%B1-BBO-90.html
        "Castech": (
            NonLinearCrystal.index_eqn1, [2.7471, 0.01878, 0.01822, 0.01354],
            NonLinearCrystal.index_eqn1, [2.37153, 0.01224, 0.01667, 0.01516],
        ),
    }


class YVO4(NonLinearCrystal):
    MODELS = {
        # https://www.mt-optics.net/YVO4.html
        # https://www.agoptics.com/YVO4.html
        ("AgOptics", "MTOptics"): (
            NonLinearCrystal.index_eqn1, [3.77834, 0.069736, 0.04724, 0.0108133],
            NonLinearCrystal.index_eqn1, [4.5905, 0.110534, 0.04813, 0.0122676],
        ),
        # https://www.castech.com/product/YVO4-92.html
        "Castech": (
            NonLinearCrystal.index_eqn1, [3.77834, 0.069736, 0.04724, 0.0108133],
            NonLinearCrystal.index_eqn1, [4.59905, 0.110534, 0.04813, 0.0122676],
        ),
    }


class MgF2(NonLinearCrystal):
    MODELS = {
        # https://refractiveindex.info/?shelf=main&book=MgF2&page=Li-o, 0.14-7.5um
        "Li": (
            NonLinearCrystal.index_eqn2, [1.27620, 0.60967, 0.08636**2, 0.0080, 18.0**2, 2.14973, 25.0**2],
            NonLinearCrystal.index_eqn2, [1.25385, 0.66405, 0.08504**2, 1.0899, 22.2**2, 0.1816, 24.4**2, 2.1227, 40.6**2],
        ),
        # https://refractiveindex.info/?shelf=main&book=MgF2&page=Dodge-o, 0.2-7.0um
        "Dodge": (
            NonLinearCrystal.index_eqn2, [1, 0.48755108, 0.04338408**2, 0.39875031, 0.09461442**2, 2.3120353, 23.793604**2],
            NonLinearCrystal.index_eqn2, [1, 0.41344023, 0.03684262**2, 0.50497499, 0.09076162**2, 2.4904862, 23.771995**2],
        ),
    }


class FusedQuartz(NonLinearCrystal):
    MODELS = {
        # https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson, 0.21-6.7um
        "Malitson": (
            NonLinearCrystal.index_eqn2, [1, 0.6961663, 0.0684043**2, 0.4079426, 0.1162414**2, 0.8974794, 9.896161**2],
            NonLinearCrystal.index_eqn2, [1, 0.6961663, 0.0684043**2, 0.4079426, 0.1162414**2, 0.8974794, 9.896161**2],
        ),
    }


class Quartz(NonLinearCrystal):
    MODELS = {
        # https://refractiveindex.info/?shelf=main&book=SiO2&page=Ghosh-o, 0.198-2.0531um
        "Ghosh": (
            NonLinearCrystal.index_eqn2, [1.28604141, 1.07044083, 1.00585997e-2, 1.10202242, 100],
            NonLinearCrystal.index_eqn2, [1.28851804, 1.09509924, 1.02101864e-2, 1.15662475, 100],
        ),
        # https://refractiveindex.info/?shelf=main&book=SiO2&page=Radhakrishnan-o, 0.18-3um
        "Radhakrishnan": (
            NonLinearCrystal.index_eqn2, [1, 0.663044, 0.060**2, 0.517852, 0.106**2, 0.175912, 0.119**2, 0.565380, 8.844**2, 1.675299, 20.742**2],
            NonLinearCrystal.index_eqn2, [1, 0.665721, 0.060**2, 0.503511, 0.106**2, 0.214792, 0.119**2, 0.539173, 8.792**2, 1.807613, 19.70**2],
        ),
    }


# Errata for 'nonlinearCrystal.py':
# - YVO4 self.ao, A = 3.77879 from Shi 2001 (https://refractiveindex.info/?shelf=main&book=YVO4&page=Shi-o-20C)
#   but the other coefficients follow MT-Optics's instead.
