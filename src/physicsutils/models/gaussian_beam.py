#!/usr/bin/env python3
"""
Justin
2022-03-01: Simplified, add more documentation.

All numbers expressed in meters.
Note that paraxial approximation only works for small divergences, e.g. ~10 degrees give 0.5% error.
Make sure the beam waist is generally larger than the wavelength.

Nice references:
- https://www.rp-photonics.com/gaussian_beams.html
- https://www.rp-photonics.com/abcd_matrix.html

Other stuff to reference:
- https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=3811&pn=C330TMD-B
- https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
- https://en.wikipedia.org/wiki/Gaussian_beam
- https://www.edmundoptics.com.sg/knowledge-center/application-notes/lasers/gaussian-beam-propagation/
- https://www.rp-photonics.com/abcd_matrix.html

Reminder to avoid thin lens approximation, see this:
- https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=14511
- But in general thin lens is a rather good approximation.

Improvements:
- Following A Cere mm_twolens, see if can implement solution to arbitrarily match two
  Gaussian beam profiles (input and output), using two lens.
  - Strategy: Two Beams, second backward-propagating. Iterate through all first lens
              positions, then iterate through all second lens positions until minimum reached.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

pi = np.pi

class Beam:
    """Defines a group of beam segments.
    
    Beams are grouped as individual segments, each with corresponding q-parameter,
    wavelength, and the refractive index of the medium. The position marker denotes
    the actual physical position of the beam, which Re(q) denotes the location of
    the beam waist *relative* to the current position.
    """

    def __init__(self, q: complex, wl: float, pos: float = 0, n: float = 1):
        """
        Args:
            q: q-parameter of beam.
            wl: Wavelength of beam, in meters.
            pos: Current position, in meters.
            n: Refractive index.
        """
        self.wl = wl
        self.q = q
        self.n = n
        self.pos = pos
        self.segments = [(q, n, pos)]

    @classmethod
    def from_params(cls, wl: float, w: float, r: float = np.inf, n: float = 1, pos: float = 0):
        """Generates q parameter for Gaussian beam.

        Args:
            wl: Wavelength, in meters.
            w: Beam waist, in meters.
            r: Beam radius of curvature, in meters.
            n: Refractive index
            pos: Relative translation
        """
        q = 1/(1/r - 1j*(wl/np.pi/n/(w**2)))
        return Beam(q, wl, n=n, pos=pos)

    def copy(self):
        segments = list(self.segments)
        beam = Beam(self.q, self.wl, self.pos, self.n)
        beam.segments = segments
        return beam

    def evolve(self, matrix=[[1,0],[0,1]], dist: float = 0, n = None):
        """Returns q-parameter after evolution by ABCD matrix.

        The workhorse of the entire class :)
        
        Args:
            matrix: The ABCD matrix to be applied.
            dist: Propagation distance before evolution, in meters.
            n: Refractive index on outgoing interface.
        Note:
            For free space propagation, simply use the identity matrix
            with non-zero dist. Or [[1,dist],[0,1]] if it floats your boat.
        """
        # Propagation
        self.pos += dist
        self.q += dist

        # Apply ray transfer matrix
        (a,b), (c,d) = matrix
        self.q = (a*self.q+b)/(c*self.q+d)

        # If refractive index changes
        if n is not None: self.n = n
        self.segments.append((self.q, self.n, self.pos))
        return self
    
    def apply_free_propagation(self, d: float = 0):
        """Applies beam propagation.
        
        Effectively shifts the current position along the beam.

        Note:
            This functionality is already provided in the other evolutions,
            through the parameter d.
        """
        self.evolve(dist=d)
        return self
    
    def apply_thin_lens(self, f: float, d: float = 0):
        """Applies thin lens.

        Args:
            f: Length, in meters.
            d: Distance to lens, in meters. Optional.
        """
        matrix = [[1,0], [-1/f,1]]
        self.evolve(matrix, d)
        return self

    def apply_curved_interface(self, r: float, n1: float, n2: float, d: float = 0):
        """Applies curved surface.

        Args:
            r: Radius of curvature of surface (+ve for convex at incidence)
            n1: Initial refractive index
            n2: Final refractive index
            d: Distance to lens, in meters. Optional.
        """
        matrix = [[1,0], [(n1-n2)/(r*n2),n1/n2]]
        self.evolve(matrix, d, n2)
        return self

    def apply_thick_lens(self, t: float, r1: float, r2: float, n: float, d: float = 0):
        """Applies thick lens, assuming air-glass interface.

        Some typical values for glass refractive indices:
          - D-ZK3: 1.5860 @ 660nm - https://refractiveindex.info/?shelf=glass&book=CDGM-ZK&page=D-ZK3

        Args:
            t: Thickness of lens, in meters.
            r1: Radius of curvature of incident surface.
            r2: Radius of curvature of outgoing surface.
            n: Refractive index of lens.
            d: Distance to lens, in meters. Optional.
        References:
            Thick lens matrix: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
        """
        self.apply_curved_interface(r1, 1, n, d)
        self.apply_curved_interface(r2, n, 1, t)
        return self

    def apply_block(self, t: float, n: float, d: float = 0):
        """Applies a solid block, e.g. a glass slide with dual-plano surfaces.
        
        Args:
            t: Thickness, in meters.
            n: Refractive index of block.
            d: Distance to lens, in meters. Optional.
        """
        self.apply_thick_lens(t, np.inf, np.inf, n, d)
        return self
        
    def get_beam_width(self, z: float, q: complex = None, n: float = None):
        if q is None:
            q = self.q
        if n is None:
            n = self.n
        z0, zr = q.real, q.imag  # beam waist position, Rayleigh range
        w0 = np.sqrt(self.wl*zr / np.pi / n)  # beam waist
        return w0 * np.sqrt(1 + ((z+z0)/zr)**2)  # beam width

    def get_waist(self, q: complex = None, n: float = None):
        if q is None:
            q = self.q
        if n is None:
            n = self.n
        z0, zr = q.real, q.imag  # beam waist position, Rayleigh range
        w0 = np.sqrt(self.wl*zr / np.pi / n)  # beam waist
        return self.pos - z0, w0

    def calculate_waist(self, target_distance, callback, *args, bounds=None, target_distance_type="relative", **kwargs):
        """Given a beam, and a callback to perform waist calculations
        find the waist size obtained at the desired target_distance.

        Args:
            target_distance (float): Position at which desired waist
            callback: Beam.apply_XXX
            callback_args (tuple): Beam.apply_XXX(self, *args, **kwargs)
        Note:
            Optimization as follows: Binary search, iterate between 0 and target_distance.
            If too far, will be further from target_distance, otherwise nearer. 

            Important: There are different regimes for lens,
              1. Near-field - at focal length, focus -> collimated
              2. Mid-field - slightly further from focus, waist at long distances  <- typically desired
              3. Far-field - much farther than focus, collimated -> focus
        See:
            Beam waist pos vs lens pos: https://pyuxiang.com/upload/20220221_114834_wlenspos_waistpos_zoom.png
            Beam waist size vs pos: https://pyuxiang.com/upload/20220221_114256_waistpos_waistsize.png
        """
        if target_distance_type == "relative":
            absolute_distance = self.pos + target_distance
        elif target_distance_type == "absolute":
            absolute_distance = target_distance
        else:
            raise ValueError("TODO")

        # Assign default bounds if None provided
        if bounds is None:
            bounds = [0, target_distance]  # but most likely will converge to far-field,
                                           # which is typically not desired...
        # The pure function to converge
        def f(d):
            beam = self.copy()
            kwargs["d"] = d
            callback(beam, *args, **kwargs)
            z, w = beam.get_waist()  # optimize for position
            return z

        return binary_converge(f, bounds, absolute_distance)

    def calculate_position(self, target_waist, callback, *args, bounds=None, target_distance_type="absolute", **kwargs):
        """Given a beam, and callback, find the position.
        
        Args:
            target_distance_type: Value of distance to be relative to lens or absolute positioning.
                One of 'absolute' or 'relative'.
        """

        # The pure function to converge
        def f(d):
            beam = self.copy()
            kwargs["d"] = d
            callback(beam, *args, **kwargs)
            z, w = beam.get_waist()
            return w

        # Assign default bounds if None provided
        if bounds is None:
            raise ValueError("Eh ps, 'bounds' must be specified. MUST.")
        
        return binary_converge(f, bounds, target_waist)
    
    def plot(self, z0=0, z1=None, show=False, label=None, xunit="m", yunit="um", color=None):
        """All-in-one function to illustrate the beam.

        Args:
            z0: Left bound of beam.
            z1: Right bound of beam, default to twice total propagation distance.

            show: Whether plot should show automatically or not, defaults to False.
            xunit: x-axis scaling factor, one of {km,m,cm,mm,um,nm}. Defaults to m.
            yunit: y-axis scaling factor, one of {km,m,cm,mm,um,nm}. Defaults to um.
            color: matplotlib.Color of beam, defaults auto.
            label: Description of beam, defaults None.
        Note:
            No figure defined - lends itself nicely to plot injection, hence the 'show' param.
        TODO:
            Fix problematic plotting when z1 is smaller than the last bound.
        """
        if z1 is None:
            z1 = 2 * self.pos

        # Scale the units of plotted diagram
        SCALES = { "km": 1000, "m": 1, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9 }
        yscale, xscale = SCALES[yunit], SCALES[xunit]

        segs = list(self.segments)
        segs.append((self.q, self.n, z1))
        for i in range(len(segs)-1):
            (q, n0, d0), (_, _, d1) = segs[i:i+2]
            zs = np.linspace(0, d1-d0, 10000) if i != 0 else np.linspace(z0, d1-d0, 10000)
            ys = [self.get_beam_width(z,q,n0)/yscale for z in zs]
            xs = (zs+d0)/xscale
            
            # Plot with consistent color and single label
            kwargs = {}
            if color: kwargs["color"] = color
            if i == 0 and label: kwargs["label"] = str(label)
            p = plt.plot(xs, ys, **kwargs)
            color = p[0].get_color()

        if show:
            plt.show()

def isclose(v1, v2, tol=1e-4, tol_type="relative"):
    if tol_type == "relative":
        return abs(v1-v2)/v1 < tol
    elif tol_type == "absolute":
        return abs(v1-v2) < tol
    else:
        raise ValueError(f"Wrong 'tol_type' bruh, no '{tol_type}'")

def binary_converge(f, bounds, target, **kwargs):
    assert hasattr(bounds, "__len__") and len(bounds) == 2
    left, right = bounds

    left_result = (f(left) - target)
    right_result = (f(right) - target)

    # Check the bounds even make sense (i.e. 0 must be behind, etc.)
    assert left_result * right_result < 0  # make sure opposite parity
    if left_result > right_result:
        bounds = [right, left]

    # Optimize here
    isclose_kwargs = {x: kwargs[x] for x in ("tol", "tol_type") if x in kwargs}
    while True:
        midpoint = sum(bounds)/2
        if isclose(midpoint, left, tol=1e-9, tol_type="absolute") \
                or isclose(midpoint, right, tol=1e-9, tol_type="absolute"):
            raise ValueError("Converged to extreme values - likely wrong inputs...")

        result = f(midpoint)
        if isclose(result, target, **isclose_kwargs):
            return midpoint
        
        if result > target:
            bounds[1] = midpoint
        else:
            bounds[0] = midpoint

