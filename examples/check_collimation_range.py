#!/usr/bin/env python3
"""Determines collimation range for specific focal lengths from fiber.
"""

import numpy as np

import physicsutils.models.gaussian_beam as gaussian

# 1310nm from SMF-28
wl = 1310e-9
mfd = 9.2e-6
beam = gaussian.Beam.from_params(wl, mfd/2)

# C220
c230_f = 4.51e-3
c430_f = 5e-3
c220_f = 11e-3
c560_f = 13.86e-3

# Using C220 lens,
# waist is 997.1325um @ 12.6mm when d=11mm-200nm
# waist is 997.1430um @   22mm when d=11mm
# This difference (10mm) is miniscule if Rayleigh range is > cm order.

def _(lens_name, lens_f, show=False):
    b = beam.copy().apply_thin_lens(lens_f, lens_f)
    zr = np.pi*b.get_waist()[1]**2/wl
    b.plot(z1=lens_f+zr, xunit="cm", show=show)
    print(f"Rayleigh range for collimated 1310nm w/ {lens_name} lens = {zr*100:.1f} cm")

_("C230", c230_f)
_("C430", c430_f)
_("C220", c220_f)
_("C560", c560_f)

"""
Rayleigh range for collimated 1310nm, C230 lens: 40.1 cm
Rayleigh range for collimated 1310nm, C430 lens: 49.3 cm
Rayleigh range for collimated 1310nm, C220 lens: 238.4 cm
Rayleigh range for collimated 1310nm, C560 lens: 378.6 cm
"""