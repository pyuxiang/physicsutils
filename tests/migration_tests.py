# Add checking directories
import pytest
import sys
from pathlib import Path
sys.path.append(str(Path().absolute().parent))

import nonlinearCrystal1 as nlc_orig
import crystal as nlc
import numpy as np

def is_close(a, b, tol=1e-7):
    return abs(a-b) < tol

# @pytest.mark.parametrize("a", [1,3])
# @pytest.mark.parametrize("b", [1,3])
# @pytest.mark.parametrize("wl", [4,5])
# def test_partFrac(a, b, wl):
#     assert nlc_orig.partFrac(a,b,wl) == nlc.part_frac(a,b,wl)

def test_nlc_trigDegree():
    assert is_close(nlc.sind(0), 0)
    assert nlc.sind(90) == 1
    assert nlc.cosd(0) == 1
    assert is_close(nlc.cosd(90), 0)

def test_nlc_refractiveIndex():
    # https://www.mt-optics.net/a-BBO.html
    length = 10; theta = 45; axis = "v";
    args = [length, theta, axis]
    c = nlc.AlphaBBO(*args, model="MTOptics")
    assert is_close(1.65790, c.index(1.064, 0), 1e-5)
    assert is_close(1.58462, c.index(1.064, 45), 1e-5)

    # https://www.pmoptics.com/files/Beam_Displacer.pdf
    # Test beam separation for normal incidence
    test_wl = 1.55
    tan_theta = np.tan(theta*np.pi/180)
    sq_no_ne = (c._index_no(test_wl) / c._index_ne(test_wl))**2
    tan_rho = (1-sq_no_ne) * tan_theta / (1 + sq_no_ne*tan_theta**2)
    assert is_close(length*tan_rho, c.beam_separation(test_wl, alpha=0), 1e-5)

    # https://www.agoptics.com/Alpha-BBO.html
    c = nlc.AlphaBBO(*args, model="AgOptics")
    assert is_close(1.65790, c.index(1.064, 0), 1e-5)
    assert is_close(1.58462, c.index(1.064, 45), 1e-5)

    # https://www.agoptics.com/Alpha-BBO.html
    c = nlc.AlphaBBO(*args, model="Castech")
    assert is_close(1.6776, c.index(0.532, 0), 1e-4)
    assert is_close(1.5534, c.index(0.532, 90), 1e-4)

    # https://www.castech.com/product/YVO4-92.html
    c = nlc.YVO4(*args, model="Castech")
    assert is_close(1.9500, c.index(1.300, 0), 1e-4)
    assert is_close(2.1554, c.index(1.300, 90), 1e-4)

    # https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
    c = nlc.FusedQuartz(*args, model="Malitson")
    assert is_close(1.4585, c.index(0.5876, 0), 1e-4)

    # https://refractiveindex.info/?shelf=main&book=SiO2&page=Ghosh-o
    c = nlc.Quartz(*args, model="Ghosh")
    assert is_close(1.5443, c.index(0.5876, 0), 1e-4)
    assert is_close(1.5534, c.index(0.5876, 90), 1e-4)

    # https://refractiveindex.info/?shelf=main&book=SiO2&page=Radhakrishnan-o
    c = nlc.Quartz(*args, model="Radhakrishnan")
    assert is_close(1.5443, c.index(0.5876, 0), 1e-4)
    assert is_close(1.5534, c.index(0.5876, 90), 1e-4)

    # https://refractiveindex.info/?shelf=main&book=MgF2&page=Li-o
    c = nlc.MgF2(*args, model="Li")
    assert is_close(1.3777, c.index(0.5876, 0), 1e-4)
    assert is_close(1.3895, c.index(0.5876, 90), 1e-4)

    # https://refractiveindex.info/?shelf=main&book=MgF2&page=Dodge-o
    c = nlc.MgF2(*args, model="Dodge")
    assert is_close(1.3777, c.index(0.5876, 0), 1e-4)
    assert is_close(1.3896, c.index(0.5876, 90), 1e-4)

def test_nlc_phase():
    # Uses MTOptics refractive index coefficients
    bbo_orig = nlc_orig.NonlinearCrystal(l=1e3, material="ABBO", theta=45, oaOrientation="up")
    bbo = nlc.AlphaBBO(length=1, theta=45, axis="v", model="MTOptics")

    # Check refractive indices
    assert bbo.index(1.32) == bbo_orig.calcN(1.32, bbo_orig.theta)

    # Check walkoff angle for 'e' polarized beam at normal incidence
    walkoff_angle = bbo_orig.calcWalkoff(1.32, bbo_orig.theta)
    assert is_close(-bbo.walkoff(1.32, "e", alpha=0), walkoff_angle)

    # Check walkoff angle for 'o' polarized beam at normal incidence
    assert bbo.walkoff(1.32, "o", alpha=0) == 0
    assert is_close(-bbo.beam_separation(1.32), np.tan(np.deg2rad(walkoff_angle))*bbo_orig.l*1e-3, 1e-5)
    
    # Check phase calculations are same for normal incidence
    assert bbo_orig.calcPhase(1.32, pol="e", alpha=0, gamma=0) == bbo.phase(1.32, pol="e", alpha=0, gamma=0)
    assert bbo_orig.calcPhase(1.32, pol="o", alpha=0, gamma=0) == bbo.phase(1.32, pol="o", alpha=0, gamma=0)

    # Check phase calculations are consistent for rotated frames
    bbo_h = nlc.AlphaBBO(length=1, theta=45, axis="h", model="MTOptics")
    assert bbo.phase(1.32, pol="e", alpha=10, gamma=23) == bbo_h.phase(1.32, pol="e", alpha=23, gamma=10)
    bbo_orig_h = nlc_orig.NonlinearCrystal(l=1e3, material="ABBO", theta=45, oaOrientation="right")
    # assert bbo_orig.calcPhase(1.32, pol="e", alpha=10, gamma=23) == bbo_orig_h.calcPhase(1.32, pol="e", alpha=23, gamma=10)  # this one fails
    
    # Check phase calculations are of same order of magnitude for non-zero alpha
    r1 = bbo_orig.calcPhase(1.32, pol="e", alpha=10, gamma=0)
    r2 = bbo.phase(1.32, pol="e", alpha=10, gamma=0)
    ratio = r1/r2 if r1 < r2 else r2/r1
    assert 0.9 < ratio < 1

def test_nlc_polarizations():
    yvo4_kw = {
        "length": 10,
        "theta": 45,
        "model": "Castech",
    }
    q_kw = {
        "length": 20,
        "theta": 34,
        "model": "Ghosh",
    }
    phase_kw = {
        "wl": 1.200,
        "alpha": 20,
        "gamma": 10,
    }
    walkoff_kw = {
        "wl": 1.310,
        "alpha": 15,
    }

    # Test phase
    c = nlc.YVO4(**yvo4_kw, axis = "v")
    assert c.phase(pol="e", **phase_kw) == c.phase(pol="v", **phase_kw)
    assert c.phase(pol="o", **phase_kw) == c.phase(pol="h", **phase_kw)

    c = nlc.Quartz(**q_kw, axis="h")
    assert c.phase(pol="e", **phase_kw) == c.phase(pol="h", **phase_kw)
    assert c.phase(pol="o", **phase_kw) == c.phase(pol="v", **phase_kw)

    # Test walkoff
    c = nlc.YVO4(**yvo4_kw, axis = "v")
    assert c.walkoff(pol="e", **walkoff_kw) == c.walkoff(pol="v", **walkoff_kw)
    assert c.walkoff(pol="o", **walkoff_kw) == c.walkoff(pol="h", **walkoff_kw)

    c = nlc.Quartz(**q_kw, axis="h")
    assert c.walkoff(pol="e", **walkoff_kw) == c.walkoff(pol="h", **walkoff_kw)
    assert c.walkoff(pol="o", **walkoff_kw) == c.walkoff(pol="v", **walkoff_kw)
    