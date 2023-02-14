#!/usr/bin/env python3
# Justin, 2022-03-10

"""
Calculates key rate according to [1], assumes balanced HV and DA errors.

Note:
    To run doctests, issue in command line: "python -m doctest keyrate.py"

References:

    [1] Neumann, Sebastian Philipp, Thomas Scheidl, Mirela Selimovic, Matej Pivoluska, Bo Liu, Martin Bohmann, and Rupert Ursin. 'Model for Optimizing Quantum Key Distribution with Continuous-Wave Pumped Entangled-Photon Sources'. Physical Review A 104, no. 2 (5 August 2021): 022406. https://doi.org/10.1103/PhysRevA.104.022406.

    [2] Gisin, Nicolas, Grégoire Ribordy, Wolfgang Tittel, and Hugo Zbinden. 'Quantum Cryptography'. Reviews of Modern Physics 74, no. 1 (8 March 2002): 145-95. https://doi.org/10.1103/RevModPhys.74.145.

    [3] Elkouss, David, Jesus Martinez-Mateo, and Vicente Martin. 'Information Reconciliation for Quantum Key Distribution'. ArXiv:1007.1616 [Quant-Ph], 1 April 2011. http://arxiv.org/abs/1007.1616.

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special

def eta_tCC(t_CC, t_delta):
    """Proportion of true coincidences falling in coincidence window, see [1.13].

    't_CC' and 't_delta' should share the same units.

    Args:
        t_CC: Width of coincidence window.
        t_delta: Timing uncertainty of measurement.
    Note:
        't_delta' is the convolution of detector jitter, chromatic dispersion, and
        coherence time of photons.
    Tests:
        >>> eta_tCC(0, 10) == 0
        True
    """
    return scipy.special.erf(np.sqrt(np.log(2)) * t_CC/t_delta)

def H2(x):
    """Binary entropy, see [1.18].
    
    Args:
        x: Real number within [0,1].
    Tests:
        >>> 1 - 2.1*H2(0.103) < 0
        True
        >>> 1 - 2.1*H2(0.101) > 0
        True
    """
    x = np.array(x)  # cast to numpy array to account for NaN
    xn = (0<x) & (x<1)  # check if within [0,1]
    return np.where(xn, -x*np.log2(x,where=xn) - (1-x)*np.log2(1-x,where=xn), 0)

def Rs(CC_m, E_bit, E_phase, f_Ebit, q):
    """See [1.17].

    Simplifies to [1.19], assuming q=0.5, f=1.1, E_bit=E_phase=QBER. q=1 for
    more efficient/asymmetric BB84 encoding scheme.

    Note:
        Typical protocols use random, uniformly selected basis choices, e.g.
        passive basis choice using beamsplitter. In such cases, q is thus 0.5.
    Tests:
        >>> Rs(100, 0.01, 0.01, 1.1, 0.5) == 0.5*100*( 1 - 2.1*H2(0.01) )  # [1.19]
        True
    """
    return q*CC_m*( 1 - f_Ebit*H2(E_bit) - H2(E_phase) )


##############################
#   MEASUREMENT PARAMETERS   #
##############################

# Afterpulse effects ignored

# Detector dark counts, per second
Sdark_alice = 100
Sdark_bob = 100

# Detector efficiency, e.g. SNSPD
eff_det_alice = 0.8
eff_det_bob = 0.8

# Coincidence window size, vs timing uncertainty
t_uncertainty = 100e-12  # 0.1ns
CC_window = 2e-9  # 2ns

# Privacy amplification efficiency
#   Typical efficiencies of CASCADE is roughly f = 1.15 for low QBER, to around
#   f = 1.2 for high QBER >= 6%. LDPC around f = 1.1. [3]
f = 1.15

#########################
#   SOURCE PARAMETERS   #
#########################

# Source heralding efficiency
eff_heralding = 0.8

# Source optical coupling, e.g. into fiber
eff_optical = 0.6

# Visibility & error due to polarization mismatch
visibility = 0.99

# Brightness, i.e. number of photon pairs generated
B = 500e3

# Link efficiency, i.e. transmission through fiber
#     Assume -0.5 dB/km @ 1310nm for single mode
#     https://www.thefoa.org/tech/ref/testing/test/loss.html
link_length_alice = 20  # 1 km
link_length_bob = 20
link_loss = -0.5  # dB/km

####################
#   CALCULATIONS   #
####################

def get_keyrate(
        link_length_bob,
    ):
    """Calculates key rate.
    
    Parameters are specified as global (environment) variables. These can be
    dynamically changed by either changing the environment variable before calling
    the function, or adding arguments to this function.
    """

    # Error due to polarization outcome mismatch
    #
    #   Commentary in [1.A]:
    #     'e_pol' obtained experimentally by reducing accidental counts to near 0,
    #     which is equivalent to measuring the visibility of the source, factoring
    #     in (1) source visibility, (2) optical component imperfections,
    #     (3) polarization compensation.
    #
    #     According to [1.16] and [2.34], we therefore get: e_pol = E = (1-V)/2
    #
    e_pol = (1-visibility) / 2

    # Estimated channel efficiency, assume symmetric Alice and Bob
    #   Contributions:
    #     Source heralding efficiency, channel losses, detection efficiencies
    link_length_alice = link_length_bob
    eff_link_alice = 10**( link_loss*link_length_alice /10)
    eff_link_bob = 10**( link_loss*link_length_bob /10)
    eff_alice = eff_heralding * eff_optical * eff_link_alice * eff_det_alice
    eff_bob = eff_heralding * eff_optical * eff_link_bob * eff_det_bob

    # Singles rate measured by Alice and Bob, per second
    S_alice = B * eff_alice + Sdark_alice
    S_bob = B * eff_bob + Sdark_bob

    # Accidental coincidence rates
    CC_acc = S_alice * S_bob * CC_window  # accidentals

    # True coincidence rates, see [1.3]
    CC_true = B * eff_alice * eff_bob

    # True coincidence rates within coincidence window
    # - Note 'CC_sift' is equivalent to the following:
    #     (measured coincidences - accidentals)
    CC_sift = eta_tCC(CC_window, t_uncertainty) * CC_true

    # Measured coincidence rates
    CC_meas = CC_sift + CC_acc  # [1.14]

    # Error in coincidence measurement due to polarization outcome mismatch
    CC_error = CC_sift * e_pol + 0.5 * CC_acc  # [1.15]
    
    # QBER
    # Assume same errors for HV and DA basis, otherwise see [1.B§4]
    QBER = CC_error / CC_meas  # [1.16]
    keyrate = Rs(CC_meas, QBER, QBER, f, 0.5)

    return keyrate


####################
#   CALCULATIONS   #
####################

def reproduce_Ref1_Figure2b():
    # Parameter overrides to generate graph in [1.Fig4]
    # https://v1.cecdn.yun300.cn/100001_2104165105/TimeTagger-SI.pdf

    def get_keyrate(B):
        """ Modified from global 'get_keyrate'. """
        S_alice = B * eff_alice + Sdark_alice
        S_bob = B * eff_bob + Sdark_bob
        CC_acc = S_alice * S_bob * CC_window  # accidentals
        CC_true = B * eff_alice * eff_bob
        CC_sift = eta_tCC(CC_window, t_uncertainty) * CC_true
        CC_meas = CC_sift + CC_acc  # [1.14]
        CC_error = CC_sift * e_pol + 0.5 * CC_acc  # [1.15]
        QBER = CC_error / CC_meas  # [1.16]
        keyrate = Rs(CC_meas, QBER, QBER, f, 0.5)
        return keyrate

    f = 1.1
    params = (
        (20, 20, 0.0121, 134e-12, "#0067B3"),
        (30, 20, 0.0535,  93e-12, "#8ACB33"),
        (40, 20, 0.0295,  86e-12, "#FF9300"),
        (30, 30, 0.0182,  76e-12, "#00D7E0"),
        (40, 40, 0.0401,  46e-12, "#83A2FF"),
    )
    for loss_alice, loss_bob, e_pol, CC_window, color in params:
        t_uncertainty = CC_window  # note still need to convolute with SNSPD jitter
        eff_alice = 10**(-loss_alice/10)
        eff_bob = 10**(-loss_bob/10)

        brightnesses = np.linspace(0, 2.7e9, 1000)
        keyrates = get_keyrate(brightnesses)
        plt.plot(brightnesses, keyrates, color)

    plt.grid()
    plt.yscale("log")
    plt.ylabel("Secure Key Rate (cps)")
    plt.xlabel("Brightness (cps)")
    plt.show()

def estimate_QBER():

    def get_keyrate(eff_link):
        """ Modified from global 'get_keyrate'. """
        eff_link_alice = 1  # assume source with Alice
        eff_link_bob = 10**( -eff_link /10)
        eff_alice = eff_heralding * eff_optical * eff_link_alice * eff_det_alice
        eff_bob = eff_heralding * eff_optical * eff_link_bob * eff_det_bob
        S_alice = B * eff_alice + Sdark_alice
        S_bob = B * eff_bob + Sdark_bob
        CC_acc = S_alice * S_bob * CC_window  # accidentals
        CC_true = B * eff_alice * eff_bob
        CC_sift = eta_tCC(CC_window, t_uncertainty) * CC_true
        CC_meas = CC_sift + CC_acc  # [1.14]
        CC_error = CC_sift * e_pol + 0.5 * CC_acc  # [1.15]
        QBER = CC_error / CC_meas  # [1.16]
        keyrate = Rs(CC_meas, QBER, QBER, f, 0.5)
        return QBER, keyrate

    f = 1.1
    B = 500e3
    visibility = 0.99
    e_pol = (1-visibility)/2
    eff_optical = 0.6
    eff_heralding = 0.8
    CC_window = 0.5e-9  # 1/2 ns
    t_uncertainty = 0.1e-9
    params = (
        (0.15, 0.15, 240000, 40000),
        (0.15, 0.9, 240000, 200),
        (0.9, 0.9, 200, 200),
    )
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    errors_link = np.linspace(0, 70, 10000)
    for eff_det_alice, eff_det_bob, Sdark_alice, Sdark_bob in params:
        QBER, keyrates = get_keyrate(errors_link)
        # plt.plot(errors_link, keyrates)
        plt.sca(ax)
        plt.plot(errors_link, keyrates,
            label=f"$\eta_A={eff_det_alice:.2f}, \eta_B={eff_det_bob:.2f}, S^{'{dark}'}_A=$ {Sdark_alice:.1e}$, S^{'{dark}'}_B=$ {Sdark_bob:.1e}")
        plt.sca(ax2)
        plt.plot(errors_link, 100*QBER, "--")

    plt.sca(ax)
    plt.legend(loc="upper right")
    plt.grid()
    plt.yscale("log")
    plt.ylabel("Secure Key Rate (cps)")
    plt.xlabel("Link loss (dB)")
    plt.xlim(left=0, right=np.max(errors_link))

    plt.sca(ax2)
    plt.ylim(top=50, bottom=0)
    plt.ylabel("QBER (%)")

    plt.title(
        f"Brightness = ${B*1e-3:.0f}$ kcps, Visibility = ${visibility:.3f}$\n"
        f"$\eta_{'{optical}'}={eff_heralding*eff_optical:.2f}, t_{'{CC}'}={CC_window*1e12:.0f}$ ps"
    )
    plt.show()

def get_qber_symmetric_attack():
    qber = np.linspace(0, 0.5, 1000)
    eve_info = 1 - H2(0.5+np.sqrt(qber*(1-qber))) # see [2.63]
    bob_info = 1 - H2(qber)
    plt.plot(qber, eve_info, "r")
    plt.plot(qber, bob_info, "b")
    plt.show()

print(estimate_QBER())
raise

# Plot graph of key rate against link length
link_lengths = np.linspace(0, 100, 1000)
keyrates = get_keyrate(link_lengths)

plt.plot(link_lengths, keyrates)
plt.yscale("log")
plt.grid()
plt.show()
