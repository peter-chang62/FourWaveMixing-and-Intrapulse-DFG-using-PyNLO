"""Convenient functions for Fiber SPM (four wave mixing) simulations """

import matplotlib.pyplot as plt
import numpy as np
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import copy


def normalize(vec):
    return vec / np.max(abs(vec))


# dB/km to 1/m
def dBkm_to_m(dBkm):
    km = 10 ** (-dBkm / 10)
    return km * 1e-3


def simulate(pulse, fiber, length_cm, epp_nJ, nsteps=100):
    pulse: fpn.Pulse
    fiber: fpn.Fiber
    _ = copy.deepcopy(fiber)
    _.length = length_cm * .01
    __ = copy.deepcopy(pulse)
    __.set_epp(epp_nJ * 1.e-9)
    return fpn.FiberFourWaveMixing().propagate(__, _, nsteps)


# ________________________________________________ Plot Functions ______________________________________________________

def get_2d_time_evolv(at2d):
    norm = np.max(abs(at2d) ** 2, axis=1)
    toplot = abs(at2d) ** 2
    toplot = (toplot.T / norm).T
    return toplot


def plot_freq_evolv(sim, ax=None, xlims=None):
    evolv = fpn.get_2d_evolv(sim.AW)

    ind = (sim.pulse.wl_um > 0).nonzero()

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.pcolormesh(sim.pulse.wl_um[ind], (sim.zs * 100.), evolv[:, ind][:, 0, :],
                  cmap='jet',
                  shading='auto')

    if xlims is None:
        # ax.set_xlim(1, 2)
        pass
    else:
        ax.set_xlim(*xlims)
    ax.set_xlabel("$\mathrm{\mu m}$")
    ax.set_ylabel("cm")


def plot_time_evolv(sim, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    toplot = get_2d_time_evolv(sim.AT)
    ax.pcolormesh(sim.pulse.T_ps, (sim.zs * 100.), toplot, cmap='jet',
                  shading='auto')
    # ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel("ps")
    ax.set_ylabel("cm")


def plot_cross_section(sim, z_cm, xlims=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    z = z_cm * 1e-2
    ind = np.argmin(abs(sim.zs - z))
    spec = sim.AW[ind].__abs__() ** 2

    if xlims is not None:
        ll, ul = xlims
        ll, ul = np.argmin(abs(sim.pulse.wl_um - ul)), np.argmin(abs(sim.pulse.wl_um - ll))

    else:
        ll, ul = 0, -1

    ax.plot(sim.pulse.wl_um[ll:ul], spec[ll:ul])


# ________________________________________________Fiber Paramters_______________________________________________________

# Pooja said that these matched her experimental results better
adhnlf = {
    "D": 4.88,
    "Dprime": 0.0228,
    "gamma": 10.9,
    "Alpha": 0.74,
}

# other AD-HNLF (from Rieker group, got parameters from Nazanin)
adhnlf_2 = {
    "D": 2.2,
    "Dprime": 0.026,
    "gamma": 10.5,
    "Alpha": 0.78,
}

# OFS ND HNLF parameters
ndhnlf = {
    "D": -2.6,
    "Dprime": 0.026,
    "gamma": 10.5,
    "Alpha": 0.8,
}

pm1550 = {
    "D": 18,
    "Dprime": 0.0612,
    "gamma": 1.,
    "Alpha": 0.18
}

fiber_adhnlf = fpn.Fiber()
fiber_adhnlf.generate_fiber(.2,
                            1550.,
                            [adhnlf["D"], adhnlf["Dprime"]],
                            adhnlf["gamma"] * 1e-3,
                            gain=dBkm_to_m(adhnlf["Alpha"]),
                            dispersion_format="D")

fiber_adhnlf_2 = fpn.Fiber()
fiber_adhnlf_2.generate_fiber(.2,
                              1550.,
                              [adhnlf_2["D"], adhnlf_2["Dprime"]],
                              adhnlf_2["gamma"] * 1e-3,
                              gain=dBkm_to_m(adhnlf_2["Alpha"]),
                              dispersion_format="D")

fiber_ndhnlf = fpn.Fiber()
fiber_ndhnlf.generate_fiber(.2,
                            1550.,
                            [ndhnlf["D"], ndhnlf["Dprime"]],
                            ndhnlf["gamma"] * 1e-3,
                            gain=dBkm_to_m(ndhnlf["Alpha"]),
                            dispersion_format="D")

fiber_pm1550 = fpn.Fiber()
fiber_pm1550.generate_fiber(.2,
                            1550.,
                            [pm1550["D"], pm1550["Dprime"]],
                            pm1550["gamma"] * 1e-3,
                            gain=dBkm_to_m(pm1550["Alpha"]),
                            dispersion_format="D")
