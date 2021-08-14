from Fiber_PPLN_NLSE import *
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import scipy.constants as sc
import scipy.interpolate as spi

clipboard_and_style_sheet.style_sheet()

normalize = lambda vec: vec / np.max(abs(vec))


def get_data(path=None):
    if path is None:
        freq_data = np.genfromtxt("Recunstructed FROG_ 21fs "
                                  "pulse/ReconstructedPulseSpectrum.txt")
    else:
        freq_data = np.genfromtxt(path)

    Amp_AW = np.sqrt(freq_data[:, 0])
    phase_AW = -freq_data[:, 1]
    wl_um = freq_data[:, 2]

    f_mks = sc.c / (wl_um * 1e-6)
    gridded_f_mks = spi.interp1d(np.linspace(0, 1, len(f_mks)), f_mks)
    center_f_mks = gridded_f_mks(.5)
    center_wavelength_nm = (sc.c / center_f_mks) * 1e9

    pulse_W = Pulse(center_wavelength_nm=center_wavelength_nm)
    pulse_W.set_AW_experiment(wl_um, Amp_AW * np.exp(1j * phase_AW))

    return pulse_W


pulse = get_data()
ppln = PPLN()
ppln.generate_ppln(pulse=pulse, length_m=1e-3,
                   center_wl_nm=pulse.center_wavelength_nm,
                   poling_period_um=np.linspace(27.5, 31.6, 5000),
                   r_um=18.)

sim = PPLNThreeWaveMixing()
sim_output = sim.propagate(pulse, ppln, 500)

fig, ax = plt.subplots(1, 1)
ind = (pulse.wl_um >= 0).nonzero()
ax.semilogy(pulse.wl_um[ind], normalize(abs(sim_output.pulse.AW[ind]) ** 2))
ax.set_xlim(1, 6)
ax.set_ylim(1e-6, 1)
