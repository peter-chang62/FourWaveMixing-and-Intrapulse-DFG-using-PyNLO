from Fiber_PPLN_NLSE import *
import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import scipy.constants as sc
import scipy.interpolate as spi

clipboard_and_style_sheet.style_sheet()

normalize = lambda vec: vec / np.max(abs(vec))

temp_data = np.genfromtxt("../Recunstructed FROG_ 21fs "
                          "pulse/ReconstructedPulseTemporal.txt")

freq_data = np.genfromtxt("../Recunstructed FROG_ 21fs "
                          "pulse/ReconstructedPulseSpectrum.txt")

Amp_AT = np.sqrt(temp_data[:, 0])
phase_AT = -temp_data[:, 1]
T_ps = temp_data[:, 2] * 1e-3

Amp_AW = np.sqrt(freq_data[:, 0])
phase_AW = -freq_data[:, 1]
wl_um = freq_data[:, 2]

f_mks = sc.c / (wl_um * 1e-6)
gridded_f_mks = spi.interp1d(np.linspace(0, 1, len(f_mks)), f_mks)
center_f_mks = gridded_f_mks(.5)
center_wavelength_nm = (sc.c / center_f_mks) * 1e9

pulse_T = Pulse(center_wavelength_nm=center_wavelength_nm)
pulse_T.set_AT_experiment(T_ps, Amp_AT * np.exp(1j * phase_AT))

pulse_W = Pulse(center_wavelength_nm=center_wavelength_nm)
pulse_W.set_AW_experiment(wl_um, Amp_AW * np.exp(1j * phase_AW))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=np.array([12.18, 4.8]))
ax1.plot(pulse_T.F_THz, normalize(abs(pulse_T.AW) ** 2), label="from time data")
ax1.plot(pulse_W.F_THz, normalize(abs(pulse_W.AW) ** 2), label="from freq data")
ax1.legend(loc='best')
ax1.set_xlim(150, 300)
ax1.set_xlabel("THz")

# fig, ax = plt.subplots(1, 1)
ax2.plot(pulse_T.T_ps, normalize(abs(pulse_T.AT) ** 2), label="from time data")
ax2.plot(pulse_W.T_ps, normalize(abs(pulse_W.AT) ** 2), label="from freq data")
ax2.legend(loc='best')
ax2.set_xlim(-.8, .4)
ax2.set_xlabel("ps")
"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
