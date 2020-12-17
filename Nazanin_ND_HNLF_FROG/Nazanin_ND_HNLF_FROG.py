import numpy as np
import matplotlib.pyplot as plt
import Fiber_PPLN_NLSE as fpn
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()


def normalize(vec):
    return vec / np.max(abs(vec))


path = 'C:/Users/pchan/Documents/Research Projects/PyNLO_project_for_new_system_design/Nazanin_FROG_pulse' \
       '/Recunstructed FROG_ 21fs pulse/'

data_temp = np.genfromtxt(path + 'ReconstructedPulseTemporal.txt')
data_spec = np.genfromtxt(path + 'ReconstructedPulseSpectrum.txt')
center_wavelength_nm = data_spec[:, 2][len(data_spec) // 2] * 1e3


# don't forget to negate the phase
def get_data(string):
    pulse = fpn.Pulse(center_wavelength_nm=center_wavelength_nm)
    if string == 'temp':
        # data_temp = np.genfromtxt(path + 'ReconstructedPulseTemporal.txt')
        amp = data_temp[:, 0]
        phase = - data_temp[:, 1]
        T_ps = data_temp[:, 2] / 1000
        AT = amp * np.exp(1j * phase)
        pulse.set_AT_experiment(T_ps, AT)
    elif string == 'spec':
        # data_spec = np.genfromtxt(path + 'ReconstructedPulseSpectrum.txt')
        lamda = data_spec[:, 2]
        phase = - data_spec[:, 1]
        amp = data_spec[:, 0]
        AW = amp * np.exp(1j * phase)
        pulse.set_AW_experiment(lamda, AW)
    else:
        raise ValueError('string should be either temp or spec')
    return pulse


class Evol:
    def __init__(self, evol):
        self.pulse = evol.pulse
        self.AW2d = evol.AW
        self.zs = evol.zs * 1e2

        toplot = abs(self.AW2d) ** 2
        toplot = (toplot.T / np.max(toplot, axis=1)).T
        self.toplot = toplot

    def plot_2dwindow(self, ll_um, ul_um, ax=None):
        cond = np.logical_and(self.pulse.wl_um >= ll_um, self.pulse.wl_um <= ul_um)
        ind = np.where(cond)
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.pcolormesh(self.pulse.wl_um[ind], self.zs, self.toplot.T[ind].T, shading='auto', cmap='jet')
        ax.set_xlabel('wavelength ($\mathrm{\mu m}$)')
        ax.set_ylabel("a.u.")

    def plot_1dwindow(self, ax=None):
        ind = np.where(self.pulse.wl_um >= 0)
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.semilogy(self.pulse.wl_um[ind], normalize(abs(self.pulse.AW[ind]) ** 2))
        ax.set_xlim(1, 6)
        ax.set_ylim(1e-6, 1)
        ax.set_xlabel('wavelength ($\mathrm{\mu m}$)')
        ax.set_ylabel("a.u.")


pulse = get_data('spec')

# it checks out
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.plot(pulse_temp.F_THz, normalize(abs(pulse_temp.AW) ** 2))
# ax1.plot(300 / data_spec[:, 2], normalize(data_spec[:, 0]))
# ax2.plot(pulse_spec.T_ps, normalize(abs(pulse_spec.AT) ** 2))
# ax2.plot(data_temp[:, 2] / 1000, data_temp[:, 0])

ppln = fpn.PPLN()
ppln.generate_ppln(pulse, .001, 1550, np.linspace(27.5, 31.6, 5000))

ssfm = fpn.PPLNThreeWaveMixing()
sim_ppln = ssfm.propagate(pulse, ppln, 100)

evol = Evol(sim_ppln)
fig, (ax1, ax2) = plt.subplots(1, 2)
evol.plot_1dwindow(ax1)
evol.plot_2dwindow(3, 5, ax2)
