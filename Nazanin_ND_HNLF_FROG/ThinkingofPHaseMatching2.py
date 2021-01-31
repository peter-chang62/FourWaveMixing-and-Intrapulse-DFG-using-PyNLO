import numpy as np
from pynlo.media.crystals.XTAL_PPLN import DengSellmeier
import matplotlib.pyplot as plt
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()


def refractive_index(wl_um):
    return DengSellmeier(24.5).n(wl_um * 1e3)


class DFGPHaseMismatch:
    c = 299792458

    def __init__(self, wl1_um, wl2_um):
        """
        wl1_um: shorter wavelength
        wl2_um: longer wavelength
        """
        self.wl1_um = wl1_um
        self.wl2_um = wl2_um
        self.wl3_um = ((1 / wl1_um) - (1 / wl2_um)) ** -1

        self.n1 = refractive_index(self.wl1_um)
        self.n2 = refractive_index(self.wl2_um)
        self.n3 = refractive_index(self.wl3_um)

        self.k1 = self.n1 / self.wl1_um
        self.k2 = self.n2 / self.wl2_um
        self.k3 = self.n3 / self.wl3_um

        self.dk = self.k1 - self.k2 - self.k3

        self.v1 = self.c / self.n1
        self.v2 = self.c / self.n2
        self.v3 = self.c / self.n3

    def plot1D_dfg_wl(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(self.wl1_um, self.wl3_um)
        ax.set_xlabel("wavelength ($\mathrm{\mu m}$)")
        ax.set_ylabel("DFG wavelength ($\mathrm{\mu m}$)")

    def plot1D_dk(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(self.wl1_um, self.dk)
        ax.set_ylabel("phase mismatch (1/$\mathrm{\mu m}$)")
        ax.set_xlabel("wavelength ($\mathrm{\mu m}$)")

    def plot1D_walkoff(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.wl1_um, 1e-3 * (1 / self.v1 - 1 / self.v2) * 1e15)
        ax.set_xlabel("wavelength ($\mathrm{\mu m}$)")
        ax.set_ylabel("walk-off after 1mm (fs)")

    def get_limit_wls1D(self, wl1_um, wl2_um):
        ind = np.where(np.logical_and(self.wl3_um >= wl1_um, self.wl3_um <= wl2_um))
        limited_class = DFGPHaseMismatch(self.wl1_um[ind], self.wl2_um)
        return limited_class, self.wl1_um[ind][[0, -1]]


wl2 = 1.56
wl1 = np.linspace(0.75, 1.25, 1000)
run = DFGPHaseMismatch(wl1, wl2)

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
# run.plot1D_dfg_wl(ax1)
# run.plot1D_walkoff(ax2)
# run.plot1D_dk(ax3)

limit, (ll, ul) = run.get_limit_wls1D(3, 5)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
limit.plot1D_dfg_wl(ax1)
limit.plot1D_walkoff(ax2)
limit.plot1D_dk(ax3)
