import numpy as np
from pynlo.media.crystals.XTAL_PPLN import DengSellmeier
import matplotlib.pyplot as plt
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()

c = 299792458


def refractive_index(wl_um):
    return DengSellmeier(24.5).n(wl_um * 1e3)


class DFGPHaseMismatch:
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

        self.v1 = c / self.n1
        self.v2 = c / self.n2
        self.v3 = c / self.n3

    def plot1d_dfg_wl(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(self.wl1_um, self.wl3_um)
        ax.set_xlabel("wavelength ($\mathrm{\mu m}$)")
        ax.set_ylabel("DFG wavelength ($\mathrm{\mu m}$")

    def plot1d_dk(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(self.wl1_um, self.dk)
        ax.set_ylabel("phase mismatch (1/$\mathrm{\mu m}$)")
        ax.set_xlabel("wavelength ($\mathrm{\mu m}$)")

    def plot1d_walkoff(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.wl1_um, (1e-3 * 1e9 / abs(self.v1 - self.v2)))
        ax.set_xlabel("wavelength ($\mathrm{\mu m}$)")
        ax.set_ylabel("walk-off after 1mm (ns)")


wl2 = 1.56
wl1 = np.linspace(0.75, 1.25, 1000)
run = DFGPHaseMismatch(wl1, wl2)

run.plot1d_dfg_wl()
run.plot1d_walkoff()
run.plot1d_dk()
