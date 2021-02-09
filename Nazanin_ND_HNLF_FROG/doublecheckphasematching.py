import numpy as np
import matplotlib.pyplot as plt
from pynlo.media.crystals.XTAL_PPLN import DengSellmeier
import clipboard_and_style_sheet
from scipy.interpolate import interp1d

clipboard_and_style_sheet.style_sheet()


def refractive_index(wl_um):
    return DengSellmeier(24.5).n(wl_um * 1e3)


class SHG:
    c = 299792458

    def __init__(self, wl_um):
        self.wl_um_fun = wl_um
        self.wl_um_shg = wl_um / 2
        self.n_fun = refractive_index(self.wl_um_fun)
        self.n_shg = refractive_index(self.wl_um_shg)

        self.k_fun = self.n_fun / self.wl_um_fun
        self.k_shg = self.n_shg / self.wl_um_shg

        self.dk = 2 * self.k_fun - self.k_shg

        # calculating the walk off
        self.wl_um_axis = np.linspace(.25, 5, 10000)
        n = refractive_index(self.wl_um_axis)
        nu = self.c / (self.wl_um_axis * 1e-6)
        k = n / (self.wl_um_axis * 1e-6)
        dk_dnu = np.gradient(k, nu)
        self.v_group_axis = 1 / dk_dnu

        self.gridded_vgroup = interp1d(self.wl_um_axis, self.v_group_axis, bounds_error=True)
        self.v_fun = self.gridded_vgroup(self.wl_um_fun)
        self.v_shg = self.gridded_vgroup(self.wl_um_shg)

    def plot1D_walkoff(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.wl_um_fun, 1e-3 * abs(1 / self.v_fun - 1 / self.v_shg) * 1e15)
        ax.set_xlabel("wavelength ($\mathrm{\mu m}$)")
        ax.set_ylabel("walk-off (fs/mm)")


wl_um = np.linspace(0.8, 4, 1000)
shg = SHG(wl_um)
shg.plot1D_walkoff()
