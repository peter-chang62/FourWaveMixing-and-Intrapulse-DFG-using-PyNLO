import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
from pynlo.media.crystals.XTAL_PPLN import DengSellmeier

ppln = DengSellmeier(24.5)
c = 299792458  # m/s


# smaller wavelength, higher wavelength -> phase mismatch in 1/um
def phase_mismatch(wl1_um, wl2_um):
    wl1_nm = wl1_um * 1e3
    wl2_nm = wl2_um * 1e3
    n1 = ppln.n(wl1_nm)
    n2 = ppln.n(wl2_nm)
    k1 = 2 * np.pi * n1 / wl1_um
    k2 = 2 * np.pi * n2 / wl2_um

    wl3_um = 1 / ((1 / wl1_um) - (1 / wl2_um))
    wl3_nm = wl3_um * 1e3
    n3 = ppln.n(wl3_nm)
    k3 = 2 * np.pi * n3 / wl3_um
    return k1 - k2 - k3, wl3_um


dk, dfgwl = phase_mismatch(1, 1.5)
