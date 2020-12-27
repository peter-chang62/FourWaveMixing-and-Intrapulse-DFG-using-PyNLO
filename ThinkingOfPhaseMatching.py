import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
from pynlo.media.crystals.XTAL_PPLN import DengSellmeier

clipboard_and_style_sheet.style_sheet()

ppln = DengSellmeier(24.5)
c = 299792458  # m/s


# smaller wavelength, higher wavelength -> phase mismatch in 1/um
def phase_mismatch(wl1_um, wl2_um):
    wl1_nm = wl1_um * 1e3
    wl2_nm = wl2_um * 1e3
    n1 = ppln.n(wl1_nm)
    n2 = ppln.n(wl2_nm)
    k1 = n1 / wl1_um
    k2 = n2 / wl2_um

    wl3_um = 1 / ((1 / wl1_um) - (1 / wl2_um))
    wl3_nm = wl3_um * 1e3
    n3 = ppln.n(wl3_nm)
    k3 = n3 / wl3_um
    return k1 - k2 - k3, wl3_um


wl2 = 1.56  # 1560 nm
wl1 = np.linspace(.7, 1.25, 5000)
dk, dfgwl = phase_mismatch(wl1, wl2)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.plot(wl1, dk)
ax2.plot(wl1, dfgwl)
# ax3.plot(wl1, ppln.n(wl1 * 1e3))
ax3.plot(wl1, 1 / dk)

[i.set_xlabel("wavelength ($\mathrm{\mu m}$)") for i in [ax1, ax2, ax3]]
ax1.set_ylabel("phase mismatch ($1/\mathrm{\mu m}$)")
ax2.set_ylabel("dfg wavelength ($\mathrm{\mu m}$)")
# ax3.set_ylabel("PPLN index of refraction")
ax3.set_ylabel("required poling ($\mathrm{\mu m}$)")
fig.suptitle("phase mismatch between 1560 nm and shorter wavelengths")

ll = np.argmin(abs(dfgwl - 3))
ul = np.argmin(abs(dfgwl - 5))
ax1.axhline(dk[ll], color='r', linestyle='--')
ax1.axhline(dk[ul], color='r', linestyle='--')
ax2.axhline(dfgwl[ll], color='r', linestyle='--')
ax2.axhline(dfgwl[ul], color='r', linestyle='--')
ax3.axhline(1 / dk[ll], color='r', linestyle='--')
ax3.axhline(1 / dk[ul], color='r', linestyle='--')

ax1.axvline(wl1[ll], color='r', linestyle='--')
ax1.axvline(wl1[ul], color='r', linestyle='--')
ax2.axvline(wl1[ll], color='r', linestyle='--')
ax2.axvline(wl1[ul], color='r', linestyle='--')
ax3.axvline(wl1[ll], color='r', linestyle='--')
ax3.axvline(wl1[ul], color='r', linestyle='--')
