import numpy as np
from Fiber_PPLN_NLSE import Pulse, PPLN, PPLNThreeWaveMixing, \
    FiberFourWaveMixing, Fiber
import matplotlib.pyplot as plt
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()


def normalize(vec):
    return vec / np.max(abs(vec))


def get_2d_evolv(aw2d):
    norm = np.max(abs(aw2d) ** 2, axis=1)
    toplot = abs(aw2d) ** 2
    toplot = (toplot.T / norm).T
    return toplot


# import Alex's electric field data and create a pulse instance
data = np.genfromtxt('NDAmplifier.txt')
T_ps = data[:, 2]
real = data[:, 0]
imag = data[:, 1]
AT = real + 1j * imag
pulse = Pulse(EPP_nJ=5.)
pulse.set_AT_experiment(T_ps, AT)

# create a ppln instance
ppln = PPLN()
poling_period = np.linspace(27.5, 34.5, 5000)  # um
ppln.generate_ppln(pulse, 1e-3, 1550, poling_period, 15.0)

# simulate
ssfm = PPLNThreeWaveMixing()
resppln = ssfm.propagate(pulse, ppln, 100)

# create a new pulse instance, this time a 200fs hyperbolic secant pulse
pulse = Pulse(T0_ps=200e-3)

# create a fiber
# length = 0.4 m, center_wavelength = 1550nm
# D = -1.1 ps/nmkm, Dprime = 0.027 ps/nm^2km
# alpha = 0.8 dB/km converted to 1/m
fiber = Fiber()
fiber.generate_fiber(0.4, 1550, [-1.1, 0.027],
                     10.5e-3, (- 10 ** (0.8 / 10)) * 1e-3, 'D')

# simulate
ssfm = FiberFourWaveMixing()
resfiber = ssfm.propagate(pulse, fiber, 100)

# plot the results for the fiber SPM simulation
toplot = get_2d_evolv(resfiber.AW)
plt.figure()
plt.pcolormesh(pulse.wl_um, resfiber.zs, toplot, shading='auto', cmap='jet')
plt.xlim(1.2, 1.8)

# plot the intrapulse DFG in PPLN using Alex's pulse
fig, ax = plt.subplots(1, 1)
indices = np.where(pulse.wl_nm >= 0)
ax.semilogy(pulse.wl_um[indices], normalize(
    abs(resppln.pulse.AW[indices]) ** 2))
ax.set_xlim(1, 6)
ax.set_ylim(1e-6, 1)

plt.show()
