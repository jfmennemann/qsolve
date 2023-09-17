import matplotlib.pyplot as plt

import numpy as np

from qsolve_core import D1_fourier_1d
from qsolve_core import D2_fourier_1d

from qsolve_core import lambda_D1_fourier_1d
from qsolve_core import lambda_D2_fourier_1d

from qsolve_core import D2_circulant_fd_1d

from qsolve_core import lambda_D2_circulant_fd_1d

from qsolve.visualization.colors import flat_ui_1 as colors


# from problem_1 import f, f_x, f_xx
from problem_2 import f, f_x, f_xx


x_min = -4
x_max = +11

Jx = 2 ** 10

x = np.linspace(x_min, x_max, Jx, endpoint=False)

dx = x[1] - x[0]

# ---------------------------------------------------------------------------------------------
# exact reference solution

Lx = x_max - x_min

assert (Lx == Jx * dx)

u = f(x)
u_d_ref = f_x(x)
u_dd_ref = f_xx(x)
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# solution via FFT

# Lx = Jx * dx

# ----
# lambda_d_x = lambda_D1_fourier_1d(Jx, dx)
#
# u_d_fft = np.fft.ifft(lambda_d_x * np.fft.fft(u))
#
# u_d_fft = np.real(u_d_fft)
# ----

# ----
lambda_d_xx_fourier = lambda_D2_fourier_1d(Jx, dx)

u_dd_fourier_fft = np.fft.ifft(lambda_d_xx_fourier * np.fft.fft(u))

u_dd_fourier_fft = np.real(u_dd_fourier_fft)
# ----
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# solution via Fourier differentiation matrix

D2_fourier = D2_fourier_1d(Jx, dx)

u_dd_fourier_matrix = D2_fourier @ u

print('||fft(D2_fourier[:, 0])-lambda_d_xx_fourier)||: ')
print(np.linalg.norm(np.fft.fft(D2_fourier[:, 0])-lambda_d_xx_fourier))
print()
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# solution via finite differences

# u_dd_fd = (
#     - np.roll(u, shift=-2)
#     + 16 * np.roll(u, shift=-1)
#     - 30 * u
#     + 16 * np.roll(u, shift=+1)
#     - np.roll(u, shift=+2)
#     ) / (12 * dx ** 2)

# u_dd_fd = (
#     + 2 * np.roll(u, shift=-3)
#     - 27 * np.roll(u, shift=-2)
#     + 270 * np.roll(u, shift=-1)
#     - 490 * u
#     + 270 * np.roll(u, shift=+1)
#     - 27 * np.roll(u, shift=+2)
#     + 2 * np.roll(u, shift=+3)
#     ) / (180 * dx ** 2)

# -9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9

# u_dd_fd = (
#     - 9 * np.roll(u, shift=-4)
#     + 128 * np.roll(u, shift=-3)
#     - 1008 * np.roll(u, shift=-2)
#     + 8064 * np.roll(u, shift=-1)
#     - 14350 * u
#     + 8064 * np.roll(u, shift=+1)
#     - 1008 * np.roll(u, shift=+2)
#     + 128 * np.roll(u, shift=+3)
#     - 9 * np.roll(u, shift=+4)
# ) / (5040 * dx ** 2)
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
D2_circulant_fd_2nd = D2_circulant_fd_1d(Jx, dx, order=2)
D2_circulant_fd_4th = D2_circulant_fd_1d(Jx, dx, order=4)
D2_circulant_fd_6th = D2_circulant_fd_1d(Jx, dx, order=6)
D2_circulant_fd_8th = D2_circulant_fd_1d(Jx, dx, order=8)

lambda_d_xx_circulant_fd_2nd = lambda_D2_circulant_fd_1d(Jx, dx, order=2)
lambda_d_xx_circulant_fd_4th = lambda_D2_circulant_fd_1d(Jx, dx, order=4)
lambda_d_xx_circulant_fd_6th = lambda_D2_circulant_fd_1d(Jx, dx, order=6)
lambda_d_xx_circulant_fd_8th = lambda_D2_circulant_fd_1d(Jx, dx, order=8)

u_dd_circulant_fd_2nd_fft = np.fft.ifft(lambda_d_xx_circulant_fd_2nd * np.fft.fft(u))
u_dd_circulant_fd_4th_fft = np.fft.ifft(lambda_d_xx_circulant_fd_4th * np.fft.fft(u))
u_dd_circulant_fd_6th_fft = np.fft.ifft(lambda_d_xx_circulant_fd_6th * np.fft.fft(u))
u_dd_circulant_fd_8th_fft = np.fft.ifft(lambda_d_xx_circulant_fd_8th * np.fft.fft(u))

u_dd_circulant_fd_2nd_matrix = D2_circulant_fd_2nd @ u
u_dd_circulant_fd_4th_matrix = D2_circulant_fd_4th @ u
u_dd_circulant_fd_6th_matrix = D2_circulant_fd_6th @ u
u_dd_circulant_fd_8th_matrix = D2_circulant_fd_8th @ u
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
rel_error_u_dd_fourier_matrix = np.linalg.norm(u_dd_fourier_matrix - u_dd_ref, ord=np.inf) / np.linalg.norm(u_dd_ref, ord=np.inf)
rel_error_u_dd_fourier_fft = np.linalg.norm(u_dd_fourier_fft - u_dd_ref, ord=np.inf) / np.linalg.norm(u_dd_ref, ord=np.inf)

rel_error_u_dd_circulant_fd_2nd_fft = np.linalg.norm(u_dd_circulant_fd_2nd_fft - u_dd_ref, ord=np.inf) / np.linalg.norm(u_dd_ref, ord=np.inf)
rel_error_u_dd_circulant_fd_4th_fft = np.linalg.norm(u_dd_circulant_fd_4th_fft - u_dd_ref, ord=np.inf) / np.linalg.norm(u_dd_ref, ord=np.inf)
rel_error_u_dd_circulant_fd_6th_fft = np.linalg.norm(u_dd_circulant_fd_6th_fft - u_dd_ref, ord=np.inf) / np.linalg.norm(u_dd_ref, ord=np.inf)
rel_error_u_dd_circulant_fd_8th_fft = np.linalg.norm(u_dd_circulant_fd_8th_fft - u_dd_ref, ord=np.inf) / np.linalg.norm(u_dd_ref, ord=np.inf)

rel_error_u_dd_circulant_fd_2nd_matrix = np.linalg.norm(u_dd_circulant_fd_2nd_matrix - u_dd_ref, ord=np.inf) / np.linalg.norm(u_dd_ref, ord=np.inf)
rel_error_u_dd_circulant_fd_4th_matrix = np.linalg.norm(u_dd_circulant_fd_4th_matrix - u_dd_ref, ord=np.inf) / np.linalg.norm(u_dd_ref, ord=np.inf)
rel_error_u_dd_circulant_fd_6th_matrix = np.linalg.norm(u_dd_circulant_fd_6th_matrix - u_dd_ref, ord=np.inf) / np.linalg.norm(u_dd_ref, ord=np.inf)
rel_error_u_dd_circulant_fd_8th_matrix = np.linalg.norm(u_dd_circulant_fd_8th_matrix - u_dd_ref, ord=np.inf) / np.linalg.norm(u_dd_ref, ord=np.inf)
# ---------------------------------------------------------------------------------------------

print('Jx: {0:d}'.format(Jx))
print()
print('rel_error_u_dd_fourier_matrix:       {0:1.2e}'.format(rel_error_u_dd_fourier_matrix))
print('rel_error_u_dd_fourier_fft:          {0:1.2e}'.format(rel_error_u_dd_fourier_fft))
print()
print('rel_error_u_dd_circulant_fd_2nd_fft: {0:1.2e}'.format(rel_error_u_dd_circulant_fd_2nd_fft))
print('rel_error_u_dd_circulant_fd_4th_fft: {0:1.2e}'.format(rel_error_u_dd_circulant_fd_4th_fft))
print('rel_error_u_dd_circulant_fd_6th_fft: {0:1.2e}'.format(rel_error_u_dd_circulant_fd_6th_fft))
print('rel_error_u_dd_circulant_fd_8th_fft: {0:1.2e}'.format(rel_error_u_dd_circulant_fd_8th_fft))
print()
print('rel_error_u_dd_circulant_fd_2nd_matrix: {0:1.2e}'.format(rel_error_u_dd_circulant_fd_2nd_matrix))
print('rel_error_u_dd_circulant_fd_4th_matrix: {0:1.2e}'.format(rel_error_u_dd_circulant_fd_4th_matrix))
print('rel_error_u_dd_circulant_fd_6th_matrix: {0:1.2e}'.format(rel_error_u_dd_circulant_fd_6th_matrix))
print('rel_error_u_dd_circulant_fd_8th_matrix: {0:1.2e}'.format(rel_error_u_dd_circulant_fd_8th_matrix))
print()

# -------------------------------------------------------------------------------------------------
fig_1 = plt.figure(num="fig_1", figsize=(8, 8))
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
gridspec = fig_1.add_gridspec(nrows=3, ncols=2,
                              left=0.1, right=0.95,
                              bottom=0.08, top=0.95,
                              wspace=0.25,
                              hspace=0.4
                              # width_ratios=[1, 1],
                              # height_ratios=[1, 1]
                              )
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
ax_00 = fig_1.add_subplot(gridspec[0, 0])
ax_10 = fig_1.add_subplot(gridspec[1, 0])
ax_20 = fig_1.add_subplot(gridspec[2, 0])

ax_01 = fig_1.add_subplot(gridspec[0, 1])
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
ax_00.plot(x, u, color='k', linestyle='-', linewidth=1)

y_min = 0.0
y_max = 1.5

ax_00.set_xlim(x_min-0.1*Lx, x_max+0.1*Lx)
# ax_00.set_ylim(y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min))

x_ticks_major = np.linspace(x_min, x_max, num=5)
y_ticks_major = np.linspace(y_min, y_max, num=7)
# y_ticks_minor = 0.5 * (y_ticks_major[0:-1] + y_ticks_major[1:])

ax_00.set_xticks(x_ticks_major, minor=False)
# ax_00.set_yticks(y_ticks_major, minor=False)
# ax_00.set_yticks(y_ticks_minor, minor=True)

ax_00.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5)
ax_00.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.25)

# ax_00.set_xlabel(r'$x$', labelpad=12)

ax_00.set_title(r'$f^{(0)}(x)$')
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
ax_10.plot(x, u_d_ref, color='k', linestyle='-', linewidth=1)

# y_min = -10.0
# y_max = +10.0

ax_10.set_xlim(x_min-0.1*Lx, x_max+0.1*Lx)
# ax_10.set_ylim(y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min))

# y_ticks_major = np.linspace(y_min, y_max, num=5)
# y_ticks_minor = 0.5 * (y_ticks_major[0:-1] + y_ticks_major[1:])

ax_10.set_xticks(x_ticks_major, minor=False)
# ax_10.set_yticks(y_ticks_major, minor=False)
# ax_00.set_yticks(y_ticks_minor, minor=True)

ax_10.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5)
ax_10.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.25)

# ax_10.set_xlabel(r'$x$', labelpad=12)

ax_10.set_title(r'$f^{(1)}(x)$')
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
ax_20.plot(x, u_dd_ref, color='k', linestyle='-', linewidth=1)

# y_min = -100.0
# y_max = +100.0

ax_20.set_xlim(x_min-0.1*Lx, x_max+0.1*Lx)
# ax_20.set_ylim(y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min))

# y_ticks_major = np.linspace(y_min, y_max, num=5)
# y_ticks_minor = 0.5 * (y_ticks_major[0:-1] + y_ticks_major[1:])

ax_20.set_xticks(x_ticks_major, minor=False)
# ax_20.set_yticks(y_ticks_major, minor=False)
# ax_00.set_yticks(y_ticks_minor, minor=True)

ax_20.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5)
ax_20.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.25)

ax_20.set_title(r'$f^{(2)}(x)$')

ax_20.set_xlabel(r'$x$', labelpad=12)
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
linewidth = 1.75

ax_01.plot(np.arange(0, Jx), np.abs(lambda_d_xx_circulant_fd_2nd), color=colors.green_sea, linestyle='-', linewidth=linewidth)
ax_01.plot(np.arange(0, Jx), np.abs(lambda_d_xx_circulant_fd_4th), color=colors.peter_river, linestyle='-', linewidth=linewidth)
ax_01.plot(np.arange(0, Jx), np.abs(lambda_d_xx_circulant_fd_6th), color=colors.orange, linestyle='-', linewidth=linewidth)
ax_01.plot(np.arange(0, Jx), np.abs(lambda_d_xx_circulant_fd_8th), color=colors.alizarin, linestyle='-', linewidth=linewidth)

ax_01.plot(np.arange(0, Jx), np.abs(lambda_d_xx_fourier), color=colors.black, linestyle='-', linewidth=linewidth)

# y_min = 0.0
# y_max = 1.5

# ax_01.set_xlim(x_min-0.1*Lx, x_max+0.1*Lx)
# ax_01.set_ylim(y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min))

# x_ticks_major = np.linspace(x_min, x_max, num=11)
# y_ticks_major = np.linspace(y_min, y_max, num=6)
# y_ticks_minor = 0.5 * (y_ticks_major[0:-1] + y_ticks_major[1:])

# ax_01.set_xticks(x_ticks_major, minor=False)
# ax_01.set_yticks(y_ticks_major, minor=False)
# ax_00.set_yticks(y_ticks_minor, minor=True)

ax_01.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5)
ax_01.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.25)

# ax_00.set_xlabel(r'$x$', labelpad=12)

ax_01.set_title(r'$|c(k)|$')
# -------------------------------------------------------------------------------------------------

plt.draw()

plt.show()
