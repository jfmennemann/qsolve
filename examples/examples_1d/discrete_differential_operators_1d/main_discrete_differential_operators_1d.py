import numpy as np

from qsolve_core import D1_fourier_1d
from qsolve_core import D2_fourier_1d

from qsolve_core import lambda_D1_fourier_1d
from qsolve_core import lambda_D2_fourier_1d

from qsolve_core import D2_circulant_fd_1d


from problem_1 import f, f_x, f_xx

import matplotlib.pyplot as plt


pi = np.pi

x_min = 0
x_max = 1

Jx = 2 ** 11

x = np.linspace(x_min, x_max, Jx, endpoint=False)

dx = x[1] - x[0]

# ---------------------------------------------------------------------------------------------
# exact reference solution

Lx = x_max - x_min

assert (Lx == Jx * dx)

u = f(x, Lx)
u_d_ref = f_x(x, Lx)
u_dd_ref = f_xx(x, Lx)
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# solution via FFT

Lx = Jx * dx

# ----
lambda_d_x = lambda_D1_fourier_1d(Jx, dx)

u_d_fft = np.fft.ifft(lambda_d_x * np.fft.fft(u))

u_d_fft = np.real(u_d_fft)
# ----

# ----
lambda_d_xx = lambda_D2_fourier_1d(Jx, dx)

u_dd_fft = np.fft.ifft(lambda_d_xx * np.fft.fft(u))

u_dd_fft = np.real(u_dd_fft)
# ----
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# solution via Fourier differentiation matrix

D1 = D1_fourier_1d(Jx, dx)
D2 = D2_fourier_1d(Jx, dx)

u_d_matrix = D1 @ u
u_dd_matrix = D2 @ u
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# solution via finite differences

u_d_fd = (np.roll(u, shift=-1) - np.roll(u, shift=+1)) / (2.0 * dx)

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

D2_circulant_fd = D2_circulant_fd_1d(Jx, dx, order=4)

c = D2_circulant_fd[:, 0]

# u_dd_fd = D2_circulant_fd @ u

u_dd_fd = np.fft.ifft(np.fft.fft(c) * np.fft.fft(u))
# ---------------------------------------------------------------------------------------------

rel_error_u_d_matrix = np.linalg.norm(u_d_matrix - u_d_ref, ord=np.inf) / np.linalg.norm(u_d_ref, ord=np.inf)
rel_error_u_d_fft = np.linalg.norm(u_d_fft - u_d_ref, ord=np.inf) / np.linalg.norm(u_d_ref, ord=np.inf)

rel_error_u_dd_matrix = np.linalg.norm(u_dd_matrix - u_dd_ref, ord=np.inf) / np.linalg.norm(u_dd_ref, ord=np.inf)
rel_error_u_dd_fft = np.linalg.norm(u_dd_fft - u_dd_ref, ord=np.inf) / np.linalg.norm(u_dd_ref, ord=np.inf)

rel_error_u_d_fd = np.linalg.norm(u_d_fd - u_d_ref, ord=np.inf) / np.linalg.norm(u_d_ref, ord=np.inf)
rel_error_u_dd_fd = np.linalg.norm(u_dd_fd - u_dd_ref, ord=np.inf) / np.linalg.norm(u_dd_ref, ord=np.inf)

rel_difference_u_d_matrix_vs_fft = np.linalg.norm(u_d_matrix - u_d_fft, ord=np.inf) / np.linalg.norm(u_d_fft, ord=np.inf)
rel_difference_u_dd_matrix_vs_fft = np.linalg.norm(u_dd_matrix - u_dd_fft, ord=np.inf) / np.linalg.norm(u_dd_fft, ord=np.inf)

print('Jx: {0:d}'.format(Jx))
print()
print('rel_error_u_d_matrix:              {0:1.2e}'.format(rel_error_u_d_matrix))
print('rel_error_u_dd_matrix:             {0:1.2e}'.format(rel_error_u_dd_matrix))
print()
print('rel_error_u_d_fft:                 {0:1.2e}'.format(rel_error_u_d_fft))
print('rel_error_u_dd_fft:                {0:1.2e}'.format(rel_error_u_dd_fft))
print()
print('rel_error_u_d_fd:                  {0:1.2e}'.format(rel_error_u_d_fd))
print('rel_error_u_dd_fd:                 {0:1.2e}'.format(rel_error_u_dd_fd))
print()
print('rel_difference_u_d_matrix_vs_fft:  {0:1.2e}'.format(rel_difference_u_d_matrix_vs_fft))
print('rel_difference_u_dd_matrix_vs_fft: {0:1.2e}'.format(rel_difference_u_dd_matrix_vs_fft))
print()


# =================================================================================================

# -------------------------------------------------------------------------------------------------
fig_1 = plt.figure(num="fig_1", figsize=(8, 8))
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
gridspec = fig_1.add_gridspec(nrows=3, ncols=1,
                              left=0.1, right=0.95,
                              bottom=0.08, top=0.95,
                              wspace=0.0,
                              hspace=0.4
                              # width_ratios=[1, 1],
                              # height_ratios=[1, 1]
                              )
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
ax_00 = fig_1.add_subplot(gridspec[0, 0])
ax_10 = fig_1.add_subplot(gridspec[1, 0])
ax_20 = fig_1.add_subplot(gridspec[2, 0])
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
ax_00.plot(x, u, color='k', linestyle='-', linewidth=1)

y_min = 0.0
y_max = 1.5

ax_00.set_xlim(x_min-0.1*Lx, x_max+0.1*Lx)
ax_00.set_ylim(y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min))

x_ticks_major = np.linspace(x_min, x_max, num=11)
y_ticks_major = np.linspace(y_min, y_max, num=6)
# y_ticks_minor = 0.5 * (y_ticks_major[0:-1] + y_ticks_major[1:])

ax_00.set_xticks(x_ticks_major, minor=False)
ax_00.set_yticks(y_ticks_major, minor=False)
# ax_00.set_yticks(y_ticks_minor, minor=True)

ax_00.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5)
ax_00.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.25)

# ax_00.set_xlabel(r'$x$', labelpad=12)

ax_00.set_title(r'$f^{(0)}(x)$')
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
ax_10.plot(x, u_d_ref, color='k', linestyle='-', linewidth=1)

y_min = -10.0
y_max = +10.0

ax_10.set_xlim(x_min-0.1*Lx, x_max+0.1*Lx)
ax_10.set_ylim(y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min))

y_ticks_major = np.linspace(y_min, y_max, num=5)
# y_ticks_minor = 0.5 * (y_ticks_major[0:-1] + y_ticks_major[1:])

ax_10.set_xticks(x_ticks_major, minor=False)
ax_10.set_yticks(y_ticks_major, minor=False)
# ax_00.set_yticks(y_ticks_minor, minor=True)

ax_10.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5)
ax_10.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.25)

# ax_10.set_xlabel(r'$x$', labelpad=12)

ax_10.set_title(r'$f^{(1)}(x)$')
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
ax_20.plot(x, u_dd_ref, color='k', linestyle='-', linewidth=1)

y_min = -100.0
y_max = +100.0

ax_20.set_xlim(x_min-0.1*Lx, x_max+0.1*Lx)
ax_20.set_ylim(y_min-0.1*(y_max-y_min), y_max+0.1*(y_max-y_min))

y_ticks_major = np.linspace(y_min, y_max, num=5)
# y_ticks_minor = 0.5 * (y_ticks_major[0:-1] + y_ticks_major[1:])

ax_20.set_xticks(x_ticks_major, minor=False)
ax_20.set_yticks(y_ticks_major, minor=False)
# ax_00.set_yticks(y_ticks_minor, minor=True)

ax_20.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5)
ax_20.grid(visible=True, which='minor', color='k', linestyle='--', linewidth=0.25)

ax_20.set_title(r'$f^{(2)}(x)$')

ax_20.set_xlabel(r'$x$', labelpad=12)
# -------------------------------------------------------------------------------------------------

plt.draw()

plt.show()
# =================================================================================================

