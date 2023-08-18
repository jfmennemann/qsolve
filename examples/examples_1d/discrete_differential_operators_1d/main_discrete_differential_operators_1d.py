import numpy as np

from qsolve_core import D1_fourier_1d
from qsolve_core import D2_fourier_1d

import matplotlib.pyplot as plt


pi = np.pi

x_min = 0
x_max = 1

Jx = 2 ** 12

x = np.linspace(x_min, x_max, Jx, endpoint=False)

dx = x[1] - x[0]

# ---------------------------------------------------------------------------------------------
# exact reference solution

Lx = x_max - x_min

assert (Lx == Jx * dx)

u = np.exp(np.sin(2 * pi * x / Lx))

u_d_ref = (2 * pi / Lx) * np.cos(2 * pi * x / Lx) * u

u_dd_ref = (-(2 * pi / Lx) ** 2 * np.sin(2 * pi * x / Lx) * u + (2 * pi / Lx) * np.cos(2 * pi * x / Lx) * u_d_ref)
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# solution via FFT

Lx = Jx * dx

# ----
xi = 2 * np.pi * np.arange(start=-Jx // 2, stop=Jx // 2) / Lx
xi[0] = 0.0
xi = np.fft.fftshift(xi)
lambda_d_x = 1j * xi

u_d_fft = np.fft.ifft(lambda_d_x * np.fft.fft(u))
u_d_fft = np.real(u_d_fft)
# ----

# ----
xi = 2 * np.pi * np.arange(start=-Jx // 2, stop=Jx // 2) / Lx
xi = np.fft.fftshift(xi)
lambda_d_xx = -1.0 * xi * xi

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

u_dd_fd = (
    - 9 * np.roll(u, shift=-4)
    + 128 * np.roll(u, shift=-3)
    - 1008 * np.roll(u, shift=-2)
    + 8064 * np.roll(u, shift=-1)
    - 14350 * u
    + 8064 * np.roll(u, shift=+1)
    - 1008 * np.roll(u, shift=+2)
    + 128 * np.roll(u, shift=+3)
    - 9 * np.roll(u, shift=+4)
) / (5040 * dx ** 2)
# ---------------------------------------------------------------------------------------------

rel_error_u_d_matrix = np.linalg.norm(u_d_matrix - u_d_ref) / np.linalg.norm(u_d_ref)
rel_error_u_d_fft = np.linalg.norm(u_d_fft - u_d_ref) / np.linalg.norm(u_d_ref)

rel_error_u_dd_matrix = np.linalg.norm(u_dd_matrix - u_dd_ref) / np.linalg.norm(u_dd_ref)
rel_error_u_dd_fft = np.linalg.norm(u_dd_fft - u_dd_ref) / np.linalg.norm(u_dd_ref)

rel_error_u_d_fd = np.linalg.norm(u_d_fd - u_d_ref) / np.linalg.norm(u_d_ref)
rel_error_u_dd_fd = np.linalg.norm(u_dd_fd - u_dd_ref) / np.linalg.norm(u_dd_ref)

rel_difference_u_d_matrix_vs_fft = np.linalg.norm(u_d_matrix - u_d_fft) / np.linalg.norm(u_d_fft)
rel_difference_u_dd_matrix_vs_fft = np.linalg.norm(u_dd_matrix - u_dd_fft) / np.linalg.norm(u_dd_fft)

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


# -------------------------------------------------------------------------------------------------
fig_1 = plt.figure(num="fig_1", figsize=(8, 8))

gridspec = fig_1.add_gridspec(nrows=2, ncols=1,
                              left=0.125, right=0.9,
                              bottom=0.08, top=0.95,
                              wspace=0.0,
                              hspace=0.2
                              # width_ratios=[1, 1],
                              # height_ratios=[1, 1]
                              )

ax_00 = fig_1.add_subplot(gridspec[0, 0])
ax_10 = fig_1.add_subplot(gridspec[1, 0])

# ax_2.set_yscale('log')

# ax_2.set_title('ground state computation')

# plt.plot(vec_iter, vec_res, linewidth=1, linestyle='-', color='k')

# ax_2.set_xlim(0, vec_iter[-1])
# ax_2.set_ylim(1e-8, 1)

# plt.xlabel(r'number of iterations', labelpad=12)
# plt.ylabel(r'relative residual error', labelpad=12)

# plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5)
# plt.grid(visible=True, which='minor', color='k', linestyle='-', linewidth=0.25)

ax_00.plot(x, u, color='k', linestyle='-', linewidth=0.65)

ax_00.set_xlim(x_min-0.1*Lx, x_max+0.1*Lx)

ax_00.set_xlabel(r'$x$', labelpad=12)

plt.draw()

plt.show()
# -------------------------------------------------------------------------------------------------
