from qsolve.solvers import SolverGPE1D

from qsolve.figures import FigureMain1D
from qsolve.figures import FigureEigenstatesLSE1D

from potential_harmonic_trap_1d import compute_external_potential

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import time

import mkl

import numpy as np

import scipy

from scipy.interpolate import pchip_interpolate

import matplotlib.pyplot as plt

import matplotlib as mpl


# -------------------------------------------------------------------------------------------------
num_threads_cpu = 8

os.environ["OMP_NUM_THREADS"] = str(num_threads_cpu)
os.environ["MKL_NUM_THREADS"] = str(num_threads_cpu)

mkl.set_num_threads(num_threads_cpu)

assert(mkl.get_max_threads() == num_threads_cpu)
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
pi = scipy.constants.pi

hbar = scipy.constants.hbar

amu = scipy.constants.physical_constants["atomic mass constant"][0]
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# close figures from previous simulation

plt.close('all')
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
export_frames_figure_main = False
# -------------------------------------------------------------------------------------------------


# =================================================================================================
n_atoms = 3500

m_Rb_87 = 87 * amu

x_min = -60e-6
x_max = +60e-6

Jx = 512

t_final = 8e-3

dt = 0.0005e-3

n_mod_times_analysis = 100

n_control_inputs = 1
# =================================================================================================

parameters_potential = {'nu_start': (22.5, "Hz"),
                        'nu_final': (22.5, "Hz")}

parameters_figure_main = {'density_min': -20,
                          'density_max': +220,
                          'V_min': -0.5,
                          'V_max': +4.5,
                          'x_ticks': np.array([-40, -20, 0, 20, 40]),
                          't_ticks': np.array([0, 2, 4, 6, 8]),
                          'n_control_inputs': n_control_inputs}

parameters_figure_eigenstates_lse = {'density_min': 0,
                                     'density_max': 200,
                                     'psi_re_min': -0.5,
                                     'psi_re_max': +0.5,
                                     'V_min': 0.0,
                                     'V_max': 4.0,
                                     'x_ticks': np.array([-40, -20, 0, 20, 40])
                                     }
# =================================================================================================

# -------------------------------------------------------------------------------------------------
simulation_id = 'harmonic_trap_1d'

simulation_id = simulation_id.replace(".", "_")
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# frames

path_frames_figure_main = "./frames/frames_figure_main/" + simulation_id + "/"

nr_frame_figure_main = 0

if export_frames_figure_main:

    if not os.path.exists(path_frames_figure_main):

        os.makedirs(path_frames_figure_main)
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# init solver
# =================================================================================================

solver = SolverGPE1D(m_atom=m_Rb_87,
                     a_s=5.24e-9,
                     omega_perp=2*np.pi*1000,
                     seed=1,
                     device='cuda:0',
                     # device='cpu',
                     num_threads_cpu=num_threads_cpu)


# =================================================================================================
# init spatial grid
# =================================================================================================

solver.init_grid(x_min=x_min, x_max=x_max, Jx=Jx)


# =================================================================================================
# init time evolution
# =================================================================================================

# -------------------------------------------------------------------------------------------------
n_time_steps = int(np.round(t_final / dt))

n_times = n_time_steps + 1

assert (np.abs(n_time_steps * dt - t_final)) < 1e-14

times = dt * np.arange(n_times)

assert (np.abs(times[-1] - t_final)) < 1e-14
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
times_analysis = times[0::n_mod_times_analysis]

n_times_analysis = times_analysis.size

assert (abs(times_analysis[-1] - t_final) / abs(t_final) < 1e-14)
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# init control inputs
# =================================================================================================

u_of_times = np.zeros((n_control_inputs, n_times))

vec_t = np.array([0.0, 0.1, 0.2, 1.0]) * t_final
vec_u = np.array([0.0, 0.0, 1.0, 1.0])

u1_of_times = pchip_interpolate(vec_t, vec_u, times)

u_of_times[0, :] = u1_of_times


# =================================================================================================
# init external potential
# =================================================================================================

solver.init_external_potential(compute_external_potential, parameters_potential)

solver.set_external_potential(t=0.0, u=u_of_times[0])


# =================================================================================================
# compute eigenstates of the linear Schrödinger equation
# =================================================================================================

time_1 = time.time()

eigenstates_lse, matrix_res_batch, vec_iter = solver.compute_eigenstates_lse(
    n_eigenstates_max=128,
    n_iter_max=2000,
    tau_0=0.25e-3,
    # propagation_method='trotter',
    # propagation_method='strang',
    # propagation_method='ite_4th',
    # propagation_method='ite_6th',
    propagation_method='ite_8th',
    # propagation_method='ite_10th',
    # propagation_method='ite_12th',
    # orthogonalization_method='gram_schmidt',
    orthogonalization_method='qr',
    return_residuals=True)

time_2 = time.time()

print('elapsed time: {0:f}'.format(time_2 - time_1))

figure_eigenstates_lse = FigureEigenstatesLSE1D(eigenstates_lse,
                                                solver.V,
                                                solver.x,
                                                parameters_figure_eigenstates_lse)

# =================================================================================================
# show convergence of linear eigenstate computation
# =================================================================================================

n_eigenstates_lse = matrix_res_batch.shape[1]

n_lines = n_eigenstates_lse

c = np.arange(0, n_lines)

cmap_tmp = mpl.colormaps['Spectral']

norm = mpl.colors.Normalize(0, vmax=n_lines-1)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_tmp)
cmap.set_array([])


fig_conv_lse = plt.figure("figure_convergence_lse", figsize=(1.5*6, 1.5*4))

fig_conv_lse.subplots_adjust(left=0.1, right=1.0, bottom=0.125, top=0.925)

ax = fig_conv_lse.add_subplot(111)

ax.set_facecolor((0.15, 0.15, 0.15))

ax.set_yscale('log')

ax.set_title('linear eigenstate computation')

plt.grid(visible=True, which='major', color=(0.5, 0.5, 0.5), linestyle='-', linewidth=0.5)
# plt.grid(visible=False, which='minor', color='k', linestyle='-', linewidth=0.25)

for nr in range(n_eigenstates_lse):
    plt.plot(vec_iter, matrix_res_batch[:, nr], linewidth=1.5, linestyle='-', color=cmap.to_rgba(nr))

ax.set_xlim(0, vec_iter[-1])
ax.set_ylim(1e-14, 1e0)

plt.xlabel(r'number of iterations', labelpad=12)
plt.ylabel(r'residual error', labelpad=12)

cbar = fig_conv_lse.colorbar(cmap, ax=ax, label=r'# eigenstate')

ticks_true = np.linspace(0, n_eigenstates_lse+1, 4)

cbar.ax.tick_params(length=6, pad=4, which="major")

fig_conv_lse.canvas.start_event_loop(0.001)

plt.draw()

# =================================================================================================
# set wave function to ground state solution
# =================================================================================================

# solver.psi = psi_0
#
# N_0 = solver.compute_n_atoms()
# mue_0 = solver.compute_chemical_potential()
# E_total_0 = solver.compute_total_energy()
# E_kinetic_0 = solver.compute_kinetic_energy()
# E_potential_0 = solver.compute_potential_energy()
# E_interaction_0 = solver.compute_interaction_energy()
#
# print('N_0 = {:1.16e}'.format(N_0))
# print('mue_0 / h: {0:1.6} kHz'.format(mue_0 / (1e3 * (2 * pi * hbar))))
# print('E_total_0 / (N_0*h): {0:1.6} kHz'.format(E_total_0 / (1e3 * (2 * pi * hbar * N_0))))
# print('E_kinetic_0 / (N_0*h): {0:1.6} kHz'.format(E_kinetic_0 / (1e3 * (2 * pi * hbar * N_0))))
# print('E_potential_0 / (N_0*h): {0:1.6} kHz'.format(E_potential_0 / (1e3 * (2 * pi * hbar * N_0))))
# print('E_interaction_0 / (N_0*h): {0:1.6} kHz'.format(E_interaction_0 / (1e3 * (2 * pi * hbar * N_0))))
# print()
# -------------------------------------------------------------------------------------------------


plt.ioff()
plt.show()
