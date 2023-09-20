from qsolve.solvers import SolverGPE1D
from qsolve.grids import Grid1D
from qsolve.units import Units

from qsolve.figures import FigureMain1D
from qsolve.figures import FigureEigenstatesLSE1D
from qsolve.figures import FigureEigenstatesBDG1D

from potential_harmonic_trap_1d import PotentialHarmonicTrap1D

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import mkl

import numpy as np

import scipy

from scipy.interpolate import pchip_interpolate

import pathlib

import h5py

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

k_B = scipy.constants.Boltzmann
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
# n_atoms = 1

m_Rb_87 = 87 * amu

m_atom = m_Rb_87

T = 20e-9

x_min = -60e-6
x_max = +60e-6

Jx = 1024

t_final = 8e-3

dt = 0.0005e-3

n_mod_times_analysis = 100

n_control_inputs = 1

device = 'cuda:0'
# =================================================================================================

parameters_potential = {
    'm_atom': m_atom,
    'nu_start': 40,
    'nu_final': 22.5
}

parameters_figure_main = {'density_min': -20,
                          'density_max': +220,
                          'V_min': -0.5,
                          'V_max': +4.5,
                          'x_ticks': np.array([-40, -20, 0, 20, 40]),
                          't_ticks': np.array([0, 2, 4, 6, 8]),
                          'n_control_inputs': n_control_inputs}

parameters_figure_eigenstates_lse = {'density_min': 0,
                                     'density_max': 200,
                                     'psi_re_min': -1.0,
                                     'psi_re_max': +1.0,
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

units = Units.solver_units(m_atom, dim=1)

grid = Grid1D(x_min=x_min, x_max=x_max, Jx=Jx)

potential = PotentialHarmonicTrap1D(grid=grid, units=units, device=device, parameters=parameters_potential)

solver = SolverGPE1D(
    units=units,
    grid=grid,
    potential=potential,
    device=device,
    m_atom=m_Rb_87,
    a_s=5.24e-9,
    omega_perp=2*np.pi*1000,
    seed=1,
    num_threads_cpu=num_threads_cpu)


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

solver.set_external_potential(t=0.0, u=u_of_times[0])


# =================================================================================================
# compute ground state solution
# =================================================================================================

psi_0, mue_0, vec_res, vec_iter = solver.compute_ground_state_solution(
    n_atoms=n_atoms,
    n_iter_max=10000,
    tau_0=0.001e-3,
    adaptive_tau=True,
    return_residuals=True)


# -------------------------------------------------------------------------------------------------
fig_conv_ground_state = plt.figure("figure_convergence_ground_state", figsize=(6, 4))

fig_conv_ground_state.subplots_adjust(left=0.175, right=0.95, bottom=0.2, top=0.9)

ax = fig_conv_ground_state.add_subplot(111)

ax.set_yscale('log')

ax.set_title('ground state computation')

plt.plot(vec_iter, vec_res, linewidth=1, linestyle='-', color='k')

ax.set_xlim(0, vec_iter[-1])
ax.set_ylim(1e-8, 1)

plt.xlabel(r'number of iterations', labelpad=12)
plt.ylabel(r'relative residual error', labelpad=12)

plt.grid(visible=True, which='major', color='k', linestyle='-', linewidth=0.5)
plt.grid(visible=True, which='minor', color='k', linestyle='-', linewidth=0.25)

plt.draw()
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# compute eigenstates of the linear Schrödinger equation
# =================================================================================================

ite = False

if ite:

    eigenstates_lse, energies_lse, matrix_res_batch, vec_iter = solver.compute_eigenstates_lse_ite(
        n_eigenstates=128,
        n_iter_max=1000,
        tau_0=0.25e-3,
        propagation_method='ite_12th',
        return_residuals=True)

    n_eigenstates_lse = matrix_res_batch.shape[0]

    n_lines = n_eigenstates_lse

    c = np.arange(0, n_lines)

    cmap_tmp = mpl.colormaps['Spectral']

    norm = mpl.colors.Normalize(vmin=0, vmax=n_lines - 1)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_tmp)
    cmap.set_array([])

    figure_convergence_lse_1d = plt.figure(num="figure_convergence_lse_1d", figsize=(1.5 * 6, 1.5 * 4))

    figure_convergence_lse_1d.subplots_adjust(left=0.1, right=1.0, bottom=0.125, top=0.925)

    ax = figure_convergence_lse_1d.add_subplot(111)

    ax.set_facecolor((0.15, 0.15, 0.15))

    ax.set_yscale('log')

    ax.set_title('linear eigenstate computation')

    plt.grid(visible=True, which='major', color=(0.5, 0.5, 0.5), linestyle='-', linewidth=0.5)
    # plt.grid(visible=False, which='minor', color='k', linestyle='-', linewidth=0.25)

    for nr in range(n_eigenstates_lse):
        plt.plot(vec_iter, matrix_res_batch[nr, :], linewidth=1.5, linestyle='-', color=cmap.to_rgba(nr))

    ax.set_xlim(0, vec_iter[-1])
    ax.set_ylim(1e-14, 1e0)

    plt.xlabel(r'number of iterations', labelpad=12)
    plt.ylabel(r'residual error', labelpad=12)

    cbar = figure_convergence_lse_1d.colorbar(cmap, ax=ax, label=r'# eigenstate')

    ticks_true = np.linspace(0, n_eigenstates_lse + 1, 4)

    cbar.ax.tick_params(length=6, pad=4, which="major")

    figure_convergence_lse_1d.canvas.start_event_loop(0.001)

    plt.draw()

else:

    eigenstates_lse, energies_lse = solver.compute_eigenstates_lse(n_eigenstates=128)

figure_eigenstates_lse = FigureEigenstatesLSE1D(eigenstates_lse,
                                                solver.V,
                                                grid.x,
                                                parameters_figure_eigenstates_lse)

# -------------------------------------------------------------------------------------------------
print('3 * k_B * T / mue_0:' )
print(3 * k_B * T / mue_0)
print()

E_cutoff = mue_0 + 3 * k_B * T

print('energies_lse / E_cutoff: ')
print(energies_lse / E_cutoff)
print()

indices_lse_selected = (energies_lse / E_cutoff) <= 1

eigenstates_lse = eigenstates_lse[indices_lse_selected, :]
energies_lse = energies_lse[indices_lse_selected]

print('energies_lse / E_cutoff: ')
print(energies_lse / E_cutoff)
print()
# -------------------------------------------------------------------------------------------------




# =================================================================================================
# compute quasiparticle amplitudes u and v
# =================================================================================================

path = "./data/bdg.hdf5"

if not os.path.exists(path):

    excitations_u, excitations_v, frequencies_omega, psi_0_bdg, mue_0_bdg = solver.bdg_experimental(
        n_atoms=n_atoms, n_excitations=16)

    pathlib.Path('./data').mkdir(parents=True, exist_ok=True)

    f_hdf5 = h5py.File(path, mode="w")

    f_hdf5.create_dataset(name="excitations_u", data=excitations_u, dtype=np.float64)
    f_hdf5.create_dataset(name="excitations_v", data=excitations_v, dtype=np.float64)
    f_hdf5.create_dataset(name="frequencies_omega", data=frequencies_omega, dtype=np.float64)
    f_hdf5.create_dataset(name="psi_0", data=psi_0_bdg, dtype=np.float64)
    f_hdf5.create_dataset(name="mue_0", data=mue_0_bdg)

    f_hdf5.close()

else:

    f_hdf5 = h5py.File(path, mode='r')

    excitations_u = f_hdf5['excitations_u'][:]
    excitations_v = f_hdf5['excitations_v'][:]

    frequencies_omega = f_hdf5['frequencies_omega'][:]

    psi_0 = f_hdf5['psi_0'][:]
    mue_0 = f_hdf5['mue_0']

    print(excitations_u.shape)
    print(excitations_v.shape)
    print(frequencies_omega.shape)
    print(psi_0.shape)
    print()
    print(frequencies_omega)
    print()
    print()

parameters_figure_eigenstates_bdg = {'u_v_re_im_min': -1.0,
                                     'u_v_re_im_max': +1.0,
                                     'V_min': 0.0,
                                     'V_max': 4.0,
                                     'x_ticks': np.array([-40, -20, 0, 20, 40])
                                     }

figure_eigenstates_bdg = FigureEigenstatesBDG1D(excitations_u,
                                                excitations_v,
                                                solver.V,
                                                grid.x,
                                                parameters_figure_eigenstates_bdg)


# =================================================================================================
# set wave function to ground state solution
# =================================================================================================

solver.psi = psi_0

N_0 = solver.compute_n_atoms()
mue_0 = solver.compute_chemical_potential()
E_total_0 = solver.compute_total_energy()
E_kinetic_0 = solver.compute_kinetic_energy()
E_potential_0 = solver.compute_potential_energy()
E_interaction_0 = solver.compute_interaction_energy()

print('N_0 = {:1.16e}'.format(N_0))
print('mue_0 / h: {0:1.6} kHz'.format(mue_0 / (1e3 * (2 * pi * hbar))))
print('E_total_0 / (N_0*h): {0:1.6} kHz'.format(E_total_0 / (1e3 * (2 * pi * hbar * N_0))))
print('E_kinetic_0 / (N_0*h): {0:1.6} kHz'.format(E_kinetic_0 / (1e3 * (2 * pi * hbar * N_0))))
print('E_potential_0 / (N_0*h): {0:1.6} kHz'.format(E_potential_0 / (1e3 * (2 * pi * hbar * N_0))))
print('E_interaction_0 / (N_0*h): {0:1.6} kHz'.format(E_interaction_0 / (1e3 * (2 * pi * hbar * N_0))))
print()
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# init main figure
# =================================================================================================

# -------------------------------------------------------------------------------------------------
figure_main = FigureMain1D(grid.x, times, parameters_figure_main)

figure_main.fig_control_inputs.update_u(u_of_times)

figure_main.fig_control_inputs.update_t(0.0)
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
figure_main.update_data(solver.psi, solver.V)

figure_main.redraw()
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# thermal state sampling
# =================================================================================================

if T > 0:

    solver.init_sgpe(
        T_temp_des=T,
        mue_des=mue_0,
        gamma=0.1,
        dt=dt
    )

    n_sgpe_max = 10000

    n_sgpe_inc = 1000

    n_sgpe = 0

    while n_sgpe < n_sgpe_max:

        N = solver.compute_n_atoms()

        print('----------------------------------------------------------------------------------------')
        print('n_sgpe: {0:4d} / {1:4d}'.format(n_sgpe, n_sgpe_max))
        print()
        print('N:      {0:1.4f}'.format(N))
        print('----------------------------------------------------------------------------------------')
        print()

        # -----------------------------------------------------------------------------------------
        figure_main.update_data(solver.psi, solver.V)

        figure_main.redraw()
        # -----------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------------
        # apply thermal state sampling process via sgpe for n_sgpe_inc time steps

        solver.propagate_sgpe(n_inc=n_sgpe_inc)
        # ---------------------------------------------------------------------------------------------

        n_sgpe = n_sgpe + n_sgpe_inc


# =================================================================================================
# projection

# solver.psi = grid.dx * eigenstates_lse.T @ (eigenstates_lse @ solver.psi)
# =================================================================================================


# =================================================================================================
# compute time evolution
# =================================================================================================

n_inc = n_mod_times_analysis

nr_times_analysis = 0

stop = False

n = 0

while True:

    psi = solver.psi
    V = solver.V

    t = times[n]

    N = solver.compute_n_atoms()

    print('t: {0:1.2f} / {1:1.2f}, n: {2:4d} / {3:4d}, N:{4:4f}'.format(t / 1e-3, times[-1] / 1e-3, n, n_times, N))

    # ---------------------------------------------------------------------------------------------
    figure_main.update_data(psi, V)

    figure_main.fig_control_inputs.update_t(t)

    figure_main.redraw()

    if export_frames_figure_main:

        filepath = path_frames_figure_main + 'frame_' + str(nr_frame_figure_main).zfill(5) + '.png'

        figure_main.export(filepath)

        nr_frame_figure_main = nr_frame_figure_main + 1
    # ---------------------------------------------------------------------------------------------

    nr_times_analysis = nr_times_analysis + 1

    if n < n_times - n_inc:

        solver.propagate_gpe(times=times, u_of_times=u_of_times, n_start=n, n_inc=n_inc, mue_shift=mue_0)

        n = n + n_inc

    else:

        break

    # input()

plt.ioff()
plt.show()
