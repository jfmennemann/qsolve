from qsolve.solvers import SolverGPE1D
from qsolve.figures import FigureMain1D

from potential_harmonic_trap_1d import compute_external_potential

import mkl

import os

import numpy as np

import scipy

from scipy.interpolate import pchip_interpolate

import matplotlib.pyplot as plt


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

x_min = -40e-6
x_max = +40e-6

Jx = 128

t_final = 8e-3

dt = 0.0005e-3

n_mod_times_analysis = 100

n_control_inputs = 1
# =================================================================================================

parameters_potential = {'nu_start': [40, "Hz"],
                        'nu_final': [20, "Hz"]}

parameters_figure_main = {'density_min': -20,
                          'density_max': +220,
                          'V_min': -0.5,
                          'V_max': +4.5,
                          'x_ticks': np.array([-40, -20, 0, 20, 40]),
                          't_ticks': np.array([0, 2, 4, 6, 8]),
                          'n_control_inputs': n_control_inputs}
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
                     num_threads_cpu=num_threads_cpu)

# =================================================================================================
# init spatial grid
# =================================================================================================

solver.init_grid(x_min=x_min, x_max=x_max, Jx=Jx)

# =================================================================================================
# init time evolution
# =================================================================================================

# -------------------------------------------------------------------------------------------------
solver.init_time_evolution(t_final=t_final, dt=dt)

times = solver.times

n_times = times.size
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

u1_of_times = pchip_interpolate(np.array([0, 0.1, 0.2, 1]) * t_final, np.array([0.0, 0.0, 1.0, 1.0]), times)

u_of_times[0, :] = u1_of_times


# =================================================================================================
# compute ground state solution
# =================================================================================================

solver.init_external_potential(compute_external_potential, parameters_potential)

solver.set_external_potential(t=0.0, u=u_of_times[0])

psi_0, vec_res, vec_iter = solver.compute_ground_state_solution(n_atoms=n_atoms,
                                                                n_iter=5000,
                                                                tau=0.001e-3,
                                                                adaptive_tau=True,
                                                                return_residuals=True)

# =================================================================================================
# set wave function to ground state solution
# =================================================================================================

solver.psi = psi_0

N_0 = solver.compute_n_atoms()
mue_0 = solver.compute_chemical_potential()
E_total_0 = solver.compute_total_energy()
E_kinetic_0 = solver.compute_kinetic_energy()
E_interaction_0 = solver.compute_interaction_energy()
E_potential_0 = solver.compute_potential_energy()

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
figure_main = FigureMain1D(solver.x, times, parameters_figure_main)

figure_main.fig_control_inputs.update_u(u_of_times)

figure_main.fig_control_inputs.update_t(0.0)
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
figure_main.update_data(solver.psi, solver.V)

figure_main.redraw()
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# show convergence of ground state computation
# =================================================================================================

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
    t = solver.t

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

        solver.propagate_gpe(u_of_times=u_of_times, n_start=n, n_inc=n_inc, mue_shift=mue_0)
        # solver.propagate_gpe(u_of_times=u_of_times, n_start=n, n_inc=n_inc)

        n = n + n_inc

    else:

        break

plt.ioff()
plt.show()
