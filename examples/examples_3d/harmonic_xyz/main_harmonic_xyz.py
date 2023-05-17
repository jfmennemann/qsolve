from qsolve.solvers import SolverGPE3D

import mkl

import os

import numpy as np

from scipy import constants

import matplotlib.pyplot as plt

from figures.figure_main.figure_main import FigureMain

from potential_harmonic_xyz import compute_external_potential

from evaluation import eval_data


# -------------------------------------------------------------------------------------------------
num_threads_cpu = 8

os.environ["OMP_NUM_THREADS"] = str(num_threads_cpu)
os.environ["MKL_NUM_THREADS"] = str(num_threads_cpu)

mkl.set_num_threads(num_threads_cpu)

assert(mkl.get_max_threads() == num_threads_cpu)
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
pi = constants.pi

hbar = constants.hbar

amu = constants.physical_constants["atomic mass constant"][0]  # atomic mass unit

mu_B = constants.physical_constants["Bohr magneton"][0]

k_B = constants.Boltzmann
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# close figures from previous simulation

plt.close('all')
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
visualization = True
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
export_frames_figure_main = False
export_frames_figure_tof = False

export_hdf5 = False

export_psi_of_times_analysis = False
# -------------------------------------------------------------------------------------------------


# =================================================================================================
N = 3500

m_Rb_87 = 87 * amu

m_atom = m_Rb_87

a_s = 5.24e-9

x_min = -5e-6
x_max = +5e-6

y_min = -5e-6
y_max = +5e-6

z_min = -10e-6
z_max = +10e-6

Jx = 50
Jy = 50
Jz = 100

t_final = 4e-3

dt = 0.001e-3

n_mod_times_analysis = 100

parameters_potential = {
    'nu_x': (200, 'Hz'),
    'nu_y': (100, 'Hz'),
    'nu_z': (50, 'Hz')
}

params_figure_main = {
    'm_atom': m_atom,
    'density_min': -0.2e20,
    'density_max': +2.2e20,
    'V_min': -1.0,
    'V_max': 11.0,
    'abs_z_restr': 30e-6
}
# =================================================================================================

# -------------------------------------------------------------------------------------------------
simulation_id = 'test'

simulation_id = simulation_id.replace(".", "_")
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# hdf5

path_f_hdf5 = "./data_hdf5/"

filepath_f_hdf5 = path_f_hdf5 + simulation_id + ".hdf5"
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# frames

path_frames_figure_main = "./frames/frames_figure_main/" + simulation_id + "/"
path_frames_figure_tof = "./frames/frames_figure_tof/" + simulation_id + "/"

nr_frame_figure_main = 0
nr_frame_figure_tof = 0

if export_frames_figure_main:

    if not os.path.exists(path_frames_figure_main):

        os.makedirs(path_frames_figure_main)

if export_frames_figure_tof:

    if not os.path.exists(path_frames_figure_tof):

        os.makedirs(path_frames_figure_tof)
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# init solver
# =================================================================================================

solver = SolverGPE3D(m_atom=m_Rb_87,
                     a_s=a_s,
                     seed=1,
                     device='cuda:0',
                     # device='cpu',
                     num_threads_cpu=num_threads_cpu)

solver.init_grid(x_min=x_min,
                 x_max=x_max,
                 y_min=y_min,
                 y_max=y_max,
                 z_min=z_min,
                 z_max=z_max,
                 Jx=Jx,
                 Jy=Jy,
                 Jz=Jz)

x = solver.x
y = solver.y
z = solver.z


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

assert (np.abs(times_analysis[-1] - t_final) / t_final < 1e-14)
# -------------------------------------------------------------------------------------------------

# =================================================================================================
# init control inputs
# =================================================================================================

u1_of_times = np.ones_like(times)
u2_of_times = np.ones_like(times)

u_of_times = np.zeros((2, n_times))

u_of_times[0, :] = u1_of_times
u_of_times[1, :] = u2_of_times


# =================================================================================================
# init external potential
# =================================================================================================

solver.init_external_potential(compute_external_potential, parameters_potential)

solver.set_external_potential(t=0.0, u=u_of_times[0])


# =================================================================================================
# compute ground state solution
# =================================================================================================

# -------------------------------------------------------------------------------------------------
psi_0 = solver.compute_ground_state_solution(n_atoms=N, n_iter=5000, tau=0.005e-3, adaptive_tau=True)

solver.psi = psi_0

N_0 = solver.compute_n_atoms()
mue_0 = solver.compute_chemical_potential()
E_0 = solver.compute_total_energy()

print('N_0 = {:1.16e}'.format(N_0))
print('mue_0 / h: {0:1.6} kHz'.format(mue_0 / (1e3 * (2 * pi * hbar))))
print('E_0 / (N_0*h): {0:1.6} kHz'.format(E_0 / (1e3 * (2 * pi * hbar * N_0))))
print()
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# init figure
# =================================================================================================

# -------------------------------------------------------------------------------------------------
figure_main = FigureMain(x, y, z, times, params_figure_main)

figure_main.fig_control_inputs.update_u(u1_of_times, u2_of_times)

figure_main.fig_control_inputs.update_t(0.0)
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
data = eval_data(solver)

figure_main.update_data(data)

figure_main.redraw()
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# compute time evolution
# =================================================================================================

# -------------------------------------------------------------------------------------------------
data_time_evolution = type('', (), {})()

if export_psi_of_times_analysis:

    data_time_evolution.psi_of_times_analysis = np.zeros((n_times_analysis, Jx, Jy, Jz), dtype=np.complex128)

else:

    data_time_evolution.psi_of_times_analysis = None

data_time_evolution.global_phase_difference_of_times_analysis = np.zeros((n_times_analysis,), dtype=np.float64)
data_time_evolution.number_imbalance_of_times_analysis = np.zeros((n_times_analysis,), dtype=np.float64)

data_time_evolution.times_analysis = times_analysis
# -------------------------------------------------------------------------------------------------

n_inc = n_mod_times_analysis

nr_times_analysis = 0

stop = False

n = 0

while True:

    t = times[n]

    data = eval_data(solver)

    if export_psi_of_times_analysis:

        data_time_evolution.psi_of_times_analysis[nr_times_analysis, :] = data.psi

    data_time_evolution.nr_times_analysis = nr_times_analysis

    print('----------------------------------------------------------------------------------------')
    print('t: {0:1.2f} / {1:1.2f}'.format(t / 1e-3, times[-1] / 1e-3))
    print('n: {0:4d} / {1:4d}'.format(n, n_times))
    print()
    print('N: {0:1.4f}'.format(data.N))
    print('----------------------------------------------------------------------------------------')
    print()

    if visualization:

        # -----------------------------------------------------------------------------------------
        figure_main.update_data(data)

        # figure_main.update_data_time_evolution(data_time_evolution)

        figure_main.fig_control_inputs.update_t(t)

        figure_main.redraw()

        if export_frames_figure_main:

            filepath = path_frames_figure_main + 'frame_' + str(nr_frame_figure_main).zfill(5) + '.png'

            figure_main.export(filepath)

            nr_frame_figure_main = nr_frame_figure_main + 1
        # -----------------------------------------------------------------------------------------

    nr_times_analysis = nr_times_analysis + 1

    if n < n_times - n_inc:

        solver.propagate_gpe(times=times, u_of_times=u_of_times, n_start=n, n_inc=n_inc, mue_shift=mue_0)

        n = n + n_inc

    else:

        break


plt.ioff()
plt.show()
