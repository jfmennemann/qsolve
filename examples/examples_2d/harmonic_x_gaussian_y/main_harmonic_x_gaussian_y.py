from qsolve.solvers import SolverGPE2D

import mkl
import os

import h5py

import numpy as np

from scipy import constants

from scipy.interpolate import pchip_interpolate

import matplotlib.pyplot as plt

from figures.figure_main.figure_main import FigureMain

from potential_harmonic_x_gaussian_y import compute_external_potential

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
export_frames_figure_main = False

export_hdf5 = False

export_psi_of_times_analysis = False
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
N = 4000

omega_perp = 2 * np.pi * 1e3

t_final = 16e-3
dt = 0.0025e-3

m_Rb_87 = 87 * amu

Jx = 48
Jy = 256

x_min = -1.5e-6
x_max = +1.5e-6

y_min = -20e-6
y_max = +20e-6

# params_potential = {
#     "omega_x": omega_perp,
#     "V_ref_gaussian": 2 * hbar * omega_perp,
#     "sigma_gaussian": 1e-6
# }

parameters_potential = {'nu_x': [1e3, 'Hz'],
                        'V_ref_gaussian_y': [2 * hbar * omega_perp, 'J'],
                        'sigma_gaussian_y': [1e-6, 'm']}
# -------------------------------------------------------------------------------------------------

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

nr_frame_figure_main = 0

if export_frames_figure_main:

    if not os.path.exists(path_frames_figure_main):

        os.makedirs(path_frames_figure_main)
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# init solver and its potential
# =================================================================================================

solver = SolverGPE2D(m_atom=m_Rb_87,
                     a_s=5.24e-9,
                     omega_z=2*np.pi*1e3,
                     seed=1,
                     device='cuda:0',
                     num_threads_cpu=num_threads_cpu)

solver.init_grid(x_min=x_min,
                 x_max=x_max,
                 y_min=y_min,
                 y_max=y_max,
                 Jx=Jx,
                 Jy=Jy)

x = solver.x
y = solver.y

# =================================================================================================
# init time evolution
# =================================================================================================

solver.init_time_evolution(t_final=t_final, dt=dt)

times = solver.times

n_times = times.size
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
n_mod_times_analysis = 50

times_analysis = times[0::n_mod_times_analysis]

n_times_analysis = times_analysis.size

assert (np.abs(times_analysis[-1] - t_final) / t_final < 1e-14)
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# init control inputs
# =================================================================================================

# -------------------------------------------------------------------------------------------------
vec_t = np.array([0.0, 0.1, 0.2, 1.0]) * t_final
vec_u = np.array([1.0, 1.0, 0.0, 0.0])

u_of_times = pchip_interpolate(vec_t, vec_u, times)
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# compute ground state solution
# =================================================================================================

# -------------------------------------------------------------------------------------------------
# u_0 = u_of_times[0]

# solver.set_external_potential(t=0, u=u_0)

# solver.init_potential(Potential, params_potential)

solver.init_external_potential(compute_external_potential, parameters_potential)

solver.set_external_potential(t=0.0, u=u_of_times[0])
# -------------------------------------------------------------------------------------------------

psi_0 = solver.compute_ground_state_solution(n_atoms=N, n_iter=5000, tau=0.005e-3)

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

# =================================================================================================
# init figure
# =================================================================================================

params_figure_main = {
    "density_max":  2e+14,
    "density_z_eff_max": 400,
    "V_min": -0.5,
    "V_max": +4.5,
    "sigma_z_min": 0.2,
    "sigma_z_max": 0.6,
    "m_atom": m_Rb_87,
    "x_ticks": [-1.5, 0.0, 1.5],
    "y_ticks": [-20, 0, 20],
    "t_ticks": np.array([0, 4, 8, 12, 16])
}

# ---------------------------------------------------------------------------------------------
figure_main = FigureMain(x, y, times, params_figure_main)

figure_main.fig_control_inputs.update_u(u_of_times)

figure_main.fig_control_inputs.update_t(0.0)
# ---------------------------------------------------------------------------------------------


# =================================================================================================
# compute time evolution
# =================================================================================================

solver.set_u_of_times(u_of_times)

if export_psi_of_times_analysis:

    psi_of_times_analysis = np.zeros((n_times_analysis, Jx, Jy), dtype=np.complex128)

else:

    psi_of_times_analysis = None


n_inc = n_mod_times_analysis

nr_times_analysis = 0

n = 0

while True:

    t = times[n]

    data = eval_data(solver)

    if export_psi_of_times_analysis:

        psi_of_times_analysis[nr_times_analysis, :] = data.psi

    print('----------------------------------------------------------------------------------------')
    print('t:             {0:1.2f} / {1:1.2f}'.format(t / 1e-3, times[-1] / 1e-3))
    print('n:             {0:4d} / {1:4d}'.format(n, n_times))
    print()
    print('N:             {0:1.4f}'.format(data.N))
    print('----------------------------------------------------------------------------------------')
    print()

    # ---------------------------------------------------------------------------------------------
    figure_main.update_data(data)

    figure_main.fig_control_inputs.update_t(t)

    figure_main.redraw()

    if export_frames_figure_main:

        filepath = path_frames_figure_main + 'frame_' + str(nr_frame_figure_main).zfill(5) + '.png'

        figure_main.export(filepath)

        nr_frame_figure_main = nr_frame_figure_main + 1
    # ---------------------------------------------------------------------------------------------

    nr_times_analysis = nr_times_analysis + 1

    if n < n_times - n_inc:

        solver.propagate_gpe(n_start=n, n_inc=n_inc, mue_shift=mue_0)

        n = n + n_inc

    else:

        break

    n = n + n_inc


if export_hdf5:

    # ---------------------------------------------------------------------------------------------
    # Create file

    f_hdf5 = h5py.File(filepath_f_hdf5, "w")

    # Create file, fail if exists
    # f_hdf5 = h5py.File(filepath_f_hdf5, "x")
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    f_hdf5.create_dataset("hbar", data=hbar)

    f_hdf5.create_dataset("N", data=N)
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    tmp = f_hdf5.create_group("time_evolution")

    if export_psi_of_times_analysis:

        tmp.create_dataset("psi_of_times_analysis", data=psi_of_times_analysis, dtype=np.complex128)

    tmp.create_dataset("times", data=times)
    tmp.create_dataset("dt", data=dt)
    tmp.create_dataset("n_mod_times_analysis", data=n_mod_times_analysis)

    tmp.create_dataset("times_analysis", data=times_analysis)
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    f_hdf5.close()
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    f_hdf5 = h5py.File(filepath_f_hdf5, 'r')

    list_all_items = True

    if list_all_items:

        def print_attrs(name, obj):

            print(name)
            for key, val in obj.attrs.items():
                print("    %s: %s" % (key, val))


        f_hdf5.visititems(print_attrs)

        print()
        print()
    # ---------------------------------------------------------------------------------------------

plt.ioff()
plt.show()
