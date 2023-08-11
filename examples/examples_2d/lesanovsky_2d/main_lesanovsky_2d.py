from qsolve.solvers import SolverGPE2D
from qsolve.grids import Grid2D
from qsolve.units import Units
from qsolve.figures import FigureEigenstatesLSE2D

from potential_lesanovsky_2d import PotentialLesanovsky2D

from figures import FigureMain2D


import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import mkl

import numpy as np

import scipy

import matplotlib.pyplot as plt

import h5py

from evaluation import eval_data


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

amu = scipy.constants.physical_constants["atomic mass constant"][0]  # atomic mass unit
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
# close figures from previous simulation

plt.close('all')
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
temperature = True

quickstart = False

visualization = True

time_of_flight = True
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
export_frames_figure_main = False
export_frames_figure_tof = False

export_hdf5 = False

export_psi_of_times_analysis = False
# -------------------------------------------------------------------------------------------------


# =================================================================================================
device = 'cuda:0'
# device='cpu'

n_atoms = 3500

u1_final = 0.56

if quickstart:

    gamma_tilt_ref = 0.0

    # xi_ext = 0.420
    xi_ext = 0.350

else:

    gamma_tilt_ref = 4.1e-26

    xi_ext = 0.0


t_final = 80e-3

if temperature:

    T = 20e-9

else:

    T = 0e-9

m_Rb_87 = 87 * amu

m_atom = m_Rb_87

a_s = 5.24e-9

Jx = 56
Jy = 240

dt = 0.0025e-3

n_mod_times_analysis = 100

x_min = -2.8e-6
x_max = +2.8e-6

y_min = -60e-6
y_max = +60e-6

parameters_potential = {
    'm_atom': m_atom,
    'nu_perp': 3e3,
    'nu_para': 22.5,
    'nu_delta_detuning': -50e3,
    'nu_trap_bottom': 1216e3,
    'nu_rabi_ref': 575e3,
    'gamma_tilt_ref': gamma_tilt_ref}

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

nr_frame_figure_main = 0

if export_frames_figure_main:

    if not os.path.exists(path_frames_figure_main):

        os.makedirs(path_frames_figure_main)
# -------------------------------------------------------------------------------------------------


units = Units.solver_units(m_atom, dim=2)

grid = Grid2D(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, Jx=Jx, Jy=Jy)

potential = PotentialLesanovsky2D(grid=grid, units=units, device=device, parameters=parameters_potential)

solver = SolverGPE2D(
    units=units,
    grid=grid,
    potential=potential,
    device=device,
    m_atom=m_Rb_87,
    a_s=5.24e-9,
    omega_z=2*np.pi*parameters_potential['nu_perp'],
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

assert (np.abs(times_analysis[-1] - t_final) / t_final < 1e-14)
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# init control inputs
# =================================================================================================

if quickstart:

    t_idle = 5e-3

    t_phase_imprint_part_1 = 1.5e-3
    t_phase_imprint_part_2 = 1.5e-3

    t0 = 0.0
    t1 = t0 + t_idle
    t2 = t1 + t_phase_imprint_part_1
    t3 = t2 + t_phase_imprint_part_2

    vec_t = np.array([t0, t1, t2, t3])

    vec_u2 = np.array([0, 0, 1, 0])

    u1_of_times = u1_final * np.ones_like(times)

    # u2_of_times = np.interp(times, vec_t, vec_u2)

    u2_of_times = np.zeros_like(times)

else:

    t_ramp_up = 21.5e-3

    t_phase_imprint_part_1 = 1.5e-3
    t_phase_imprint_part_2 = 1.5e-3
    t_ramp_down = 3.0e-3
    t_help = 10.0e-3

    t0 = 0.0
    t1 = t0 + t_ramp_up
    t2 = t1 + t_phase_imprint_part_1
    t3 = t2 + t_phase_imprint_part_2
    t4 = t3 + t_ramp_down
    t5 = t4 + t_help

    u1_0 = 0.0
    u1_1 = 0.65
    u1_2 = 0.65
    u1_3 = 0.65
    u1_4 = u1_final
    u1_5 = u1_final

    vec_t = np.array([t0, t1, t2, t3, t4, t5])

    vec_u1 = np.array([u1_0, u1_1, u1_2, u1_3, u1_4, u1_5])
    vec_u2 = np.array([0, 0, 1, 0, 0, 0])

    u1_of_times = np.interp(times, vec_t, vec_u1)
    u2_of_times = np.interp(times, vec_t, vec_u2)

u_of_times = np.zeros((2, n_times))

u_of_times[0, :] = u1_of_times
u_of_times[1, :] = u2_of_times


# =================================================================================================
# set external potential
# =================================================================================================

solver.set_external_potential(t=0.0, u=u_of_times[0])


# =================================================================================================
# compute ground state solution
# =================================================================================================

psi_0 = solver.compute_ground_state_solution(n_atoms=n_atoms, n_iter=5000, tau=0.005e-3)

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
    "V_min": 0,
    "V_max": 4,
    "sigma_z_min": 0.2,
    "sigma_z_max": 0.6,
    "m_atom": m_Rb_87,
    "x_ticks": [-2, -1, 0, 1, 2],
    "y_ticks": [-60, -40, -20, 0, 20, 40, 60],
    "t_ticks": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80])
}

# -------------------------------------------------------------------------------------------------
figure_main = FigureMain2D(grid.x, grid.y, times, params_figure_main)

figure_main.fig_control_inputs.update_u(u_of_times)

figure_main.fig_control_inputs.update_t(0.0)
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# init figure_eigenstates_lse_2d
# =================================================================================================

params_figure_eigenstates_lse_2d = {
    "density_max":  2e+14,
    "density_z_eff_max": 400,
    "V_min": 0,
    "V_max": 4,
    "sigma_z_min": 0.2,
    "sigma_z_max": 0.6,
    "m_atom": m_Rb_87,
    "x_ticks": [-2, -1, 0, 1, 2],
    "y_ticks": [-60, -40, -20, 0, 20, 40, 60],
    "t_ticks": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80])
}

# -------------------------------------------------------------------------------------------------
figure_eigenstates_lse_2d = FigureEigenstatesLSE2D(psi_0, solver.V, grid.x, grid.y, params_figure_eigenstates_lse_2d)
# -------------------------------------------------------------------------------------------------


# =================================================================================================
# thermal state sampling
# =================================================================================================

if T > 0:

    solver.init_sgpe_z_eff(
        T_temp_des=T,
        mue_des=mue_0,
        gamma=0.1,
        dt=dt,
        filter_y1=-45e-6,
        filter_y2=+45e-6,
        filter_y_s=1.0e-6
    )

    n_sgpe_max = 10000

    n_sgpe_inc = 1000

    n_sgpe = 0

    while n_sgpe < n_sgpe_max:

        data = eval_data(solver, grid)

        print('------------------------------------------------------------------------------------')
        print('n_sgpe: {0:4d} / {1:4d}'.format(n_sgpe, n_sgpe_max))
        print()
        print('N:      {0:1.4f}'.format(data.N))
        print('------------------------------------------------------------------------------------')
        print()

        if visualization:

            # -------------------------------------------------------------------------------------
            figure_main.update_data(data)

            figure_main.redraw()
            # -------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        # apply thermal state sampling process via sgpe for n_sgpe_inc time steps

        solver.propagate_sgpe_z_eff(n_inc=n_sgpe_inc)
        # -----------------------------------------------------------------------------------------

        n_sgpe = n_sgpe + n_sgpe_inc


# =================================================================================================
# compute time evolution
# =================================================================================================

data_time_evolution = type('', (), {})()

data_time_evolution.global_phase_difference_of_times_analysis = np.zeros(shape=(n_times_analysis,), dtype=np.float64)
data_time_evolution.number_imbalance_of_times_analysis = np.zeros(shape=(n_times_analysis,), dtype=np.float64)

# print(times_analysis)
# print(n_times_analysis)
# print(times_analysis.size)
# input()

data_time_evolution.times_analysis = times_analysis

if export_psi_of_times_analysis:

    psi_of_times_analysis = np.zeros(shape=(n_times_analysis, Jx, Jy), dtype=np.complex128)

else:

    psi_of_times_analysis = None


n_inc = n_mod_times_analysis

nr_times_analysis = 0

n = 0

while True:

    t = times[n]

    data = eval_data(solver, grid)

    data_time_evolution.global_phase_difference_of_times_analysis[nr_times_analysis] = data.global_phase_difference
    data_time_evolution.number_imbalance_of_times_analysis[nr_times_analysis] = data.number_imbalance

    data_time_evolution.nr_times_analysis = nr_times_analysis

    if export_psi_of_times_analysis:

        psi_of_times_analysis[nr_times_analysis, :] = data.psi

    print('t: {0:1.2f} / {1:1.2f}, n: {2:4d} / {3:4d}, n_atoms:{4:4f}'.format(t / 1e-3, times[-1] / 1e-3, n, n_times, n_atoms))

    # ---------------------------------------------------------------------------------------------
    figure_main.update_data(data)

    figure_main.fig_control_inputs.update_t(t)

    figure_main.update_data_time_evolution(data_time_evolution)

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


if export_hdf5:

    # ---------------------------------------------------------------------------------------------
    # Create file

    f_hdf5 = h5py.File(filepath_f_hdf5, "w")

    # Create file, fail if exists
    # f_hdf5 = h5py.File(filepath_f_hdf5, "x")
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    f_hdf5.create_dataset("hbar", data=hbar)

    f_hdf5.create_dataset("n_atoms", data=n_atoms)
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
