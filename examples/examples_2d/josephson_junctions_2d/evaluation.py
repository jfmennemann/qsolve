import numpy as np

from time import time


def get_indices_x1_x2(V_x, Jx, index_center_x):

    index_x1 = -1

    for jx in range(1, index_center_x + 1):

        if V_x[jx - 1] >= V_x[jx] and V_x[jx] <= V_x[jx + 1]:
            index_x1 = jx
            break

    assert (index_x1 > 0)

    index_x2 = -1

    for jx in range(index_center_x, Jx):

        if V_x[jx - 1] >= V_x[jx] and V_x[jx] <= V_x[jx + 1]:
            index_x2 = jx
            break

    assert (index_x2 > 0)

    return index_x1, index_x2


def compute_psi_complete(psi, fill_boundaries=False):

    Jx = psi.shape[0]
    Jy = psi.shape[1]
    Jz = psi.shape[2]

    psi_complete = np.zeros((Jx+1, Jy+1, Jz+1), dtype=np.complex128)

    psi_complete[:Jx, :Jy, :Jz] = psi

    if fill_boundaries:

        psi_complete[-1, :, :] = psi_complete[0, :, :]
        psi_complete[:, -1, :] = psi_complete[:, 0, :]
        psi_complete[:, :, -1] = psi_complete[:, :, 0]

    return psi_complete


def compute_phase_difference(psi):

    psi_complete = compute_psi_complete(psi)

    psi_complete_flip_x = np.flip(psi_complete, 0)

    tmp = psi_complete * np.conj(psi_complete_flip_x)

    phase_difference_complete = np.angle(tmp)

    phase_difference = phase_difference_complete[:-1, :-1, :-1]

    return phase_difference


def compute_phase_difference_z_x1_x2(psi_z_x1, psi_z_x2):

    phase_difference_z_x1_x2 = np.angle(psi_z_x1 * np.conj(psi_z_x2))

    return phase_difference_z_x1_x2


def compute_global_phase_difference(psi, index_center_x):

    psi_complete = compute_psi_complete(psi)

    psi_complete_flip_x = np.flip(psi_complete, 0)

    tmp = psi_complete * np.conj(psi_complete_flip_x)

    tmp = np.sum(tmp[:index_center_x, :, :])

    delta_phi = np.angle(tmp)

    return delta_phi


def compute_number_imbalance(psi, dx, dy, dz, index_center_x):

    density = np.real(psi * np.conj(psi))

    N = (dx * dy * dz) * np.sum(density)

    density_1 = density[:index_center_x+1, :, :]
    density_2 = density[index_center_x:, :, :]

    N_1 = (dx * dy * dz) * np.sum(density_1)
    N_2 = (dx * dy * dz) * np.sum(density_2)

    number_imbalance = (N_2 - N_1) / N

    return number_imbalance


def eval_data(solver):

    dx = solver.dx
    dy = solver.dy
    dz = solver.dz

    Jx = solver.Jx

    index_center_x = solver.index_center_x
    index_center_y = solver.index_center_y
    index_center_z = solver.index_center_z

    data = type('', (), {})()

    # ---------------------------------------------------------------------------------------------
    V = solver.V

    V_x = V[:, index_center_y, index_center_z].squeeze()

    index_x1, index_x2 = get_indices_x1_x2(V_x, Jx, index_center_x)

    data.V_x = V_x

    data.V_y_x1 = V[index_x1, :, index_center_z].squeeze()
    data.V_y_x2 = V[index_x2, :, index_center_z].squeeze()

    data.V_z_x1 = V[index_x1, index_center_y, :].squeeze()
    data.V_z_x2 = V[index_x2, index_center_y, :].squeeze()
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    psi = solver.psi

    psi_x = psi[:, index_center_y, index_center_z].squeeze()

    psi_y_x1 = psi[index_x1, :, index_center_z].squeeze()
    psi_y_x2 = psi[index_x2, :, index_center_z].squeeze()

    psi_z_x1 = psi[index_x1, index_center_y, :].squeeze()
    psi_z_x2 = psi[index_x2, index_center_y, :].squeeze()

    psi_xz = psi[:, index_center_y, :].squeeze()
    psi_xy = psi[:, :, index_center_z].squeeze()

    data.psi = psi

    data.psi_x = psi_x

    data.psi_y_x1 = psi_y_x1
    data.psi_y_x2 = psi_y_x2

    data.psi_z_x1 = psi_z_x1
    data.psi_z_x2 = psi_z_x2

    data.psi_xz = psi_xz
    data.psi_xy = psi_xy
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    density_max = np.max(np.abs(psi)**2)

    data.density_x = np.abs(psi_x) ** 2

    data.density_y_x1 = np.abs(psi_y_x1) ** 2
    data.density_y_x2 = np.abs(psi_y_x2) ** 2

    data.density_z_x1 = np.abs(psi_z_x1) ** 2
    data.density_z_x2 = np.abs(psi_z_x2) ** 2

    data.density_xz = np.abs(psi_xz) ** 2 / density_max
    data.density_xy = np.abs(psi_xy) ** 2 / density_max
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    amplitude_psi_imag = np.max(np.abs(np.imag(psi)))

    psi_imag_xz = np.imag(psi_xz)
    psi_imag_xy = np.imag(psi_xy)

    if amplitude_psi_imag > 0:

        psi_imag_xz = psi_imag_xz / amplitude_psi_imag
        psi_imag_xy = psi_imag_xy / amplitude_psi_imag

    data.psi_imag_xz = psi_imag_xz
    data.psi_imag_xy = psi_imag_xy
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    data.N = solver.compute_n_atoms()
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    data.global_phase_difference = compute_global_phase_difference(psi, index_center_x)
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    data.number_imbalance = compute_number_imbalance(psi, dx, dy, dz, index_center_x)
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    phase_difference = compute_phase_difference(psi)

    phase_difference_xz = phase_difference[:, index_center_y, :].squeeze()
    phase_difference_xy = phase_difference[:, :, index_center_z].squeeze()

    data.phase_difference_xz = (data.density_xz / np.max(data.density_xz)) * phase_difference_xz
    data.phase_difference_xy = (data.density_xy / np.max(data.density_xy)) * phase_difference_xy
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    data.phase_difference_z_x1_x2 = compute_phase_difference_z_x1_x2(psi_z_x1, psi_z_x2)
    # ---------------------------------------------------------------------------------------------

    return data


def eval_data_tof(solver):

    data_tof = type('', (), {})()



    # time_1 = time()
    # psi_f_tof_free_schroedinger = solver.psi_f_tof_free_schroedinger
    # time_2 = time()
    # print('elapsed_time: {0:f}'.format(time_2 - time_1))
    # print()

    # ---------------------------------------------------------------------------------------------
    psi_tof_free_gpe_xy = solver.trace_psi_tof_free_gpe_xy()
    psi_tof_free_gpe_xz = solver.trace_psi_tof_free_gpe_xz()

    density_tof_free_gpe_xy = np.abs(psi_tof_free_gpe_xy) ** 2
    density_tof_free_gpe_xz = np.abs(psi_tof_free_gpe_xz) ** 2

    data_tof.density_xy_tof_gpe = density_tof_free_gpe_xy / np.max(density_tof_free_gpe_xy)
    data_tof.density_xz_tof_gpe = density_tof_free_gpe_xz / np.max(density_tof_free_gpe_xz)

    data_tof.density_xy_mask_tof_gpe = data_tof.density_xy_tof_gpe > 1e-6
    data_tof.density_xz_mask_tof_gpe = data_tof.density_xz_tof_gpe > 1e-6
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    # data_tof.spectrum_abs_xz_tof_gpe = solver.compute_spectrum_abs_xz('psi_tof_gpe', rescaling=True)
    # data_tof.spectrum_abs_xy_tof_gpe = solver.compute_spectrum_abs_xy('psi_tof_gpe', rescaling=True)
    #
    # data_tof.spectrum_abs_xz_mask_tof_gpe = data_tof.spectrum_abs_xz_tof_gpe > 1e-2
    # data_tof.spectrum_abs_xy_mask_tof_gpe = data_tof.spectrum_abs_xy_tof_gpe > 1e-2
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    psi_f_tof_free_schroedinger_xy = solver.trace_psi_f_tof_free_schroedinger_xy()
    psi_f_tof_free_schroedinger_xz = solver.trace_psi_f_tof_free_schroedinger_xz()

    density_f_tof_free_schroedinger_xy = np.abs(psi_f_tof_free_schroedinger_xy) ** 2
    density_f_tof_free_schroedinger_xz = np.abs(psi_f_tof_free_schroedinger_xz) ** 2

    data_tof.density_xy_tof_final = density_f_tof_free_schroedinger_xy / np.max(density_f_tof_free_schroedinger_xy)
    data_tof.density_xz_tof_final = density_f_tof_free_schroedinger_xz / np.max(density_f_tof_free_schroedinger_xz)
    # ---------------------------------------------------------------------------------------------

    return data_tof
