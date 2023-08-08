import numpy as np


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

    psi_complete = np.zeros(shape=(Jx+1, Jy+1), dtype=np.complex128)

    psi_complete[:Jx, :Jy] = psi

    if fill_boundaries:

        psi_complete[-1, :, :] = psi_complete[0, :]
        psi_complete[:, -1, :] = psi_complete[:, 0]

    return psi_complete


def compute_phase_difference(psi):

    psi_complete = compute_psi_complete(psi)

    psi_complete_flip_x = np.flip(psi_complete, axis=0)

    tmp = psi_complete * np.conj(psi_complete_flip_x)

    phase_difference_complete = np.angle(tmp)

    phase_difference = phase_difference_complete[:-1, :-1]

    return phase_difference


def compute_phase_difference_y_x1_x2(psi_y_x1, psi_y_x2):

    phase_difference_y_x1_x2 = np.angle(psi_y_x1 * np.conj(psi_y_x2))

    return phase_difference_y_x1_x2


def compute_global_phase_difference(psi, index_center_x):

    psi_complete = compute_psi_complete(psi)

    psi_complete_flip_x = np.flip(psi_complete, axis=0)

    tmp = psi_complete * np.conj(psi_complete_flip_x)

    tmp = np.sum(tmp[:index_center_x, :])

    delta_phi = np.angle(tmp)

    return delta_phi


def compute_number_imbalance(psi, dx, dy, index_center_x):

    density = np.real(psi * np.conj(psi))

    N = (dx * dy) * np.sum(density)

    density_1 = density[:index_center_x+1, :]
    density_2 = density[index_center_x:, :]

    N_1 = (dx * dy) * np.sum(density_1)
    N_2 = (dx * dy) * np.sum(density_2)

    number_imbalance = (N_2 - N_1) / N

    return number_imbalance


def eval_data(solver, grid):

    # ---------------------------------------------------------------------------------------------
    Jx = grid.Jx

    dx = grid.dx
    dy = grid.dy

    index_center_x = grid.index_center_x
    index_center_y = grid.index_center_y
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    N = solver.compute_n_atoms()
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    V = solver.V

    V_x = V[:, index_center_y].squeeze()
    V_y = V[index_center_x, :].squeeze()
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    psi = solver.psi

    psi_x = psi[:, index_center_y].squeeze()
    psi_y = psi[index_center_x, :].squeeze()

    density = np.abs(psi) ** 2

    density_x = np.abs(psi_x) ** 2
    density_y = np.abs(psi_y) ** 2

    real_part_x = np.real(psi_x)
    real_part_y = np.real(psi_y)

    imag_part_x = np.imag(psi_x)
    imag_part_y = np.imag(psi_y)
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    index_x1, index_x2 = get_indices_x1_x2(V_x, Jx, index_center_x)

    V_y_x1 = V[index_x1, :].squeeze()
    V_y_x2 = V[index_x2, :].squeeze()

    psi_y_x1 = psi[index_x1, :].squeeze()
    psi_y_x2 = psi[index_x2, :].squeeze()

    density_y_x1 = np.abs(psi_y_x1)**2
    density_y_x2 = np.abs(psi_y_x2)**2
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    data = type('', (), {})()

    data.N = N

    data.psi = psi

    data.psi_x = psi_x
    data.psi_y = psi_y

    data.density = density

    data.density_x = density_x
    data.density_y = density_y

    data.real_part_x = real_part_x
    data.real_part_y = real_part_y

    data.imag_part_x = imag_part_x
    data.imag_part_y = imag_part_y

    data.V_x = V_x
    data.V_y = V_y

    data.V_y_x1 = V_y_x1
    data.V_y_x2 = V_y_x2

    data.density_y_x1 = density_y_x1
    data.density_y_x2 = density_y_x2
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    data.number_imbalance = compute_number_imbalance(psi, dx, dy, index_center_x)
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    data.global_phase_difference = compute_global_phase_difference(psi, index_center_x)
    # ---------------------------------------------------------------------------------------------

    print('number_imbalance: {0:f}'.format(data.number_imbalance))
    print('global_phase_difference: {0:f}'.format(data.global_phase_difference))
    print()

    return data
