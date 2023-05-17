import numpy as np


def eval_data(solver):

    index_center_x = solver.index_center_x
    index_center_y = solver.index_center_y
    index_center_z = solver.index_center_z

    data = type('', (), {})()

    # ---------------------------------------------------------------------------------------------
    V = solver.V

    V_x = V[:, index_center_y, index_center_z].squeeze()

    data.V_x = V_x

    data.V_y = V[index_center_x, :, index_center_z].squeeze()
    data.V_z = V[index_center_x, index_center_y, :].squeeze()
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    psi = solver.psi

    psi_x = psi[:, index_center_y, index_center_z].squeeze()
    psi_y = psi[index_center_x, :, index_center_z].squeeze()
    psi_z = psi[index_center_x, index_center_y, :].squeeze()

    psi_xz = psi[:, index_center_y, :].squeeze()
    psi_xy = psi[:, :, index_center_z].squeeze()

    data.psi = psi

    data.psi_x = psi_x
    data.psi_y = psi_y
    data.psi_z = psi_z

    data.psi_xz = psi_xz
    data.psi_xy = psi_xy
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    density_max = np.max(np.abs(psi)**2)

    data.density_x = np.abs(psi_x) ** 2
    data.density_y = np.abs(psi_y) ** 2
    data.density_z = np.abs(psi_z) ** 2

    data.density_xz = np.abs(psi_xz) ** 2 / density_max
    data.density_xy = np.abs(psi_xy) ** 2 / density_max
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    data.real_part_x = np.real(psi_x)
    data.real_part_y = np.real(psi_y)
    data.real_part_z = np.real(psi_z)

    data.imag_part_x = np.imag(psi_x)
    data.imag_part_y = np.imag(psi_y)
    data.imag_part_z = np.imag(psi_z)
    # ---------------------------------------------------------------------------------------------

    data.N = solver.compute_n_atoms()

    return data
